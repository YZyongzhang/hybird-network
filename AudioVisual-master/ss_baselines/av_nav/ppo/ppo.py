#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itsdangerous import NoneAlgorithm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from habitat import Config, logger
from zmq import device
from torch.distributions import Categorical
import queue
import random
from torch.utils.data import DataLoader, dataset, TensorDataset

EPS_PPO = 1e-5
from .policy import grad_reverse
import numpy as np
from PIL import Image
import time


class PPO(nn.Module):

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)
        self.actor_critic.net.train()
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                # 就是PPO cliped的Loss！
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ)
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (0.5 * torch.max(value_losses, value_losses_clipped).mean())
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self):
        pass


class WyxPPO(nn.Module):

    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr=None, regressor_lr=None, classifier_lr=None, eps=None, max_grad_norm=None, use_clipped_value_loss=True, use_normalized_advantage=True, config=None):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.config = config
        ordinary_params = []
        classifier_params = []
        regressor_params = []
        for pname, p in actor_critic.named_parameters():
            if pname.find("classifier") != -1:
                classifier_params += [p]
            elif pname.find("regressor") != -1:
                regressor_params += [p]
            else:
                ordinary_params += [p]

        self.optimizer = optim.Adam([{'params': ordinary_params}, {'params': regressor_params, 'lr': regressor_lr if regressor_lr is not None else lr}, {'params': classifier_params, 'lr': classifier_lr if classifier_lr is not None else lr}], lr=lr, eps=eps)
        """
        if len(classifier_params) > 0: # 需要去检查是否启用classifier
            self.classifier_optimizer = optim.Adam(classifier_params,
                                                   lr=classifier_lr if classifier_lr is not None else lr, eps=eps)
        """
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        # if self.config.use_buffer_train or self.config.use_buffer_more:
        #     self.buffer_list_spect = list()
        #     self.buffer_list_label = list()
        #     self.buffer_list_depth = list()

        if self.config.use_buffer_train or self.config.use_buffer_more:
            self.buffer_spect = torch.zeros((self.config.buffer_maxlen, ) + actor_critic.observation_space.spaces["spectrogram"].shape, device="cpu")
            self.buffer_label = torch.zeros((self.config.buffer_maxlen, 1), dtype=torch.long, device="cpu")
            self.buffer_depth = torch.zeros((self.config.buffer_maxlen, ) + actor_critic.observation_space.spaces["depth"].shape, device="cpu")
            self.buffer_pointer = 0

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts, lambda_grad=1.0):
        # whc 于2.13 2:00添加
        self.actor_critic.net.train()

        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        classifier_loss_epoch = 0
        classifier_entropy_epoch = 0
        classifier_acc_epoch = 0
        regressor_loss_epoch = 0

        lambda_grad *= self.config.lambda_classifier
        lambda_regressor = self.config.lambda_regressor
        classifier_acc = 0
        classifier_num = 0
        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)

            for data_generator_idx, sample in enumerate(data_generator):
                (obs_batch, recurrent_hidden_states_batch, actions_batch, prev_actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, info, sound_id_batch, x_batch, y_batch, z_batch) = sample
                # Reshape to do in a single forward pass for all steps
                (values, action_log_probs, dist_entropy, _, predicted_labels, predicted_dir,) = \
                    self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        prev_actions_batch,
                        masks_batch,
                        actions_batch,
                        lambda_grad=lambda_grad,
                    )

                if self.config.use_buffer_train or self.config.use_buffer_more:
                    if self.buffer_pointer + len(obs_batch["spectrogram"]) > self.config.buffer_maxlen:
                        self.buffer_spect[self.buffer_pointer:] = obs_batch["spectrogram"][:self.config.buffer_maxlen - self.buffer_pointer].to("cpu")
                        self.buffer_depth[self.buffer_pointer:] = obs_batch["depth"][:self.config.buffer_maxlen - self.buffer_pointer].to("cpu")
                        self.buffer_label[self.buffer_pointer:] = sound_id_batch[:self.config.buffer_maxlen - self.buffer_pointer].to("cpu")
                        self.buffer_spect[:len(obs_batch["spectrogram"]) - self.config.buffer_maxlen + self.buffer_pointer] = obs_batch["spectrogram"][self.config.buffer_maxlen - self.buffer_pointer:].to("cpu")
                        self.buffer_depth[:len(obs_batch["spectrogram"]) - self.config.buffer_maxlen + self.buffer_pointer] = obs_batch["depth"][self.config.buffer_maxlen - self.buffer_pointer:].to("cpu")
                        self.buffer_label[:len(obs_batch["spectrogram"]) - self.config.buffer_maxlen + self.buffer_pointer] = sound_id_batch[self.config.buffer_maxlen - self.buffer_pointer:].to("cpu")
                    else:
                        self.buffer_spect[self.buffer_pointer:self.buffer_pointer + len(obs_batch["spectrogram"])] = obs_batch["spectrogram"].to("cpu")
                        self.buffer_depth[self.buffer_pointer:self.buffer_pointer + len(obs_batch["spectrogram"])] = obs_batch["depth"].to("cpu")
                        self.buffer_label[self.buffer_pointer:self.buffer_pointer + len(obs_batch["spectrogram"])] = sound_id_batch.to("cpu")
                    self.buffer_pointer = (self.buffer_pointer + len(obs_batch["spectrogram"])) % self.config.buffer_maxlen


                # 就是PPO cliped的Loss！
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ)
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (0.5 * torch.max(value_losses, value_losses_clipped).mean())
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # 保存当前的总loss
                total_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)
                # logger.info(f"RL_loss: {total_loss}")

                # if predicted_x_y is not None:
                #     lambda_regressor = self.config.lambda_regressor
                #     reg_x_y_loss = F.mse_loss(predicted_x_y, torch.stack([x_batch, y_batch], dim=0))
                #     logger.info(f"reg_loss: %.3f" % reg_x_y_loss)
                #     regressor_loss_epoch += reg_x_y_loss.item()
                #     total_loss += lambda_regressor * reg_x_y_loss

                # 计算八分类regressor loss
                z_info = None
                if predicted_dir is not None:
                    # dir_num = 8 # 方向的分类数
                    # dir_label = x_y_to_dir(x_batch, y_batch, dir_num)
                    # rot_label = x_y_to_rot(x_batch, y_batch)  # 已将(-pi, pi)归一化到(-1, 1)

                    rot_label = x_y_to_sin_cos(x_batch, y_batch, z_batch)

                    # regressor_loss = F.cross_entropy(predicted_dir, dir_label)

                    # 计算角度旋转差异
                    # diff = t orch.abs(rot_label - predicted_dir)
                    # diff = torch.where(diff > 1, 2 - diff, diff)
                    # regressor_loss = F.mse_loss(diff, torch.zeros_like(diff, device=diff.device))
                    regressor_loss = F.mse_loss(predicted_dir, rot_label)
                    logger.info(f"regressor_loss: {regressor_loss}")

                    regressor_loss_epoch += regressor_loss.item()
                    if e == self.ppo_epoch - 1:  # 每一个大循环print一次distrabution，要不然实在是太长了找其他的太费劲了
                        # logger.info(f"z_batch: {z_batch}")
                        logger.info(f"z_max: {torch.max(torch.abs(z_batch))}")
                        # logger.info(f"z_sin_label: {rot_label[:,2]}")
                        # logger.info(f"z_sin_max: {torch.max(torch.abs(rot_label[:,2]))}")
                        z_info = (torch.max(torch.abs(z_batch)), rot_label[:, 2], torch.max(torch.abs(rot_label[:, 2])))
                    total_loss += lambda_regressor * regressor_loss

                # 依次判断是否启用classifier或regression(根据返回变量predicted_labels与redicted_dis_and_rots是否为None判断)

                if predicted_labels is not None:
                    if self.config.use_buffer_train:
                        lambda_classifier = 1.0

                        # buffer_list_spect_np = torch.stack(self.buffer_list_spect, dim=0)
                        # buffer_list_depth_np = torch.stack(self.buffer_list_depth, dim=0)
                        # buffer_list_label_np = torch.stack(self.buffer_list_label, dim=0)
                        index = torch.from_numpy(np.random.randint(0, self.buffer_label.shape[0], self.config.train_batch)).long()
                        # # without replacement
                        # # index = torch.from_numpy(np.arange(len(self.buffer_list_spect))[:self.config.train_batch]).long()
                        # # index = torch.LongTensor([random.randint(0, len(self.buffer_list_spect) - 1) for _ in range(self.config.train_batch)])

                        # sound_id_batch = torch.gather(buffer_list_label_np, dim=0, index=index.unsqueeze(dim=1))
                        # index = index.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
                        # index_sp = index.repeat((1, ) + buffer_list_spect_np.shape[1:])

                        # # index_sp = index.repeat(1, buffer_list_spect_np.shape[1], buffer_list_spect_np.shape[2], buffer_list_spect_np.shape[3])

                        # spect_batch = torch.gather(buffer_list_spect_np, dim=0, index=index_sp)
                        # index_de = index.repeat((1, ) + buffer_list_depth_np.shape[1:])

                        # # index_de = index.repeat(1, buffer_list_depth_np.shape[1], buffer_list_depth_np.shape[2], buffer_list_depth_np.shape[3])

                        # depth_batch = torch.gather(buffer_list_depth_np, dim=0, index=index_de)

                        sound_id_batch = self.buffer_label[index]
                        spect_batch = self.buffer_spect[index]
                        depth_batch = self.buffer_depth[index]

                        train_spectrograms_tensor = dict()
                        train_spectrograms_tensor["spectrogram"] = spect_batch.to(self.device)
                        train_label_tensor = torch.tensor(sound_id_batch).squeeze(dim=1).to(self.device)
                        feature_a = self.actor_critic.net.audio_encoder(train_spectrograms_tensor).squeeze(0)
                        feature_a_g = grad_reverse(feature_a, lambda_grad)
                        if self.config.classifier_behind is False:
                            predicted_labels = self.actor_critic.net.classifier(feature_a_g)
                        else:
                            train_depth_tensor = dict()
                            train_depth_tensor["depth"] = depth_batch.to(self.device)
                            feature_v = self.actor_critic.net.visual_encoder(train_depth_tensor).squeeze(0)
                            if self.config.is_cross_attention is True:
                                x_v_a = self.actor_critic.net.v_a_attention(feature_v, feature_a, feature_a).squeeze(1)
                                x_a_v = self.actor_critic.net.v_a_attention(feature_a, feature_v, feature_v).squeeze(1)
                                x1 = torch.cat([x_v_a, x_a_v], dim=1)  # dim = 2n
                            else:
                                x1 = torch.cat([feature_v, feature_a], dim=1)  # dim = 2n
                            x1_g = grad_reverse(x1, lambda_grad)
                            predicted_labels = self.actor_critic.net.classifier(x1_g)
                        classifier_loss = F.cross_entropy(predicted_labels, train_label_tensor)
                        classifier_entropy = Categorical(probs=predicted_labels).entropy().mean()
                        classifier_loss -= self.config.lambda_classifier_entropy * classifier_entropy
                        predicted_labels_arg_max = torch.argmax(predicted_labels.detach(), dim=1)
                        classifier_acc += (predicted_labels_arg_max == train_label_tensor).sum()
                        classifier_num += self.config.train_batch
                        classifier_loss_epoch += classifier_loss.item()
                        classifier_entropy_epoch += classifier_entropy.item()
                        classifier_acc_epoch += classifier_acc.item()

                        if e == self.ppo_epoch - 1:  # 每一个大循环print一次distrabution，要不然实在是太长了找其他的太费劲了
                            logger.info(f"classifier_loss: {classifier_loss}")
                            logger.info(f"classifier_entropy: {classifier_entropy}")
                            logger.info(f"classifier_acc: {classifier_acc/classifier_num}")
                            logger.info(f"classifier_dist: {predicted_labels.detach()[0]}")

                            # debug
                            if predicted_dir is not None:
                                logger.info(f"z_batch: {z_batch}")
                                logger.info(f"z_max: {torch.max(torch.abs(z_batch))}")
                                logger.info(f"z_sin_label: {rot_label[:,2]}")
                                logger.info(f"z_sin_max: {torch.max(torch.abs(rot_label[:,2]))}")

                        # print(classifier_loss.item())
                        total_loss += lambda_classifier * classifier_loss

                    else:
                        lambda_classifier = 1.0

                        sound_id_batch = sound_id_batch.squeeze(-1)

                        # debug
                        # print("shape:", sound_id_batch.shape, "啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊\n")

                        classifier_loss = F.cross_entropy(predicted_labels, sound_id_batch)

                        classifier_entropy = Categorical(probs=predicted_labels).entropy().mean()
                        classifier_loss -= self.config.lambda_classifier_entropy * classifier_entropy

                        predicted_labels_arg_max = torch.argmax(predicted_labels.detach(), dim=1)
                        classifier_acc = (predicted_labels_arg_max == sound_id_batch).sum() / sound_id_batch.shape[0]

                        # debug
                        logger.info(f"classifier_loss: {classifier_loss}")
                        logger.info(f"classifier_entropy: {classifier_entropy}")
                        logger.info(f"classifier_acc: {classifier_acc}")

                        classifier_loss_epoch += classifier_loss.item()
                        classifier_entropy_epoch += classifier_entropy.item()
                        classifier_acc_epoch += classifier_acc.item()

                        if e == 0:  # 每一个大循环print一次distrabution，要不然实在是太长了找其他的太费劲了
                            logger.info(f"classifier_dist: {predicted_labels.detach()[0]}")

                        # print(classifier_loss.item())
                        total_loss += lambda_classifier * classifier_loss
                    """
                    if is_against:
                        if self.config.is_against is True:
                            total_loss -= lambda_classifier * classifier_loss
                        elif self.config.is_help_classify is True:
                            total_loss += lambda_classifier * classifier_loss
                    """
                '''
                if predicted_dis_and_rots is not None:
                    lambda_distance = 1e-2 # todo: this should be modified and add to config yaml
                    lambda_rotation = 1e-1
                    # lambda_regressor = 1
                    
                    predicted_dis = predicted_dis_and_rots[:,0].unsqueeze(1)
                    reg_dis_loss = F.mse_loss(predicted_dis, distance_batch)
                    
                    # 计算rotation loss的时候注意，因为它的两边(-pi和pi)是一样的，考虑从一个弧度旋转到另一个弧度的距离，
                    # 两个弧度的距离应该指的是 min(|r1-r2|, 2pi - |r1-r2|)
                    predicted_rot = predicted_dis_and_rots[:,1].unsqueeze(1)
                    abs = torch.abs(predicted_rot - rotation_batch)
                    diff = torch.where(abs < np.pi, abs, 2 * np.pi - abs)
                    reg_rot_loss = F.mse_loss(diff, torch.zeros_like(predicted_rot, device=predicted_rot.device))
                    
                    # regression_loss = F.mse_loss(predicted_dis_and_rots, torch.cat([distance_batch, rotation_batch], dim=1))
                    regression_loss = lambda_distance * reg_dis_loss + lambda_rotation * reg_rot_loss
                    
                    # debug
                    logger.info(f"reg_loss: {regression_loss}, dis_loss: {reg_dis_loss}, rot_loss: {reg_rot_loss}")
                    print(regression_loss.item())
                    
                    total_loss += regression_loss
                '''


                self.optimizer.zero_grad()
                # total_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef) \
                #              + lambda_classifier * classifier_loss \
                #              + lambda_regressor * regression_loss

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()
                """
                if predicted_labels is not None:
                    self.classifier_optimizer.zero_grad()
                    classifier_loss_detached.backward()
                    self.before_step()
                    self.classifier_optimizer.step()
                    self.after_step()
                """

                # todo:检查是否需要regressor optimizer, 如果需要就补上
                if hasattr(self.config, "use_buffer_more") and self.config.use_buffer_more:
                    self.more_train()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        classifier_loss_epoch /= num_updates
        classifier_entropy_epoch /= num_updates
        if classifier_num == 0:  # avoid divide by zero
            classifier_acc_epoch = 0
        else:
            classifier_acc_epoch /= (num_updates * classifier_num)  # debug fix classifier_acc display in Tensorboard
        regressor_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, classifier_loss_epoch, classifier_entropy_epoch, \
               classifier_acc_epoch, regressor_loss_epoch, z_info

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self):
        pass

    def more_train(self):
        train_spectrograms_tensor = torch.stack(self.buffer_list_spect, dim=0).squeeze(dim=1)
        train_label_tensor = torch.tensor(self.buffer_list_label)
        deal_dataset = TensorDataset(train_spectrograms_tensor, train_label_tensor)
        train_loader = DataLoader(dataset=deal_dataset, batch_size=self.config.train_batch, shuffle=True)
        optimizer = optim.Adam(self.actor_critic.net.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
        for epoch in range(self.config.num_epoch):
            train_loss = 0
            spects = dict()
            classifier_acc_num = 0
            classifier_num = 0
            for i, data in enumerate(train_loader):
                spects["spectrogram"], sound_id = data
                spects["spectrogram"] = spects["spectrogram"].to(self.device)
                sound_id = sound_id.to(self.device)
                # print("s: ", spects["spectrogram"].shape)
                feature = self.actor_critic.net.audio_encoder(spects).squeeze(0)
                predicted_labels = self.actor_critic.net.classifier(feature)
                classifier_loss = F.cross_entropy(predicted_labels, sound_id)
                optimizer.zero_grad()

                classifier_loss.backward()
                torch.nn.utils.clip_grad_value_(self.actor_critic.net.classifier.parameters(), 10)
                optimizer.step()
                train_loss += classifier_loss.item()
                predicted_labels_arg_max = torch.argmax(predicted_labels, dim=1)
                classifier_acc_num += (predicted_labels_arg_max == sound_id).sum()
                classifier_num += sound_id.shape[0]
            logger.info("more train epoch:{} acc:{}".format(epoch, classifier_acc_num / classifier_num))


# 将x,y坐标转换为n分类的方向
# n个方位范围为(k*pi/n - pi/2n, k*pi/n + pi/2n), k=0,1,...,n-1
# 这是为了避免四个与坐标轴平行的方位出现在两类分界线处，导致不好判断
def x_y_to_dir(x, y, n):
    eps = 1e-6
    z_rot = torch.where(torch.abs(x) < eps, torch.where(y > 0, np.pi / 2, -np.pi / 2), torch.arctan(y / x))
    z_rot = torch.where(x < -eps, z_rot + np.pi * ((n + 1) / n), z_rot + np.pi * (1 / n))
    z_class = torch.floor(z_rot / (np.pi / (n / 2)))
    z_class = z_class.type(torch.long).squeeze()
    dir_label = torch.where(z_class < 0, z_class + n, z_class)
    return dir_label


# 将x,y坐标转换为rotation角度
def x_y_to_rot(x, y):
    eps = 1e-6
    z_rot = torch.where(torch.abs(x) < eps, torch.where(y > 0, np.pi / 2, -np.pi / 2), torch.arctan(y / x))
    z_rot = torch.where(x < -eps, z_rot + np.pi, z_rot)
    z_rot = torch.where(z_rot > np.pi, z_rot - 2 * np.pi, z_rot)
    # z_class = torch.floor(z_rot / (np.pi / (n/2)))
    # z_class = z_class.type(torch.long).squeeze()
    # dir_label = torch.where(z_class < 0, z_class + n, z_class)
    return z_rot / np.pi  # 归一化到(-1, 1)


def x_y_to_sin_cos(x, y, z):
    eps = 1e-5
    dist = torch.sqrt(x * x + y * y) + eps
    dist_z = torch.sqrt(x * x + y * y + z * z) + eps
    sin_batch = x / dist
    cos_batch = y / dist
    sinzbatch = z / dist_z
    ret = torch.cat([sin_batch, cos_batch, sinzbatch], dim=1)
    return ret
