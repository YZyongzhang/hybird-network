#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from ipdb import set_trace
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNet
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_nav.models.visual_cnn import VisualCNN, WhcVisualCNN
from ss_baselines.av_nav.models.audio_cnn import AudioCNN, WhcAudioCNN
from ss_baselines.av_nav.ppo.attention.attention import BasicAttention
from torch.autograd import Function
from zmq import device

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):

    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        # print('Features: ', features.cpu().numpy())
        distribution = self.action_distribution(features)
        # print('Distribution: ', distribution.logits.cpu().numpy())
        value = self.critic(features)
        # print('Value: ', value.item())

        if deterministic:
            action = distribution.mode()
            # print('Deterministic action: ', action.item())
        else:
            action = distribution.sample()
            # print('Sample action: ', action.item())

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):

    def __init__(self, observation_space, action_space, goal_sensor_uuid, hidden_size=512, extra_rgb=False):
        super().__init__(
            AudioNavBaselineNet(observation_space=observation_space, hidden_size=hidden_size, goal_sensor_uuid=goal_sensor_uuid, extra_rgb=extra_rgb),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'

            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            # set_trace()
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        # if 'depth' in observation_space.spaces:
        #     depth_shape = observation_space.spaces['depth'].shape
        #     summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1


class WhcAudioNavBaselinePolicy(Policy):

    def __init__(self, observation_space, action_space, goal_sensor_uuid, hidden_size=512, extra_rgb=False, sound_num=128, config=None):
        super().__init__(
            WhcAudioNavBaselineNet(observation_space=observation_space,
                                   hidden_size=hidden_size,
                                   goal_sensor_uuid=goal_sensor_uuid,
                                   extra_rgb=extra_rgb,
                                   config=config,
                                   sound_num=sound_num),
            action_space.n,
        )
        self.config = config
        self.observation_space = observation_space

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        lambda_grad=1.0,
    ):
        features, rnn_hidden_states, predicted_labels, predicted_rot = self.net(observations,
                                                                                rnn_hidden_states,
                                                                                prev_actions,
                                                                                masks,
                                                                                lambda_grad=lambda_grad)
        distribution = self.action_distribution(features)
        value = self.critic(features)
        # set_trace()

        # 表示是否是确定性的，确定性的就直接按概率最大的来，不确定性的还要sample一下
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution, predicted_labels, predicted_rot, None, None, None

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0):
        features, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks, lambda_grad=lambda_grad)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action, lambda_grad=1.0):
        features, rnn_hidden_states, predicted_labels, predicted_rot = self.net(observations,
                                                                                rnn_hidden_states,
                                                                                prev_actions,
                                                                                masks,
                                                                                lambda_grad=lambda_grad)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        # loss在这里！！
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, predicted_labels, predicted_rot
        # 注意：使用返回的predict进行计算的函数，需要检查返回值是否为None！！！


class WhcAudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False, config=None, sound_num=128):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0
        self.config = config
        self.sound_num = sound_num
        self.observation_space = observation_space
        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = WhcVisualCNN(observation_space, hidden_size, extra_rgb, config)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'

            self.audio_encoder = WhcAudioCNN(observation_space, hidden_size, audiogoal_sensor, config)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            # set_trace()
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        # if 'depth' in observation_space.spaces:
        #     depth_shape = observation_space.spaces['depth'].shape
        #     summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if (hasattr(self.config, 'classifier_behind')):
            self.classifier_behind = self.config.classifier_behind
        else:
            self.classifier_behind = False
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            # summary(self.audio_encoder.cnn1, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        if self.config.is_classify:
            if self.config.is_visual_classify or self.classifier_behind:  # 同时classify visual+audio
                # assert self.is_visual_encode
                self.classifier = Sound_Classifier2(2 * self._hidden_size, sound_num, num_layers=self.config.classifier_num_layers)  # new classifier!
            else:  # 只classify audio
                self.classifier = Sound_Classifier2(self._hidden_size, sound_num, num_layers=self.config.classifier_num_layers)
        if self.config.is_cross_attention:
            self.v_a_attention = BasicAttention(self._hidden_size, self._hidden_size, self._hidden_size)
            self.a_v_attention = BasicAttention(self._hidden_size, self._hidden_size, self._hidden_size)
        if self.config.is_regression:
            # self.regressor = x_y_regressor(self._hidden_size, 2)
            # self.regressor = X_Y_Regresor(self._hidden_size, 2)  # 输出Δx Δy两个维度
            # self.regressor = DirRegresor(self._hidden_size, 8)  #目前使用八分类离散访问分类
            # self.regressor = RotRegresor(self._hidden_size, 1)  # 改回角度回归！
            self.regressor = RotRegresor(self._hidden_size, 3)  # 回归角度的sin和cos值
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0):
        x = []

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x_a = self.audio_encoder(observations)
            x.append(x_a)
        if not self.is_blind:
            x_v = self.visual_encoder(observations)
            x.append(x_v)

        if self.config.is_cross_attention:
            x_v_a = self.v_a_attention(x_v, x_a, x_a).squeeze(1)
            x_a_v = self.a_v_attention(x_a, x_v, x_v).squeeze(1)
            x1 = torch.cat([x_v_a, x_a_v], dim=1)  # dim = 2n
            # x_av = self.cross_attention_cnn(x_av)
        else:
            x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        if self.config.is_classify:
            if self.config.is_visual_classify:
                # assert self.is_visual_encode
                x1 = grad_reverse(x1, lambda_grad)
                predicted_labels = self.classifier(x1)
            elif self.classifier_behind is False:
                x_a_g = grad_reverse(x_a, lambda_grad)
                predicted_labels = self.classifier(x_a_g)
            elif self.classifier_behind is True:
                x1_g = grad_reverse(x1, lambda_grad)
                predicted_labels = self.classifier(x1_g)
        else:
            predicted_labels = None

        if self.config.is_regression:
            if self.config.reg_behind:
                predicted_rot = self.regressor(x2)
            else:
                predicted_rot = self.regressor(x_a)
        else:
            predicted_rot = None

        return x2, rnn_hidden_states1, predicted_labels, predicted_rot


# 修改前的classifier，过于复杂很难训好
class Sound_Classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Sound_Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.net(x)


class Sound_Classifier2(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=2):
        super(Sound_Classifier2, self).__init__()
        self.net = []
        num_decrease_layers = int(np.ceil(np.log2(input_dim / output_dim)))
        for i in range(num_layers - num_decrease_layers):
            self.net.append(spectral_norm(nn.Linear(input_dim, input_dim)))
            self.net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(min(num_decrease_layers, num_layers) - 1):
            self.net.append(spectral_norm(nn.Linear(input_dim, input_dim // 2)))
            self.net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            input_dim = input_dim // 2

        self.net.append(spectral_norm(nn.Linear(input_dim, output_dim)))
        self.net.append(nn.Softmax())
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class X_Y_Regresor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(X_Y_Regresor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# n分类离散方位回归(n=8)
class DirRegresor(nn.Module):

    def __init__(self, input_dim, output_dim):  # output_dim为分类数
        super(DirRegresor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(True), nn.Linear(1024, 1024), nn.ReLU(True), nn.Linear(1024, 512), nn.ReLU(True),
                                 nn.Linear(512, output_dim), nn.Softmax())

    def forward(self, x):
        return self.net(x)


# 角度回归 (sin, cos)
class RotRegresor(nn.Module):

    def __init__(self, input_dim, output_dim):  # output_dim = 2为分类数
        super(RotRegresor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(True), nn.Linear(1024, 1024), nn.ReLU(True), nn.Linear(1024, 512), nn.ReLU(True),
                                 nn.Linear(512, output_dim), nn.Tanh())  # 把输出归一化到(-1, 1)

    def forward(self, x):
        return self.net(x)


class MLPForVisualAndAudio(nn.Module):
    """
        用在没有av cross attention，但有gm与av的attention的情况下
        来保证a+v的维度与gm维度一致
    """

    def __init__(self, input_dim):
        super(MLPForVisualAndAudio, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim * 2), nn.ReLU(True), nn.Linear(input_dim * 2, input_dim), nn.ReLU(True),
                                 nn.Linear(input_dim, input_dim // 2))

    def forward(self, x):
        return self.net(x)


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(x, lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        _, lambd = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (-lambd)
        return grad_input, None


def grad_reverse(x, lambd):
    lambd = torch.autograd.Variable(torch.FloatTensor([lambd])).to(x.device)
    return GradReverse.apply(x, lambd)
