#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from ipdb import set_trace
from copy import deepcopy
import numpy as np
import os
import networkx as nx
import pickle
import cv2
from habitat.utils.visualizations.utils import images_to_video
from ss_baselines.common.utils import images_to_video_with_audio

import torch


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def json2point(input_shape, c_x, c_y, sigma):
    img_height = input_shape[0]
    img_width = input_shape[1]
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)

    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


def render_color_heat(heat, image):
    heat /= heat.sum()
    # 先norm这张heatmap
    gray_img = heat
    norm_img = np.zeros(gray_img.shape)
    norm_img = cv2.normalize(gray_img, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    # 搞个热力图
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    img_add = cv2.addWeighted(image, 0.5, heat_img, 0.5, 0)

    return img_add


class RolloutStorageAffordance:
    r"""Class for storing rollout information for affordance trainers.

    """

    def __init__(
        self,
        num_envs,
        vec_envs,
        root_path,
        config,
    ):
        self.num_envs = num_envs
        self.caches = {env_idx: [] for env_idx in range(self.num_envs)}
        self.num_trajs = {}
        self.vec_envs = vec_envs
        self.root_path = root_path
        self.sample_num = 0
        self.config = config

    def process_traj_and_save(self, traj, scene_idx):
        # for obs in traj:
        #     print(obs['depth'].max())
        # set_trace()
        cur_training_pairs = []
        hfov = 130
        thres = 0
        # 最后一个不算
        graph = deepcopy(traj[0][0]['graph'])
        scene_name = traj[0][0]['scene_name']
        demo_full = []
        demo_label = []
        depth_full = []
        audio_label = []
        audio_full = []
        for idx_cur, (obs_cur, info_cur) in enumerate(traj[:-1]):
            id_cur = obs_cur['index']
            p1 = graph.nodes[id_cur]['point']
            ori_cur = obs_cur['orientation']
            grid_size = obs_cur['grid_size']
            depth = obs_cur['depth'] * 2
            audiogoal = obs_cur['audiogoal']
            # 然后把这条轨迹，从当前点开始的，后续的点，倒过来找
            flag_mask = False
            for idx_next, (obs_next, info_next) in enumerate(traj[:idx_cur:-1]):
                # 考察cur和next是否是有效pair
                id_next = obs_next['index']
                p2 = graph.nodes[id_next]['point']
                # 说明是转弯的action
                if p1 == p2:
                    continue
                # 下面relative_dir 实锤是从X+ -> Z+方向的[0, 360)角度
                relative_dir = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                delta = abs(relative_dir - ori_cur)
                delta = 360 - delta if delta > 180 else delta
                # print(relative_dir)
                # print(ori_cur)
                # print(delta)
                # set_trace()
                if delta < (hfov / 2 - thres):
                    # 判断完视角，就再看遮挡问题
                    # shortest = nx.shortest_path_length(graph, id_next, id_cur) * grid_size
                    # line_dist = np.sum(np.abs(np.array(p1) - np.array(p2)))
                    # flag_visible = np.abs(shortest - line_dist) < grid_size
                    # flag_visible = line_dist < grid_size * 12
                    # flag_visible = True
                    l2_dist = np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
                    # 下面判断遮挡，算flag_visible, 首先要获得目标点的h w
                    im_size = obs_cur['rgb'].shape[0]  #
                    # 因为标的点总是和摄像机海拔一致，且摄像机朝正前方
                    point_h = im_size // 2
                    # 往左是角度的正向，delta目前是角度制
                    delta_rad = np.deg2rad(delta)
                    hfov_rad = np.deg2rad(hfov)
                    ratio_center = (np.sin(delta_rad) + np.sin(hfov_rad / 2)) / (2 * np.sin(hfov_rad))
                    # ratio = (delta + hfov//2)/hfov
                    # 因为图像向左递减，而方向角向左递增
                    point_w = int(im_size * (1 - ratio_center))
                    if point_w < 0:
                        point_w = 0
                    if point_w >= im_size:
                        point_w = im_size
                    # 现在这个点(假如visible的话)就在 [point_h, point_w]

                    true_dist = depth[point_h][point_w][0]
                    # set_trace()
                    flag_visible = l2_dist < true_dist
                    # print(depth.max())
                    # print(l2_dist)
                    # print(true_dist)
                    # print(flag_visible)
                    # set_trace()
                    if flag_visible:
                        obs_cur['heatmap'] = json2point((im_size, im_size), point_w, point_h, 15)
                        obs_cur['mask'] = render_color_heat(obs_cur['heatmap'], obs_cur['rgb'])
                        obs_cur['label'] = {
                            'point_h': point_w,
                            'point_w': point_h,
                            # 'radius': radius,
                        }
                        cur_training_pairs.append((obs_cur, obs_next))
                        flag_mask = True
                        break
            demo_full.append(obs_cur['mask'] if flag_mask else obs_cur['rgb'])
            audio_full.append(audiogoal)
            depth_full.append(obs_cur['depth'])
            # print(obs_cur['depth'].max())
            # print(obs_cur['depth'].min())
            set_trace()
            if flag_mask:
                demo_label.append(obs_cur['mask'])
                audio_label.append(audiogoal)

        # print(len(traj) - len(cur_training_pairs))
        # 其实标不上label的比例还挺高的
        # 存数据
        exists_or_mkdir(f'{self.root_path}/{scene_name}')
        exists_or_mkdir(f'{self.root_path}/{scene_name}/{scene_idx}')
        for pair_idx, (obs_cur, obs_next) in enumerate(cur_training_pairs):
            pickle_path = f'{self.root_path}/{scene_name}/{scene_idx}/{pair_idx}_meta.pickle'
            rgb_path = f'{self.root_path}/{scene_name}/{scene_idx}/{pair_idx}_rgb.png'
            mask_path = f'{self.root_path}/{scene_name}/{scene_idx}/{pair_idx}_mask.png'
            heatmap_path = f'{self.root_path}/{scene_name}/{scene_idx}/{pair_idx}_heatmap.png'
            heatmap_np_path = f'{self.root_path}/{scene_name}/{scene_idx}/{pair_idx}_heatmap.np'
            data = {'audio': obs_cur['spectrogram'], 'position': obs_cur['position'], 'orientation': obs_cur['orientation'], 'target_pos': obs_next['position'], 'label': obs_cur['label']}
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
            with open(heatmap_np_path, 'wb') as f:
                pickle.dump(obs_cur['heatmap'], f)
            cv2.imwrite(rgb_path, obs_cur['rgb'])
            cv2.imwrite(mask_path, obs_cur['mask'])
            # 可视化得画清楚点
            cv2.imwrite(heatmap_path, obs_cur['heatmap'] * 200)
            with open(f'{self.root_path}/paths.txt', 'a+') as f:
                f.write(f'{self.root_path}/{scene_name}/{scene_idx}/{pair_idx}\n')
            self.sample_num += 1
            if self.sample_num % 1000 == 0:
                print(self.sample_num)

        # 存video
        exists_or_mkdir(f'{self.root_path}/{scene_name}/videos')
        images_to_video(demo_full, f'{self.root_path}/videos', f'full_{scene_name}_{scene_idx}', fps=1, debug=False)
        images_to_video(demo_label, f'{self.root_path}/videos', f'label_{scene_name}_{scene_idx}', fps=1, debug=False)
        # images_to_video(depth_full, f'{self.root_path}/videos', f'depth_{scene_name}_{scene_idx}', fps=1, debug=False)
        images_to_video_with_audio(demo_full, f'{self.root_path}/videos', f'audio_full_{scene_name}_{scene_idx}', audio_full, sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE, fps=1)

    def print_cache(self):
        # check现在cache里缓存的obs数量是多少
        res = ''
        for key in self.caches.keys():
            res += f'{key}:\t{len(self.caches[key])}\t'
        res += '\n'
        print(res)

    def print_trajs(self):
        # check现在收集了多少有效的trajs
        res = ''
        for idx, num in enumerate(self.num_trajs):
            res += f'{idx}:\t{num}\t'
        res += '\n'
        print(res)

    def update(
        self,
        observations,
        dones,
        successes,
        infos,
    ):
        for idx, (obs, done, success, info) in enumerate(zip(observations, dones, successes, infos)):
            self.caches[idx].append((obs, info))
            if done:
                if success:
                    # 把这一条拎出来处理一下存好
                    scene_name = obs['scene_name']
                    if scene_name not in self.num_trajs.keys():
                        self.num_trajs[scene_name] = 1
                    else:
                        self.num_trajs[scene_name] += 1
                    self.process_traj_and_save(self.caches[idx], self.num_trajs[scene_name])
                # 清空缓存
                self.caches[idx] = []
        # set_trace()
        # self.caches[0][0]['spectrogram']
        # dist = np.sum(np.abs(self.caches[0][-1]['spectrogram'] - self.caches[0][-2]['spectrogram']))
        # self.print_cache()
        # self.print_trajs()


class RolloutStorageNaive:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(num_steps + 1, num_envs, *observation_space.spaces[sensor].shape)
        # # for additional info, best_action and goals
        # self.observations['best_action'] = torch.zeros(
        #     num_steps + 1,
        #     num_envs,
        # )
        # self.observations['goals'] = torch.zeros(
        #     num_steps + 1,
        #     num_envs,
        #     3,
        # )

        self.rewards = torch.zeros(num_steps, num_envs, 1)

        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        # step是指针
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.rewards = self.rewards.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        actions,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(observations[sensor])
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.masks[0].copy_(self.masks[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(num_steps + 1, num_envs, *observation_space.spaces[sensor].shape)

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(observations[sensor])
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step])
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step])

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, ("Trainer requires the number of processes ({}) "
                                                 "to be greater than or equal to the number of "
                                                 "trainer mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(self.observations[sensor][:-1, ind])

                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0, :, ind])

                actions_batch.append(self.actions[:, ind])
                prev_actions_batch.append(self.prev_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])

                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], 1)

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(T, N, observations_batch[sensor])

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class WhcRolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(self, num_steps, num_envs, observation_space, action_space, recurrent_hidden_state_size, num_recurrent_layers=1, config=None):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(num_steps + 1, num_envs, *observation_space.spaces[sensor].shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)
        self.info = [None] * (num_steps + 1)
        self.sound_ids = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.long)
        self.x_delta = torch.zeros(num_steps + 1, num_envs, 1)
        self.y_delta = torch.zeros(num_steps + 1, num_envs, 1)
        self.z_delta = torch.zeros(num_steps + 1, num_envs, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)
        self.sound_ids = self.sound_ids.to(device)
        self.x_delta = self.x_delta.to(device)
        self.y_delta = self.y_delta.to(device)
        self.z_delta = self.z_delta.to(device)

    def insert(self, observations, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks, infos, sound_ids, x_delta, y_delta, z_delta):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(observations[sensor])
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.info[self.step] = deepcopy(infos)
        self.sound_ids[self.step].copy_(sound_ids)
        self.x_delta[self.step].copy_(x_delta)
        self.y_delta[self.step].copy_(y_delta)
        self.z_delta[self.step].copy_(z_delta)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step])
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step])

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, ("Trainer requires the number of processes ({}) "
                                                 "to be greater than or equal to the number of "
                                                 "trainer mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            info_batch = []
            sound_id_batch = []
            x_delta_batch = []
            y_delta_batch = []
            z_delta_batch = []
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(self.observations[sensor][:-1, ind])

                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0, :, ind])

                actions_batch.append(self.actions[:, ind])
                prev_actions_batch.append(self.prev_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])

                adv_targ.append(advantages[:, ind])
                sound_id_batch.append(self.sound_ids[:-1, ind])
                x_delta_batch.append(self.x_delta[:-1, ind])
                y_delta_batch.append(self.y_delta[:-1, ind])
                z_delta_batch.append(self.z_delta[:-1, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], 1)

            # info_batch = torch.stack(actions_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            sound_id_batch = torch.stack(sound_id_batch, 1)
            x_delta_batch = torch.stack(x_delta_batch, 1)
            y_delta_batch = torch.stack(y_delta_batch, 1)
            z_delta_batch = torch.stack(z_delta_batch, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(T, N, observations_batch[sensor])

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = self._flatten_helper(T, N, adv_targ)
            sound_id_batch = self._flatten_helper(T, N, sound_id_batch)
            x_delta_batch = self._flatten_helper(T, N, x_delta_batch)
            y_delta_batch = self._flatten_helper(T, N, y_delta_batch)
            z_delta_batch = self._flatten_helper(T, N, z_delta_batch)
            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                info_batch,
                sound_id_batch,
                x_delta_batch,
                y_delta_batch,
                z_delta_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
