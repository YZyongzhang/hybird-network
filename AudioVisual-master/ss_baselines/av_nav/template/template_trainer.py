#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque
from typing import Dict, List
import json
import random
from ipdb import set_trace

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.env_dys import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorageNaive
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation
)
from ss_baselines.av_nav.ppo.policy import AudioNavBaselinePolicy
from ss_baselines.av_nav.ppo.ppo import PPO
import cv2

SKIP_LIST = ['best_action', 'goals', 'episode_id', 'position', 'rotation', 'orientation',
             'graph', 'index', 'grid_size', 'scene_name']

@baseline_registry.register_trainer(name="AVTemplateTrainer")
class AVTemplateTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.envs = None

    # #render
    def transform_rgb_bgr(self,image):
        return image[:, :, [2, 1, 0]]

    def render(self, observations):
        #cv2.imshow(window_name, image)
        for i in range(5):
            cv2.imshow('rgb'+str(i), self.transform_rgb_bgr(observations[i]["rgb"]))
            keystroke = cv2.waitKey(1)
       # print("Agent stepping around inside environment.")

    # #render


    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_step, episode_rewards,
            episode_spls, episode_counts, episode_steps, expert_actions=None
    ):
        pth_time = 0.0
        env_time = 0.0
        t_sample_action = time.time()
        if expert_actions is not None:
            # 如果有expert actions，就用expert
            actions = expert_actions
        else:
            # sample actions， naive版，不用网络策略
            m = Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]).unsqueeze(0).repeat(self.envs.num_envs, 1))
            actions = m.sample()
            actions = actions.unsqueeze(1)
        # actions = rollouts.observations['best_action'][int(current_episode_step[0])].unsqueeze(1)
        # set_trace()
        # action size 必须得 [self.envs.num_envs, 1]

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        # 最重要的step env！
        # 有可能用expert action，所以要int
        outputs = self.envs.step([int(a[0].item()) for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        set_trace()
        
        #scene_id add infos
        for i in range(5):
            scene_id=self.envs.current_episodes()[i].scene_id.split('/')[3]
            infos[i]['scene_id']=scene_id
            #set_trace()

        # render
        self.render(observations)

        #self.envs.render()
        #set_trace()
        # 每个step传回下一步的expert actions是啥
        # 注意这里就不是从obs里拿，而是从infos里拿
        next_expert_actions = torch.tensor([obs['best_action'] for obs in observations]).unsqueeze(1)
        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, skip_list=SKIP_LIST)
        # batch.keys() = ['depth', 'spectrogram', 'best_action', 'goals']
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )
        spls = torch.tensor(
            [[info['spl']] for info in infos]
        )

        current_episode_reward += rewards
        current_episode_step += 1
        # current_episode_reward is accumulating rewards across multiple updates,
        # as long as the current episode is not finished
        # the current episode reward is added to the episode rewards only if the current episode is done
        # the episode count will also increase by 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_spls += (1 - masks) * spls
        episode_steps += (1 - masks) * current_episode_step
        # 如果done了+1  没done就不变
        episode_counts += 1 - masks
        # 如果done了  这俩就会清零，没done就一直保持累计
        current_episode_reward *= masks
        current_episode_step *= masks

        # 传的是rollout的引用，这里rollout就被更新了
        rollouts.insert(
            batch,
            actions,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs, next_expert_actions

    def train(self) -> None:
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # set_trace()
        # self.config['NUM_PROCESSES'] = 1
        # set_trace()
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        ppo_cfg = self.config.RL.PPO
        # ppo_cfg.num_steps = 150
        rollouts = RolloutStorageNaive(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
        )

        # 这封装妙哇
        rollouts.to(self.device)
        observations = self.envs.reset()
        # goals = self.envs.habitat_env._env.current_episode.goals

        # render
        self.render(observations)

        batch = batch_obs(observations, skip_list=SKIP_LIST)
        expert_actions = torch.tensor([obs['best_action'] for obs in observations]).unsqueeze(1)
        # self.envs.num_envs = 5
        # batch['depth'].shape = torch.Size([5, 128, 128, 1])
        # batch['spectrogram'].shape = torch.Size([5, 65, 69, 2])

        # rollouts.observations.keys() = dict_keys(['depth', 'spectrogram'])
        # 这里是把当前的batch塞进rollouts这个buffer里，用reset的状态初始化rollout！
        # set_trace()
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        # 卧槽  还要手动释放的，我以前都不知道
        batch = None
        observations = None

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        # spl 就是 inverse path length
        # num_envs = 5， reward_window_size = 50
        episode_spls = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_spl = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                # NUM_UPDATES: 40000
                # 虽然采了ppo_cfg.num_steps = 150这么多的trajs，但episodes长度是变化的
                for step in range(ppo_cfg.num_steps):
                    # 下面这句都是传引用进去！
                    delta_pth_time, delta_env_time, delta_steps, expert_actions = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        current_episode_step,
                        episode_rewards,
                        episode_spls,
                        episode_counts,
                        episode_steps,
                        expert_actions=expert_actions,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                    # set_trace()
                    # rollouts.observations['depth'].shape = [151, 5, 128, 128, 1]
                    # rollouts.observations['spectrogram'].shape = [151, 5, 65, 69, 2]
                    # rollouts.rewards.shape = [150, 5, 1]
                    # rollouts.actions.shape = [150, 5, 1]
                    # rollouts.prev_actions.shape = [151, 5, 1]
                    # rollouts.masks.shape = [151, 5, 1]
                    # rollouts.num_steps = 150

                # 也说明前面都是浅拷贝
                window_episode_reward.append(episode_rewards.clone())
                window_episode_spl.append(episode_spls.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())

                stats = zip(
                    ["count", "reward", "step", 'spl'],
                    [window_episode_counts, window_episode_reward, window_episode_step, window_episode_spl],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                # this reward is averaged over all the episodes happened during window_size updates
                # approximately number of steps is window_size * num_steps
                if update % 10 == 0:
                    writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/SPL", deltas["spl"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)

                # 用来打印到屏幕上
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

            self.envs.close()
