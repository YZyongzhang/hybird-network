#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import os
import time
import logging
from collections import deque
from typing import Dict, List
import json
import random
from torch.optim import Adam
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from habitat import Config, logger
from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage, WhcRolloutStorage
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (batch_obs, generate_video, linear_decay, exponential_decay, plot_top_down_map,
                                       resize_observation)
from ss_baselines.av_wan.ppo import AudioNavBaselinePolicy, WyxAudioNavBaselinePolicy, XcxAudioNavBaselinePolicy
from ss_baselines.av_wan.ppo import PPO, WyxPPO
from habitat.utils.visualizations.utils import images_to_video
from ipdb import set_trace
import cv2
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image
import matplotlib
import matplotlib.pyplot as plt
from pandas import Series
import math
from interval import Interval
from sklearn.manifold import TSNE
import re
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, dataset, TensorDataset

# import heartrate

# heartrate.trace(browser=True)


def draw_top_down_map(info):
    top_down_map = info["top_down_map"]["map"]

    top_down_map = maps.colorize_topdown_map(top_down_map)
    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["top_down_map"]["agent_angle"],
        agent_radius_px=top_down_map.shape[0] // 25,
    )

    return top_down_map


@baseline_registry.register_trainer(name="AVWanTrainer")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]
    images = []

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

    def render(self, observations, infos):
        # cv2.imshow(window_name, image)
        for i in range(5):
            # wpmap = np.zeros((200, 200, 1))
            # set_trace()

            img = draw_top_down_map(infos[i])

            # img *= 100
            # set_trace()
            # cv2.imshow('demo'+str(i), img)
            # cv2.waitKey(0)
            # cv2.imwrite('/data/AudioVisual/sound-spaces/ss_baselines/av_wan/Demo/'+str(i)+'.png', img)
            self.images[i].append(img)

    # print("Agent stepping around inside environment.")

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
        self.actor_critic = AudioNavBaselinePolicy(observation_space=observation_space,
                                                   hidden_size=ppo_cfg.hidden_size,
                                                   goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                                                   masking=self.config.MASKING,
                                                   encode_rgb=self.config.ENCODE_RGB,
                                                   encode_depth=self.config.ENCODE_DEPTH,
                                                   action_map_size=self.config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE)
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name), _use_new_zipfile_serialization=True)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(self, rollouts, current_episode_reward, current_episode_step, episode_rewards,
                              episode_spls, episode_counts, episode_steps, episode_distances):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}

            (values, actions, actions_log_probs, recurrent_hidden_states, distributions) = self.actor_critic.act(
                step_observation, rollouts.recurrent_hidden_states[rollouts.step], rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step])

        # ipdb > step_observation.keys()
        # dict_keys(['action_map', 'am', 'collision', 'depth', 'ego_map', 'gm', 'intensity', 'pointgoal_with_gps_compass',
        #            'spectrogram'])
        # ipdb > rollouts.recurrent_hidden_states[rollouts.step]
        # tensor([[[0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.]]], device='cuda:0')
        # ipdb > rollouts.prev_actions[rollouts.step]
        # tensor([[0],
        #         [0],
        #         [0],
        #         [0],
        #         [0]], device='cuda:0')
        # ipdb > rollouts.masks[rollouts.step]
        # tensor([[1.],
        #         [1.],
        #         [1.],
        #         [1.],
        #         [1.]], device='cuda:0')

        pth_time += time.time() - t_sample_action
        t_step_env = time.time()

        outputs = self.envs.step([{"action": a[0].item()} for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        # self.render(observations, infos)
        # set_trace()
        # ipdb > infos[0]
        # {'distance_to_goal': 7.5, 'normalized_distance_to_goal': 1.1538461538461537, 'success': 0.0, 'spl': 0.0,
        #  'softspl': 0.0, 'na': 3, 'sna': 0.0, 'reaching_waypoint': True, 'cant_reach_waypoint': False,
        #  'waypoints_pos': (240, 250)}

        # ipdb > observations[0].keys()
        # dict_keys(
        #     ['depth', 'spectrogram', 'ego_map', 'pointgoal_with_gps_compass', 'gm', 'action_map', 'collision', 'am',
        #      'intensity'])

        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float)
        spls = torch.tensor([[info['spl']] for info in infos])

        distances = torch.tensor([[info['distance_to_goal']] for info in infos])

        current_episode_reward += rewards
        current_episode_step += 1
        # current_episode_reward is accumulating rewards across multiple updates,
        # as long as the current episode is not finished
        # the current episode reward is added to the episode rewards only if the current episode is done
        # the episode count will also increase by 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_spls += (1 - masks) * spls
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        episode_distances += (1 - masks) * distances
        current_episode_reward *= masks
        current_episode_step *= masks

        rollouts.insert(batch, recurrent_hidden_states, actions, actions_log_probs, values, rewards, masks)

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {k: v[-1] for k, v in rollouts.observations.items()}
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        global lr_lambda
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        for _ in range(5):
            self.images.append([])
        # set_trace()

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME), auto_reset_done=False)

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info("agent number of parameters: {}".format(sum(param.numel() for param in self.agent.parameters())))

        rollouts = RolloutStorage(ppo_cfg.num_steps, self.envs.num_envs, self.envs.observation_spaces[0],
                                  self.envs.action_spaces[0], ppo_cfg.hidden_size)
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_spls = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        episode_distances = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_spl = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_distances = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        if ppo_cfg.use_linear_lr_decay:

            def lr_lambda(x):
                return linear_decay(x, self.config.NUM_UPDATES)
        elif ppo_cfg.use_exponential_lr_decay:

            def lr_lambda(x):
                return exponential_decay(x, self.config.NUM_UPDATES, ppo_cfg.exp_decay_lambda)
        else:

            def lr_lambda(x):
                return 1

        lr_scheduler = LambdaLR(optimizer=self.agent.optimizer, lr_lambda=lr_lambda)

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay or ppo_cfg.use_exponential_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(update, self.config.NUM_UPDATES)

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(rollouts,
                                                                                             current_episode_reward,
                                                                                             current_episode_step,
                                                                                             episode_rewards,
                                                                                             episode_spls,
                                                                                             episode_counts,
                                                                                             episode_steps,
                                                                                             episode_distances)
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                # for i in range(5):
                #     images_to_video(self.images[i], '/data/AudioVisual/sound-spaces/ss_baselines/av_wan/Demo', str(i))

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_spl.append(episode_spls.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())
                window_episode_distances.append(episode_distances.clone())

                losses = [value_loss, action_loss, dist_entropy]
                stats = zip(
                    ["count", "reward", "step", 'spl', 'distance'],
                    [window_episode_counts, window_episode_reward, window_episode_step, window_episode_spl,
                     window_episode_distances],
                )
                deltas = {k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()) for k, v in stats}
                deltas["count"] = max(deltas["count"], 1.0)

                # this reward is averaged over all the episodes happened during window_size updates
                # approximately number of steps is window_size * num_steps
                writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)

                writer.add_scalar("Environment/SPL", deltas["spl"] / deltas["count"], count_steps)

                logging.debug('Number of steps: {}'.format(deltas["step"] / deltas["count"]))
                writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)

                writer.add_scalar("Environment/Distance_to_goal", deltas["distance"] / deltas["count"], count_steps)

                # writer.add_scalars(
                #     "losses",
                #     {k: l for l, k in zip(losses, ["value", "policy"])},
                #     count_steps,
                # )

                writer.add_scalar('Policy/Value_Loss', value_loss, count_steps)
                writer.add_scalar('Policy/Action_Loss', action_loss, count_steps)
                writer.add_scalar('Policy/Entropy', dist_entropy, count_steps)
                writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(update, count_steps / ((time.time() - t_start) + prev_time)))

                    logger.info("update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                                "frames: {}".format(update, env_time, pth_time, count_steps))

                    window_rewards = (window_episode_reward[-1] - window_episode_reward[0]).sum()
                    window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()

                    if window_counts > 0:
                        logger.info("Average window size {} reward: {:3f}".format(
                            len(window_episode_reward),
                            (window_rewards / window_counts).item(),
                        ))
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # dzj
        # print("########")

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.TASK_CONFIG.TASK.SENSORS.append("AUDIOGOAL_SENSOR")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME), auto_reset_done=False)

        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 3)
        else:
            observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        self.metric_uuids = []
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(metric_cfg.TYPE)
            # dys modified
            # set_trace()
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_reaching_waypoint = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_cant_reach_waypoint = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_step_count = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [[] for _ in range(self.config.NUM_PROCESSES)]  # type: List[List[np.ndarray]]
        audios = [[] for _ in range(self.config.NUM_PROCESSES)]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (len(stats_episodes) < self.config.TEST_EPISODE_COUNT and self.envs.num_envs > 0):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states, distributions = self.actor_critic.act(batch,
                                                                                                   test_recurrent_hidden_states,
                                                                                                   prev_actions,
                                                                                                   not_done_masks,
                                                                                                   deterministic=True)

                prev_actions.copy_(actions)

            outputs = self.envs.step([{"action": a[0].item()} for a in actions])
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            if config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)

            batch = batch_obs(observations, self.device)
            if len(self.config.VIDEO_OPTION) > 0:
                rgb_frames[0] += infos[0]['rgb_frames']
                audios[0] += infos[0]['audios']

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            logging.debug('Reward: {}'.format(rewards[0]))

            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(1)
            current_episode_reward += rewards
            current_episode_step_count += 1
            next_episodes = self.envs.current_episodes()
            n_envs = self.envs.num_envs
            envs_to_pause = []
            for i in range(n_envs):
                # pause envs which runs out of episodes
                if (
                        next_episodes[i].scene_id,
                        next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)
                    logging.info('Pause env {} and remaining number of envs: {}'.format(i, self.envs.num_envs))

                current_episode_reaching_waypoint[i] += infos[i]['reaching_waypoint']
                current_episode_cant_reach_waypoint[i] += infos[i]['cant_reach_waypoint']

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = norm(
                        np.array(current_episodes[i].goals[0].position) - np.array(current_episodes[i].start_position))
                    episode_stats["reaching_waypoint"] = current_episode_reaching_waypoint[i].item() / \
                                                         current_episode_step_count[i].item()
                    episode_stats["cant_reach_waypoint"] = current_episode_cant_reach_waypoint[i].item() / \
                                                           current_episode_step_count[i].item()
                    current_episode_reaching_waypoint[i] = 0
                    current_episode_cant_reach_waypoint[i] = 0
                    current_episode_step_count[i] = 0
                    current_episode_reward[i] = 0
                    logging.debug(episode_stats)
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        if self.config.VISUALIZE_FAILURE_ONLY and infos[i]['success'] > 0:
                            pass
                        else:
                            fps = self.config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
                                if self.config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
                            if 'sound' in current_episodes[i].info:
                                sound = current_episodes[i].info['sound']
                            else:
                                sound = current_episodes[i].sound_id.split('/')[1][:-4]

                            # set_trace()
                            generate_video(video_option=self.config.VIDEO_OPTION,
                                           video_dir=self.config.VIDEO_DIR,
                                           images=rgb_frames[i][:-1],
                                           scene_name=current_episodes[i].scene_id.split('/')[3],
                                           sound=sound,
                                           sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                                           episode_id=current_episodes[i].episode_id,
                                           checkpoint_idx=checkpoint_index,
                                           metric_name='spl',
                                           metric_value=infos[i]['spl'],
                                           tb_writer=writer,
                                           audios=audios[i][:-1],
                                           fps=fps)

                        rgb_frames[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        top_down_map = plot_top_down_map(infos[i])
                        scene = current_episodes[i].scene_id.split('/')[-3]
                        writer.add_image('{}_{}_{}/{}'.format(config.EVAL.SPLIT, scene, current_episodes[i].episode_id,
                                                              config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]),
                                         top_down_map, dataformats='WHC')

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum([v[stat_key] for v in stats_episodes.values()])
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(config.TENSORBOARD_DIR, '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_reaching_waypoint_mean = aggregated_stats["reaching_waypoint"] / num_episodes
        episode_cant_reach_waypoint_mean = aggregated_stats["cant_reach_waypoint"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        logger.info(f"Average episode reaching_waypoint: {episode_reaching_waypoint_mean:.6f}")
        logger.info(f"Average episode cant_reach_waypoint: {episode_cant_reach_waypoint_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}")

        if not config.EVAL.SPLIT.startswith('test'):
            writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
            writer.add_scalar("{}/reaching_waypoint".format(config.EVAL.SPLIT), episode_reaching_waypoint_mean,
                              checkpoint_index)
            writer.add_scalar("{}/cant_reach_waypoint".format(config.EVAL.SPLIT), episode_cant_reach_waypoint_mean,
                              checkpoint_index)
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                                  checkpoint_index)

        self.envs.close()

        result = {'episode_reward_mean': episode_reward_mean,
                  'episode_reaching_waypoint_mean': episode_reaching_waypoint_mean,
                  'episode_cant_reach_waypoint_mean': episode_cant_reach_waypoint_mean}
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result


@baseline_registry.register_trainer(name="WhcAVWanTrainer")
class WhcPPOTrainer(BaseRLTrainer):
    r"""
        new PPO trainer from nav
    """
    supported_tasks = ["Nav-v0"]
    images = []

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

    def render(self, observations, infos):
        # cv2.imshow(window_name, image)
        for i in range(5):
            # wpmap = np.zeros((200, 200, 1))
            # set_trace()

            img = draw_top_down_map(infos[i])

            # img *= 100
            # set_trace()
            # cv2.imshow('demo'+str(i), img)
            # cv2.waitKey(0)
            # cv2.imwrite('/data/AudioVisual/sound-spaces/ss_baselines/av_wan/Demo/'+str(i)+'.png', img)
            self.images[i].append(img)

    # print("Agent stepping around inside environment.")

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None, sound_num=128) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]

        # 注意我们现在只传config！！！
        self.actor_critic = XcxAudioNavBaselinePolicy(observation_space=observation_space, sound_num=sound_num,
                                                      config=self.config)
        self.actor_critic.to(self.device)

        # 这个地方因为PPO基本上是固定不需要动的，所以可以暂时不改动，将来需要也可以改成传一个config
        self.agent = WyxPPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            classifier_lr=ppo_cfg.classifier_lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            config=ppo_cfg,
        )

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {"state_dict": self.agent.state_dict(), "config": self.config,
                      'optimizer_state': self.agent.optimizer.state_dict(), "update": self.update}
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name), _use_new_zipfile_serialization=True)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(self, rollouts, current_episode_reward, current_episode_step, episode_rewards,
                              episode_spls, episode_counts, episode_steps, episode_distances):
        pth_time = 0.0
        env_time = 0.0
        # whc 于2.13 2:00添加
        self.actor_critic.net.eval()

        t_sample_action = time.time()
        # sample actions

        with torch.no_grad():
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                distributions,
                _,
                _,
            ) = self.actor_critic.act(step_observation, rollouts.recurrent_hidden_states[rollouts.step],
                                      rollouts.prev_actions[rollouts.step], rollouts.masks[rollouts.step])

        # ipdb > step_observation.keys()
        # dict_keys(['action_map', 'am', 'collision', 'depth', 'ego_map', 'gm', 'intensity', 'pointgoal_with_gps_compass',
        #            'spectrogram'])
        # ipdb > rollouts.recurrent_hidden_states[rollouts.step]
        # tensor([[[0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.],
        #          [0., 0., 0., ..., 0., 0., 0.]]], device='cuda:0')
        # ipdb > rollouts.prev_actions[rollouts.step]
        # tensor([[0],
        #         [0],
        #         [0],
        #         [0],
        #         [0]], device='cuda:0')
        # ipdb > rollouts.masks[rollouts.step]
        # tensor([[1.],
        #         [1.],
        #         [1.],
        #         [1.],
        #         [1.]], device='cuda:0')

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        # compare
        outputs = self.envs.step([{"action": a[0].item()} for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        # self.render(observations, infos)
        # set_trace()
        # ipdb > infos[0]
        # {'distance_to_goal': 7.5, 'normalized_distance_to_goal': 1.1538461538461537, 'success': 0.0, 'spl': 0.0,
        #  'softspl': 0.0, 'na': 3, 'sna': 0.0, 'reaching_waypoint': True, 'cant_reach_waypoint': False,
        #  'waypoints_pos': (240, 250)}

        # ipdb > observations[0].keys()
        # dict_keys(
        #     ['depth', 'spectrogram', 'ego_map', 'pointgoal_with_gps_compass', 'gm', 'action_map', 'collision', 'am',
        #      'intensity'])

        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()

        # compare
        # infos = []
        for i in range(len(observations)):
            if "info" in observations[i].keys():
                # infos.append(observations[i]['info'])
                del observations[i]['info']

        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float)
        spls = torch.tensor([[info['spl']] for info in infos])
        distances = torch.tensor([[info['current_agent_dis_sound']] for info in infos])

        sound_ids = torch.LongTensor([[info['current_sound_id']] for info in infos])
        rotations = torch.tensor([[info['current_agent_rotation']] for info in infos])
        x_delta = torch.tensor([[info['x_delta']] for info in infos])
        y_delta = torch.tensor([[info['y_delta']] for info in infos])
        z_delta = torch.tensor([[info['z_delta']] for info in infos])

        current_episode_reward += rewards
        current_episode_step += 1
        # current_episode_reward is accumulating rewards across multiple updates,
        # as long as the current episode is not finished
        # the current episode reward is added to the episode rewards only if the current episode is done
        # the episode count will also increase by 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_spls += (1 - masks) * spls
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        episode_distances += (1 - masks) * distances
        current_episode_reward *= masks
        current_episode_step *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
            infos,
            sound_ids,
            x_delta,
            y_delta,
            z_delta,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts, lambda_grad=1.0):
        # 根据rollout套PPO算法
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {k: v[-1] for k, v in rollouts.observations.items()}
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.prev_actions[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)

        # value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
        # compare
        value_loss, action_loss, dist_entropy, classifier_loss, classifier_entropy, classifier_acc, regressor_loss, z_info = self.agent.update(rollouts, lambda_grad=lambda_grad)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            classifier_loss,
            classifier_entropy,
            classifier_acc,
            regressor_loss,
            z_info,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        global lr_lambda
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        for _ in range(5):
            self.images.append([])
        # set_trace()

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME), auto_reset_done=False)

        observations = self.envs.reset()
        info = observations[0]['info']
        infos = []
        for i in range(len(observations)):
            infos.append(observations[i]['info'])
            del observations[i]['info']
        sound_num = info['sound_num']

        batch = batch_obs(observations)  # compare SKIP_LIST
        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg, sound_num=sound_num)
        self.update_start = 0
        self.update = 0

        # 断点续训：加载第load_ckpt个checkpoint，接着训
        load_ckpt = ppo_cfg.load_ckpt
        if load_ckpt > 0:
            ckpt_path = os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{load_ckpt}.pth")
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.agent.to(self.device)
            self.agent.optimizer.load_state_dict(ckpt_dict["optimizer_state"])
            self.update = ckpt_dict["update"] + 1

        logger.info("agent number of parameters: {}".format(sum(param.numel() for param in self.agent.parameters())))

        # 同理，可改为只传config
        rollouts = WhcRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )

        # now rollouts are all Zero
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # compare
        # rollouts.info[0] = deepcopy(info)

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None
        info = None

        # episode_rewards and episode_counts accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_spls = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        episode_distances = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_spl = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_distances = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = load_ckpt
        prev_time = 0

        # 这里可能有bug: LR scheduler不一定保存了断点续训的状态
        if ppo_cfg.use_linear_lr_decay:

            def lr_lambda(x):
                return linear_decay(x, self.config.NUM_UPDATES)
        elif ppo_cfg.use_exponential_lr_decay:

            def lr_lambda(x):
                return exponential_decay(x, self.config.NUM_UPDATES, ppo_cfg.exp_decay_lambda)
        else:

            def lr_lambda(x):
                return 1

        lr_scheduler = LambdaLR(optimizer=self.agent.optimizer, lr_lambda=lr_lambda)

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            for update in range(self.update_start, self.config.NUM_UPDATES):
                self.update = update
                p = float(update / self.config.NUM_UPDATES)
                lambda_p = 0.8 / (1.0 + np.exp(-10 * p)) - 0.4
                lambda_p *= ppo_cfg.lambda_classifier

                if ppo_cfg.use_linear_lr_decay or ppo_cfg.use_exponential_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(update, self.config.NUM_UPDATES)

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = \
                        self._collect_rollout_step(rollouts, current_episode_reward, current_episode_step,
                                                   episode_rewards, episode_spls, episode_counts, episode_steps,
                                                   episode_distances)
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy, classifier_loss, classifier_entropy, \
                classifier_acc, regressor_loss, z_info = \
                    self._update_agent(ppo_cfg, rollouts, lambda_p)

                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_spl.append(episode_spls.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())
                window_episode_distances.append(episode_distances.clone())

                losses = [value_loss, action_loss, dist_entropy]
                ppo_loss = value_loss * ppo_cfg.value_loss_coef + action_loss - dist_entropy * ppo_cfg.entropy_coef
                total_loss = ppo_loss - lambda_p * classifier_loss + self.agent.config.lambda_regressor * regressor_loss
                stats = zip(
                    ["count", "reward", "step", 'spl', 'distance'],
                    [window_episode_counts, window_episode_reward, window_episode_step, window_episode_spl,
                     window_episode_distances],
                )
                deltas = {k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()) for k, v in stats}
                deltas["count"] = max(deltas["count"], 1.0)

                # this reward is averaged over all the episodes happened during window_size updates
                # approximately number of steps is window_size * num_steps
                if z_info is not None:
                    z_max, z_sin_label, z_sin_max = z_info
                if update % 10 == 0:
                    writer.add_scalar("Environment/Reward", deltas["reward"] / deltas["count"], count_steps)

                    writer.add_scalar("Environment/SPL", deltas["spl"] / deltas["count"], count_steps)

                    logging.debug('Number of steps: {}'.format(deltas["step"] / deltas["count"]))
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)

                    writer.add_scalar("Environment/Distance_to_goal", deltas["distance"] / deltas["count"], count_steps)

                    # writer.add_scalars(
                    #     "losses",
                    #     {k: l for l, k in zip(losses, ["value", "policy"])},
                    #     count_steps,
                    # )

                    writer.add_scalar('Policy/Value_Loss', value_loss, count_steps)
                    writer.add_scalar('Policy/Action_Loss', action_loss, count_steps)
                    writer.add_scalar('Policy/Entropy', dist_entropy, count_steps)
                    writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)
                    writer.add_scalar('Policy/Classifier_Loss', classifier_loss, count_steps)
                    writer.add_scalar('Policy/Classifier_Entropy', classifier_entropy, count_steps)
                    writer.add_scalar('Policy/Classifier_Acc', classifier_acc, count_steps)
                    writer.add_scalar('Policy/Regressor_Loss', regressor_loss, count_steps)
                    writer.add_scalar('Policy/Lambda_p', lambda_p, count_steps)
                    if z_info is not None:
                        writer.add_scalar('Policy/z_max', z_max, count_steps)
                        writer.add_scalar('Policy/z_sin_max', z_sin_max, count_steps)
                    writer.add_scalar('Policy/Total_Loss', total_loss, count_steps)
                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info("update: {}\tfps: {:.3f}\t".format(update, count_steps / (time.time() - t_start)))

                    logger.info("update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                                "frames: {}".format(update, env_time, pth_time, count_steps))

                    window_rewards = (window_episode_reward[-1] - window_episode_reward[0]).sum()
                    window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()

                    if window_counts > 0:
                        logger.info("Average window size {} reward: {:3f}".format(
                            len(window_episode_reward),
                            (window_rewards / window_counts).item(),
                        ))
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    self.actor_critic.net.train()
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # dzj
        # print("########")

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
        self.config = config
        config.defrost()
        config.RL.PPO = ckpt_dict["config"].RL.PPO
        config.freeze()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.TASK_CONFIG.TASK.SENSORS.append("AUDIOGOAL_SENSOR")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME), auto_reset_done=False)

        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 3)
        else:
            observation_space = self.envs.observation_spaces[0]

        # compare
        sound_ids = np.load(self.config.SOURCE_SOUND_IDS_PATH, allow_pickle=True).item()
        sound_num = len(set(sound_ids.values()))

        self._setup_actor_critic_agent(ppo_cfg, observation_space, sound_num=sound_num)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.net.eval()
        self.metric_uuids = []
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(metric_cfg.TYPE)
            # dys modified
            # set_trace()
            self.metric_uuids.append(
                measure_type(sim=None, task=None, config=config.TASK_CONFIG.TASK.TOP_DOWN_MAP)._get_uuid())

        observations = self.envs.reset()
        for i in range(len(observations)):
            del observations[i]['info']
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
        batch = batch_obs(observations, self.device)

        # compare
        # self._setup_actor_critic_agent(ppo_cfg, sound_num=infos[0]['sound_num'])

        # self.agent.load_state_dict(ckpt_dict["state_dict"])
        # self.actor_critic = self.agent.actor_critic

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_reaching_waypoint = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_cant_reach_waypoint = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_step_count = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [[] for _ in range(self.config.NUM_PROCESSES)]  # type: List[List[np.ndarray]]
        audios = [[] for _ in range(self.config.NUM_PROCESSES)]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)

        classify_success = 0
        step_count = 0

        writer_suffix = re.sub("data/sounds/1s_all", "", self.config.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_DIR)
        writer_suffix = re.sub("_", "/", writer_suffix)
        if hasattr(self.config, "DEPTH_NOISE_LEVEL"):
            writer_suffix += ("/depth_%.2f" % self.config.DEPTH_NOISE_LEVEL).replace(".", "")
        if hasattr(self.config, "AUDIO_NOISE_LEVEL"):
            writer_suffix += ("/audio_%.2f" % self.config.AUDIO_NOISE_LEVEL).replace(".", "")

        while (len(stats_episodes) < self.config.TEST_EPISODE_COUNT and self.envs.num_envs > 0):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states, _, predicted_labels, predicted_x_y,\
                    = self.actor_critic.act(batch, test_recurrent_hidden_states, prev_actions, not_done_masks,
                                            deterministic=True)  # Why True?

                prev_actions.copy_(actions)

            outputs = self.envs.step([{"action": a[0].item()} for a in actions])
            # outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observations[i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i])
                            rgb_frames[i].append(frame)
                        del observations[i]['intermediate']

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros(
                            (self.config.DISPLAY_RESOLUTION, self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]['audiogoal'])

            if config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)
            for i in range(len(observations)):
                if "info" in dict(observations[i]).keys():
                    del observations[i]['info']
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            logging.debug('Reward: {}'.format(rewards[0]))

            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(1)
            current_episode_reward += rewards
            current_episode_step_count += 1
            next_episodes = self.envs.current_episodes()
            n_envs = self.envs.num_envs
            envs_to_pause = []
            for i in range(n_envs):
                # pause envs which runs out of episodes
                if (
                        next_episodes[i].scene_id,
                        next_episodes[i].episode_id,
                        next_episodes[i].info["sound"],
                ) in stats_episodes:
                    envs_to_pause.append(i)
                    logging.info('Pause env {} and remaining number of envs: {}'.format(i, self.envs.num_envs))

                current_episode_reaching_waypoint[i] += infos[i]['reaching_waypoint']
                current_episode_cant_reach_waypoint[i] += infos[i]['cant_reach_waypoint']

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = norm(
                        np.array(current_episodes[i].goals[0].position) - np.array(current_episodes[i].start_position))
                    episode_stats["reaching_waypoint"] = current_episode_reaching_waypoint[i].item() / \
                                                         current_episode_step_count[i].item()
                    episode_stats["cant_reach_waypoint"] = current_episode_cant_reach_waypoint[i].item() / \
                                                           current_episode_step_count[i].item()
                    current_episode_reaching_waypoint[i] = 0
                    current_episode_cant_reach_waypoint[i] = 0
                    current_episode_step_count[i] = 0

                    sound_id = infos[i]['current_sound_id']
                    if predicted_labels is not None:
                        classify_success += int(torch.argmax(predicted_labels[i], dim=0) == sound_id)

                    current_episode_reward[i] = 0
                    logging.debug(episode_stats)
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                        current_episodes[i].info["sound"],
                    )] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        if self.config.VISUALIZE_FAILURE_ONLY and infos[i]['success'] > 0:
                            pass
                        else:
                            fps = self.config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
                                if self.config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
                            if 'sound' in current_episodes[i].info:
                                sound = current_episodes[i].info['sound']
                            else:
                                sound = current_episodes[i].sound_id.split('/')[1][:-4]

                            # set_trace()
                            generate_video(video_option=self.config.VIDEO_OPTION,
                                           video_dir=self.config.VIDEO_DIR,
                                           images=rgb_frames[i][:-1],
                                           scene_name=current_episodes[i].scene_id.split('/')[3],
                                           sound=sound,
                                           sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                                           episode_id=current_episodes[i].episode_id,
                                           checkpoint_idx=checkpoint_index,
                                           metric_name='spl',
                                           metric_value=infos[i]['spl'],
                                           tb_writer=writer,
                                           audios=audios[i][:-1],
                                           fps=fps)

                        rgb_frames[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        top_down_map = plot_top_down_map(infos[i])
                        scene = current_episodes[i].scene_id.split('/')[-3]
                        writer.add_image('{}_{}_{}/{}_{}'.format(config.EVAL.SPLIT + writer_suffix, scene,
                                                                 current_episodes[i].episode_id, sound,
                                                                 config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]),
                                         top_down_map, dataformats='WHC')

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum([v[stat_key] for v in stats_episodes.values()])
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(config.TENSORBOARD_DIR, '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_reaching_waypoint_mean = aggregated_stats["reaching_waypoint"] / num_episodes
        episode_cant_reach_waypoint_mean = aggregated_stats["cant_reach_waypoint"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        logger.info(f"Average episode reaching_waypoint: {episode_reaching_waypoint_mean:.6f}")
        logger.info(f"Average episode cant_reach_waypoint: {episode_cant_reach_waypoint_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}")

        # if not config.EVAL.SPLIT.startswith('test'):
        #     writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
        #     writer.add_scalar("{}/reaching_waypoint".format(config.EVAL.SPLIT), episode_reaching_waypoint_mean, checkpoint_index)
        #     writer.add_scalar("{}/cant_reach_waypoint".format(config.EVAL.SPLIT), episode_cant_reach_waypoint_mean, checkpoint_index)
        #     for metric_uuid in self.metric_uuids:
        #         writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid], checkpoint_index)
        writer.add_scalar("{}/reward".format(config.EVAL.SPLIT + writer_suffix), episode_reward_mean, checkpoint_index)
        writer.add_scalar("{}/reaching_waypoint".format(config.EVAL.SPLIT + writer_suffix),
                          episode_reaching_waypoint_mean,
                          checkpoint_index)
        writer.add_scalar("{}/cant_reach_waypoint".format(config.EVAL.SPLIT + writer_suffix),
                          episode_cant_reach_waypoint_mean,
                          checkpoint_index)
        for metric_uuid in self.metric_uuids:
            writer.add_scalar(f"{config.EVAL.SPLIT + writer_suffix}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                              checkpoint_index)

        self.envs.close()

        result = {'episode_reward_mean': episode_reward_mean,
                  'episode_reaching_waypoint_mean': episode_reaching_waypoint_mean,
                  'episode_cant_reach_waypoint_mean': episode_cant_reach_waypoint_mean}
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        result['classifier_acc'] = classify_success / num_episodes
        # if ppo_cfg.is_classify and ppo_cfg.is_tsne:
        #     print("start tsne")
        #     self.visualize_enccoder_with_model(os.path.dirname(self.config.LOG_FILE), checkpoint_index, sound_num)
        #     print("done")

        return result

    def _eval_demo(self, checkpoint_index: int = 0) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # dzj
        # print("########")
        self.video_num = 0
        self.s_a_name = dict()
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        # Map location CPU is almost always better than mapping to a CUDA device.
        # whc
        checkpoint_path = "/data/AudioVisual/sound-spaces/data/pretrained_weights/audionav/av_wan/replica/unheard.pth"
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
        # config = self.config.clone()
        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.TASK_CONFIG.TASK.SENSORS.append("AUDIOGOAL_SENSOR")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME), auto_reset_done=False)

        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 3)
        else:
            observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        self.metric_uuids = []
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(metric_cfg.TYPE)
            # dys modified
            # set_trace()
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_reaching_waypoint = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_cant_reach_waypoint = torch.zeros(self.envs.num_envs, 1, device=self.device)
        current_episode_step_count = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [[] for _ in range(self.config.NUM_PROCESSES)]  # type: List[List[np.ndarray]]
        audios = [[] for _ in range(self.config.NUM_PROCESSES)]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            while (len(stats_episodes) < self.config.TEST_EPISODE_COUNT and self.envs.num_envs > 0):
                current_episodes = self.envs.current_episodes()

                with torch.no_grad():
                    _, actions, _, test_recurrent_hidden_states, distributions = self.actor_critic.act(batch,
                                                                                                       test_recurrent_hidden_states,
                                                                                                       prev_actions,
                                                                                                       not_done_masks,
                                                                                                       deterministic=True)

                    prev_actions.copy_(actions)

                outputs = self.envs.step([{"action": a[0].item()} for a in actions])
                observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
                if config.DISPLAY_RESOLUTION != model_resolution:
                    resize_observation(observations, model_resolution)

                batch = batch_obs(observations, self.device)
                if len(self.config.VIDEO_OPTION) > 0:
                    rgb_frames[0] += infos[0]['rgb_frames']
                    audios[0] += infos[0]['audios']

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )
                logging.debug('Reward: {}'.format(rewards[0]))

                rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(1)
                current_episode_reward += rewards
                current_episode_step_count += 1
                next_episodes = self.envs.current_episodes()
                n_envs = self.envs.num_envs
                envs_to_pause = []
                for i in range(n_envs):
                    # pause envs which runs out of episodes
                    if (
                            next_episodes[i].scene_id,
                            next_episodes[i].episode_id,
                    ) in stats_episodes:
                        envs_to_pause.append(i)
                        logging.info('Pause env {} and remaining number of envs: {}'.format(i, self.envs.num_envs))

                    current_episode_reaching_waypoint[i] += infos[i]['reaching_waypoint']
                    current_episode_cant_reach_waypoint[i] += infos[i]['cant_reach_waypoint']

                    # episode ended
                    if not_done_masks[i].item() == 0:
                        episode_stats = dict()
                        for metric_uuid in self.metric_uuids:
                            episode_stats[metric_uuid] = infos[i][metric_uuid]
                        episode_stats["reward"] = current_episode_reward[i].item()
                        episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                        episode_stats['euclidean_distance'] = norm(
                            np.array(current_episodes[i].goals[0].position) - np.array(
                                current_episodes[i].start_position))
                        episode_stats["reaching_waypoint"] = current_episode_reaching_waypoint[i].item() / \
                                                             current_episode_step_count[i].item()
                        episode_stats["cant_reach_waypoint"] = current_episode_cant_reach_waypoint[i].item() / \
                                                               current_episode_step_count[i].item()
                        current_episode_reaching_waypoint[i] = 0
                        current_episode_cant_reach_waypoint[i] = 0
                        current_episode_step_count[i] = 0
                        current_episode_reward[i] = 0
                        logging.debug(episode_stats)
                        # use scene_id + episode_id as unique id for storing stats
                        stats_episodes[(
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )] = episode_stats
                        t.update()

                        if len(self.config.VIDEO_OPTION) > 0:
                            if self.config.VISUALIZE_FAILURE_ONLY and infos[i]['success'] > 0:
                                pass
                            else:
                                fps = self.config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
                                    if self.config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
                                if 'sound' in current_episodes[i].info:
                                    sound = current_episodes[i].info['sound']
                                else:
                                    sound = current_episodes[i].sound_id.split('/')[1][:-4]

                                # set_trace()
                                s_name = current_episodes[i].scene_id.split('/')[3]
                                if s_name in self.s_a_name.keys():
                                    self.s_a_name[s_name] += 1
                                else:
                                    self.s_a_name[s_name] = 1
                                if self.video_num < 500 and self.s_a_name[s_name] <= 3 and infos[i]['success'] > 0:
                                    self.video_num += 1

                                    generate_video(video_option=self.config.VIDEO_OPTION,
                                                   video_dir=self.config.VIDEO_DIR,
                                                   images=rgb_frames[i][:-1],
                                                   scene_name=current_episodes[i].scene_id.split('/')[3],
                                                   sound=sound,
                                                   sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                                                   episode_id=current_episodes[i].episode_id,
                                                   checkpoint_idx=checkpoint_index,
                                                   metric_name='spl',
                                                   metric_value=infos[i]['spl'],
                                                   tb_writer=writer,
                                                   audios=audios[i][:-1],
                                                   fps=fps)

                            rgb_frames[i] = []
                            audios[i] = []

                        if "top_down_map" in self.config.VISUALIZATION_OPTION:
                            top_down_map = plot_top_down_map(infos[i])
                            scene = current_episodes[i].scene_id.split('/')[-3]
                            writer.add_image(
                                '{}_{}_{}/{}'.format(config.EVAL.SPLIT, scene, current_episodes[i].episode_id,
                                                     config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]), top_down_map,
                                dataformats='WHC')

                (
                    self.envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    self.envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum([v[stat_key] for v in stats_episodes.values()])
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(config.TENSORBOARD_DIR, '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_reaching_waypoint_mean = aggregated_stats["reaching_waypoint"] / num_episodes
        episode_cant_reach_waypoint_mean = aggregated_stats["cant_reach_waypoint"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        logger.info(f"Average episode reaching_waypoint: {episode_reaching_waypoint_mean:.6f}")
        logger.info(f"Average episode cant_reach_waypoint: {episode_cant_reach_waypoint_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}")

        # if not config.EVAL.SPLIT.startswith('test'):
        #     writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
        #     writer.add_scalar("{}/reaching_waypoint".format(config.EVAL.SPLIT), episode_reaching_waypoint_mean, checkpoint_index)
        #     writer.add_scalar("{}/cant_reach_waypoint".format(config.EVAL.SPLIT), episode_cant_reach_waypoint_mean, checkpoint_index)
        #     for metric_uuid in self.metric_uuids:
        #         writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid], checkpoint_index)

        self.envs.close()

        result = {'episode_reward_mean': episode_reward_mean,
                  'episode_reaching_waypoint_mean': episode_reaching_waypoint_mean,
                  'episode_cant_reach_waypoint_mean': episode_cant_reach_waypoint_mean}
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result

    def visualize_encoder(self, model_dir, prev_checkpoint_index, num_classes, args=None) -> None:
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))
        model_resolution = self.config.DISPLAY_RESOLUTION
        ppo_cfg = self.config.RL.PPO
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 1)
        else:
            observation_space = self.envs.observation_spaces[0]

        sound_ids_for_training = np.load(self.config.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_IDS_PATH,
                                         allow_pickle=True).item()

        sound_num = len(set(sound_ids_for_training.values()))
        start_check = prev_checkpoint_index
        while True:
            self._setup_actor_critic_agent(ppo_cfg, observation_space, sound_num=sound_num)
            ckpt_dict = self.load_checkpoint(os.path.join(model_dir, "data", "ckpt.%d.pth" % start_check),
                                             map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic
            self.visualize_encoder_with_model(model_dir, start_check, num_classes)
            start_check += args.eval_interval

    def process_visualization_data_loader(self, num_classes):
        data_samples = 20000

        file_dir = "/home/user/Documents/pre_train_data"  # self.config.PRE_TRAIN_DATA_PATH
        # path2model = "/data/AudioVisual/pre_train/2021-11-09-17-10-15/model/0-130-6.757573768496513.pth"
        # model_CKPT = torch.load(path2model, map_location=lambda storage, loc: storage)
        # self.actor_critic.net.audio_encoder.load_state_dict(model_CKPT)

        name_list = []
        spectrograms = []
        for file in os.listdir(file_dir):
            scene_name = os.path.join(file_dir, file)
            for spkl in os.listdir(scene_name):
                if spkl[-3:] == "pkl":
                    name_list.append(os.path.join(scene_name, spkl))
        l_name = len(name_list)
        index_name = random.sample(range(0, l_name), min(30, l_name))
        for i in index_name:
            spectrograms.append(pd.read_pickle(name_list[i]))
            # break
        self.sound_ids_in_sim = np.load("/data/AudioVisual/sound-spaces/data/sounds/sound_ids_1s_all.npy",
                                        allow_pickle=True).item()
        sound_ids_for_training = np.load(self.config.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_IDS_PATH,
                                         allow_pickle=True).item()

        sound_num = len(set(sound_ids_for_training.values()))
        selected_sound_ids = []
        for k in sound_ids_for_training.keys():
            if self.sound_ids_in_sim[k] not in selected_sound_ids:
                selected_sound_ids.append(self.sound_ids_in_sim[k])
        # np.random.shuffle(selected_sound_ids)
        selected_sound_ids = selected_sound_ids[:min(10, num_classes)]
        if sound_num >= num_classes:
            appended_sound_ids = list(range(len(set(self.sound_ids_in_sim.values()))))
            for sound_id in selected_sound_ids:
                appended_sound_ids.remove(sound_id)
            np.random.shuffle(appended_sound_ids)
            selected_sound_ids += appended_sound_ids[:max(0, num_classes - sound_num)]

        label_r = []
        spectrograms_tensor = []
        selected_num = {sound_id: 0 for sound_id in selected_sound_ids}
        for spect in spectrograms:
            exist_keys = []
            for p in spect:
                for key in p.keys():
                    if key.find("c_") == 0:
                        key_tmp = key[2:].split('.')[0]
                    else:
                        key_tmp = key.split('.')[0]
                    if key != "rot" and key != "dis" and key_tmp not in exist_keys \
                            and self.sound_ids_in_sim[key_tmp] in selected_sound_ids \
                            and selected_num[self.sound_ids_in_sim[key_tmp]] < data_samples / num_classes:
                        spectrograms_tensor.append(torch.from_numpy(p[key][np.newaxis, :]))
                        label_r.append(self.sound_ids_in_sim[key_tmp])
                        exist_keys.append(key_tmp)
                        selected_num[self.sound_ids_in_sim[key_tmp]] += 1

        test_spectrograms_tensor = torch.stack(spectrograms_tensor, dim=0).squeeze(dim=1)
        test_label_tensor = torch.tensor(label_r)
        test_dataset = TensorDataset(test_spectrograms_tensor, test_label_tensor)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

    def visualize_encoder_with_model(self, model_dir, prev_checkpoint_index, num_classes) -> None:
        # if not hasattr(self, 'test_loader'):
        self.process_visualization_data_loader(num_classes)

        X = []
        labels = []
        spects = dict()

        self.actor_critic.net.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                spects["spectrogram"], sound_id = data
                spects["spectrogram"] = spects["spectrogram"].to(self.device)
                sound_id = sound_id.to(self.device)
                feature = self.actor_critic.net.audio_encoder(spects).squeeze(0)
                X.append(feature)
                labels.append(sound_id)
            X = torch.cat(X, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
        self.plot_tsne(X, labels, prev_checkpoint_index, model_dir)
        self.test_loader = None

    def plot_tsne(self, X, labels, idx, model_dir):
        color_dict = {
            'aliceblue': '#F0F8FF',
            'antiquewhite': '#FAEBD7',
            'aqua': '#00FFFF',
            'aquamarine': '#7FFFD4',
            'azure': '#F0FFFF',
            'beige': '#F5F5DC',
            'bisque': '#FFE4C4',
            'black': '#000000',
            'blanchedalmond': '#FFEBCD',
            'blue': '#0000FF',
            'blueviolet': '#8A2BE2',
            'brown': '#A52A2A',
            'burlywood': '#DEB887',
            'cadetblue': '#5F9EA0',
            'chartreuse': '#7FFF00',
            'chocolate': '#D2691E',
            'coral': '#FF7F50',
            'cornflowerblue': '#6495ED',
            'cornsilk': '#FFF8DC',
            'crimson': '#DC143C',
            'cyan': '#00FFFF',
            'darkblue': '#00008B',
            'darkcyan': '#008B8B',
            'darkgoldenrod': '#B8860B',
            'darkgray': '#A9A9A9',
            'darkgreen': '#006400',
            'darkkhaki': '#BDB76B',
            'darkmagenta': '#8B008B',
            'darkolivegreen': '#556B2F',
            'darkorange': '#FF8C00',
            'darkorchid': '#9932CC',
            'darkred': '#8B0000',
            'darksalmon': '#E9967A',
            'darkseagreen': '#8FBC8F',
            'darkslateblue': '#483D8B',
            'darkslategray': '#2F4F4F',
            'darkturquoise': '#00CED1',
            'darkviolet': '#9400D3',
            'deeppink': '#FF1493',
            'deepskyblue': '#00BFFF',
            'dimgray': '#696969',
            'dodgerblue': '#1E90FF',
            'firebrick': '#B22222',
            'floralwhite': '#FFFAF0',
            'forestgreen': '#228B22',
            'fuchsia': '#FF00FF',
            'gainsboro': '#DCDCDC',
            'ghostwhite': '#F8F8FF',
            'gold': '#FFD700',
            'goldenrod': '#DAA520',
            'gray': '#808080',
            'green': '#008000',
            'greenyellow': '#ADFF2F',
            'honeydew': '#F0FFF0',
            'hotpink': '#FF69B4',
            'indianred': '#CD5C5C',
            'indigo': '#4B0082',
            'ivory': '#FFFFF0',
            'khaki': '#F0E68C',
            'lavender': '#E6E6FA',
            'lavenderblush': '#FFF0F5',
            'lawngreen': '#7CFC00',
            'lemonchiffon': '#FFFACD',
            'lightblue': '#ADD8E6',
            'lightcoral': '#F08080',
            'lightcyan': '#E0FFFF',
            'lightgoldenrodyellow': '#FAFAD2',
            'lightgreen': '#90EE90',
            'lightgray': '#D3D3D3',
            'lightpink': '#FFB6C1',
            'lightsalmon': '#FFA07A',
            'lightseagreen': '#20B2AA',
            'lightskyblue': '#87CEFA',
            'lightslategray': '#778899',
            'lightsteelblue': '#B0C4DE',
            'lightyellow': '#FFFFE0',
            'lime': '#00FF00',
            'limegreen': '#32CD32',
            'linen': '#FAF0E6',
            'magenta': '#FF00FF',
            'maroon': '#800000',
            'mediumaquamarine': '#66CDAA',
            'mediumblue': '#0000CD',
            'mediumorchid': '#BA55D3',
            'mediumpurple': '#9370DB',
            'mediumseagreen': '#3CB371',
            'mediumslateblue': '#7B68EE',
            'mediumspringgreen': '#00FA9A',
            'mediumturquoise': '#48D1CC',
            'mediumvioletred': '#C71585',
            'midnightblue': '#191970',
            'mintcream': '#F5FFFA',
            'mistyrose': '#FFE4E1',
            'moccasin': '#FFE4B5',
            'navajowhite': '#FFDEAD',
            'navy': '#000080',
            'oldlace': '#FDF5E6',
            'olive': '#808000',
            'olivedrab': '#6B8E23',
            'orange': '#FFA500',
            'orangered': '#FF4500',
            'orchid': '#DA70D6',
            'palegoldenrod': '#EEE8AA',
            'palegreen': '#98FB98',
            'paleturquoise': '#AFEEEE',
            'palevioletred': '#DB7093',
            'papayawhip': '#FFEFD5',
            'peachpuff': '#FFDAB9',
            'peru': '#CD853F',
            'pink': '#FFC0CB',
            'plum': '#DDA0DD',
            'powderblue': '#B0E0E6',
            'purple': '#800080',
            'red': '#FF0000',
            'rosybrown': '#BC8F8F',
            'royalblue': '#4169E1',
            'saddlebrown': '#8B4513',
            'salmon': '#FA8072',
            'sandybrown': '#FAA460',
            'seagreen': '#2E8B57',
            'seashell': '#FFF5EE',
            'sienna': '#A0522D',
            'silver': '#C0C0C0',
            'skyblue': '#87CEEB',
            'slateblue': '#6A5ACD',
            'slategray': '#708090',
            'snow': '#FFFAFA',
            'springgreen': '#00FF7F',
            'steelblue': '#4682B4',
            'tan': '#D2B48C',
            'teal': '#008080',
            'thistle': '#D8BFD8',
            'tomato': '#FF6347',
            'turquoise': '#40E0D0',
            'violet': '#EE82EE',
            'wheat': '#F5DEB3',
            'white': '#FFFFFF',
            'whitesmoke': '#F5F5F5',
            'yellow': '#FFFF00',
            'yellowgreen': '#9ACD32'
        }

        colors = [color_dict['royalblue'], color_dict['yellowgreen'], color_dict['coral'], color_dict['yellow'],
                  color_dict['black'], color_dict['goldenrod'], color_dict['skyblue'], color_dict['mediumpurple'],
                  color_dict['lightslategray'], color_dict['mediumseagreen']]

        sorted_labels = sorted(np.unique(labels))
        labels_num = [sorted_labels.index(label) for label in labels]
        colors = np.choose(labels_num, colors)

        def get_sound_name(value):
            for k, v in self.sound_ids_in_sim.items():
                if v == value:
                    return re.sub("^c_|[_0-9]*$", "", k)

        labels = [get_sound_name(label) for label in labels]

        model = TSNE()
        np.set_printoptions(suppress=True)

        Y = model.fit_transform(X)  # 将X降维(默认二维)后保存到Y中

        exist_labels = []
        for i in tqdm(range(len(labels))):
            if labels[i] not in exist_labels:
                plt.scatter(Y[i][0], Y[i][1], 50, colors[i], label=labels[i])
                exist_labels.append(labels[i])
            else:
                plt.scatter(Y[i][0], Y[i][1], 50, colors[i])

        plt.title("T-SNE for encoding after %d updates" % (idx * 50))
        plt.legend()
        if not os.path.exists(os.path.join(model_dir, "tsne")):
            os.makedirs(os.path.join(model_dir, "tsne"))
        path = os.path.join(model_dir, "tsne")
        plt.savefig(os.path.join(path, "encoder_tsne_for_%d.png" % idx))
        # plt.show()
        plt.close()
