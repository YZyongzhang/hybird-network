#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging
from ipdb import set_trace
import os

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat import Config, Dataset
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.utils import euler_from_quaternion


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="DYS")
class AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)
        # goal_radius = config['TASK_CONFIG']['TASK']['SUCCESS_DISTANCE']
        # self.follower = ShortestPathFollower(
        #     # 第一个是simulator，第二个是定义离goal距离多少时算达到目标了
        #     self._env.sim._sim, goal_radius, False
        # )

    # reset没有infos函数  只能用observations
    def hack_observations(self, observations):
        goals = self._env.current_episode.goals[0].position
        best_action = self._env.sim.get_oracle_action()
        episode_id = self.habitat_env.current_episode.episode_id
        agent_state = self._env.sim._sim.get_agent_state()
        observations['best_action'] = best_action
        observations['goals'] = goals
        observations['episode_id'] = episode_id
        observations['position'] = agent_state.position
        observations['rotation'] = euler_from_quaternion(agent_state.rotation)
        observations['orientation'] = self._env.sim.get_orientation()
        observations['index'] = self._env.sim._position_to_index(agent_state.position)
        observations['graph'] = self._env.sim.graph
        observations['grid_size'] = self._env.sim.config.GRID_SIZE
        observations['scene_name'] = self._env.sim.current_scene_name
        # # set_trace()

        # self._env.sim.compute_semantic_index_mapping()
        # observations['semantic_scene'] = self._env.sim._instance2label_mapping

        #observations['semantic_scene'] = self._env.sim._sim.semantic_scene

        return observations

    def hack_infos(self, infos):
        goals = self._env.current_episode.goals[0].position
        best_action = self._env.sim.get_oracle_action()
        episode_id = self.habitat_env.current_episode.episode_id
        infos['best_action'] = best_action
        infos['goals'] = goals
        infos['episode_id'] = episode_id
        # print(episode_id)
        return infos

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        logging.debug(super().current_episode)
        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return self.hack_observations(observations)

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        observations, rewards, dones, infos = super().step(*args, **kwargs)
        return self.hack_observations(observations), rewards, dones, infos
        # return observations, rewards, dones, self.hack_infos(infos)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            # if current_target_distance < self._previous_target_distance:
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_positions = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(
            current_position, target_positions
        )
        return distance

    def _episode_success(self):
        if (
            self._env.task.is_stop_called
            # and self._distance_target() < self._success_distance
            and self._env.sim.reaching_goal
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id
