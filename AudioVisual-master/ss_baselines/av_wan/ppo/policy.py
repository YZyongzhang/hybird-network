#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from ipdb import set_trace
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNetWithMask
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_wan.models.visual_cnn import VisualCNN, WhcVisualCNN
from ss_baselines.av_wan.models.map_cnn import MapCNN, WhcMapCNN
from ss_baselines.av_wan.models.audio_cnn import AudioCNN, WhcAudioCNN
from ipdb import set_trace
from ss_baselines.av_nav.ppo.attention.attention import BasicAttention
from ss_baselines.common.utils import Flatten
from ss_baselines.av_nav.models.visual_cnn import conv_output_dim, layer_init
from torch.autograd import Function
# from zmq import device
import numpy as np
import torch.nn.functional as F

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):

    def __init__(self, net, dim_actions, masking=True):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNetWithMask(self.net.output_size, self.dim_actions, masking)
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
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)
        # set_trace()

        # 表示是否是确定性的，确定性的就直接按概率最大的来，不确定性的还要sample一下
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        # loss在这里！！
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

    def __init__(self, observation_space, goal_sensor_uuid, masking, action_map_size, hidden_size=512, encode_rgb=False,
                 encode_depth=False):
        super().__init__(
            AudioNavBaselineNet(observation_space=observation_space, hidden_size=hidden_size,
                                goal_sensor_uuid=goal_sensor_uuid, encode_rgb=encode_rgb, encode_depth=encode_depth),
            # action_space.n,
            action_map_size ** 2,
            masking=masking)


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

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, encode_rgb, encode_depth):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._spectrogram = False
        self._gm = 'gm' in observation_space.spaces
        self._am = 'am' in observation_space.spaces

        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid
        self.visual_encoder = VisualCNN(observation_space, hidden_size, encode_rgb, encode_depth)
        if self._spectrogram:
            self.audio_encoder = AudioCNN(observation_space, hidden_size)
        if self._gm:
            self.gm_encoder = MapCNN(observation_space, hidden_size, map_type='gm')
        if self._am:
            self.am_encoder = MapCNN(observation_space, hidden_size, map_type='am')

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._hidden_size if self._spectrogram else 0) + \
                         (self._hidden_size if self._gm else 0) + \
                         (self._hidden_size if self._am else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and encode_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces and encode_depth:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if 'spectrogram' in observation_space.spaces:
            audio_shape = observation_space.spaces['spectrogram'].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        if self._gm:
            gm_shape = observation_space.spaces['gm'].shape
            summary(self.gm_encoder.cnn, (gm_shape[2], gm_shape[0], gm_shape[1]), device='cpu')
        if self._am:
            am_shape = observation_space.spaces['am'].shape
            summary(self.am_encoder.cnn, (am_shape[2], am_shape[0], am_shape[1]), device='cpu')

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

        if self._spectrogram:
            x.append(self.audio_encoder(observations))
        if self._gm:
            x.append(self.gm_encoder(observations))
        if self._am:
            x.append(self.am_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1


class WyxAudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, encode_rgb, encode_depth, sound_num):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._spectrogram = False
        self._gm = 'gm' in observation_space.spaces
        self._am = 'am' in observation_space.spaces

        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid
        self.visual_encoder = VisualCNN(observation_space, hidden_size, encode_rgb, encode_depth)
        if self._spectrogram:
            self.audio_encoder = AudioCNN(observation_space, hidden_size)
        if self._gm:
            self.gm_encoder = MapCNN(observation_space, hidden_size, map_type='gm')
        if self._am:
            self.am_encoder = MapCNN(observation_space, hidden_size, map_type='am')

        self.rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                              (self._hidden_size if self._spectrogram else 0) + \
                              (self._hidden_size if self._gm else 0) + \
                              (self._hidden_size if self._am else 0)
        self.state_encoder = RNNStateEncoder(self.rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and encode_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces and encode_depth:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if 'spectrogram' in observation_space.spaces:
            audio_shape = observation_space.spaces['spectrogram'].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        if self._gm:
            gm_shape = observation_space.spaces['gm'].shape
            summary(self.gm_encoder.cnn, (gm_shape[2], gm_shape[0], gm_shape[1]), device='cpu')
        if self._am:
            am_shape = observation_space.spaces['am'].shape
            summary(self.am_encoder.cnn, (am_shape[2], am_shape[0], am_shape[1]), device='cpu')

        self.v_gm_attention = BasicAttention(hidden_size, hidden_size, hidden_size)
        self.a_gm_attention = BasicAttention(hidden_size, hidden_size, hidden_size)
        self.attention_cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(16 * hidden_size, hidden_size),
            nn.ReLU(True),
        )
        layer_init(self.attention_cnn)

        self.sound_num = sound_num
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
        x_v = None
        x_a = None
        x_gm = None

        if self._spectrogram:
            x_a = self.audio_encoder(observations)
            # x.append(x_a)
        if self._gm:
            x_gm = self.gm_encoder(observations)
            x.append(x_gm)
        if self._am:
            x.append(self.am_encoder(observations))
        if not self.is_blind:
            x_v = self.visual_encoder(observations)
            # x.append(x_v)

        x_v_gm = self.v_gm_attention(x_v, x_gm, x_gm).squeeze(1)
        x_a_gm = self.a_gm_attention(x_a, x_gm, x_gm).squeeze(1)

        x.append(x_v_gm)
        x.append(x_a_gm)

        # x_attention = self.attention_cnn(x_attention)
        # x.append(x_attention)

        x1 = torch.cat(x, dim=1)  # shape = (1, 2048)

        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1, x_v, x_a


class XcxAudioNavBaselineNet(Net):

    def __init__(self, observation_space, config, sound_num):
        super().__init__()
        self.goal_sensor_uuid = config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID
        self._hidden_size = config.RL.PPO.hidden_size
        self._spectrogram = False
        self._gm = 'gm' in observation_space.spaces
        self._am = 'am' in observation_space.spaces
        self.config = config.RL.PPO
        self.sound_num = sound_num

        # 是否encode visual image, 若否，并且以下所有参数都否，则等同于原版wan
        self.is_visual_encode = config.RL.PPO['is_visual_encode']

        # 是否含av cross attention, map_attention, classifier, regression
        self.is_cross_att = config.RL.PPO['is_cross_attention']
        self.is_map_att = config.RL.PPO['is_map_attention']
        self.is_classify = config.RL.PPO['is_classify']
        self.is_visual_classify = config.RL.PPO['is_visual_classify']
        self.is_regression = config.RL.PPO['is_regression']
        self.reg_behind = hasattr(config.RL.PPO, "reg_behind") and config.RL.PPO.reg_behind
        self.double_cnn = hasattr(config.RL.PPO, "double_cnn") and config.RL.PPO.double_cnn
        self.normalized_confd_map = hasattr(config.RL.PPO,
                                            "normalized_confd_map") and config.RL.PPO.normalized_confd_map
        self.ENCODE_RGB = config.ENCODE_RGB
        self.ENCODE_DEPTH = config.ENCODE_DEPTH

        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid

        # start net definition
        if self.is_visual_encode:
            self.visual_encoder = WhcVisualCNN(observation_space, self._hidden_size, config.ENCODE_RGB,
                                               config.ENCODE_DEPTH, config.RL.PPO)
        else:
            # 为了防止下面summary出错，需要保证RGB和depth都不能被encode
            # 并且任何情况下都不能有cross attention
            # assert not config.ENCODE_RGB
            # assert not config.ENCODE_DEPTH
            self.ENCODE_RGB = False
            self.ENCODE_DEPTH = False
            assert not self.is_cross_att
            # self.is_bilnd = True


        self.audio_hidden_size = self._hidden_size
        self.gm_hidden_size = self._hidden_size

        if self._spectrogram:
            self.audio_encoder = WhcAudioCNN(observation_space, self.audio_hidden_size, config.RL.PPO)

        if self._gm:
            self.gm_encoder = WhcMapCNN(observation_space, self.gm_hidden_size, map_type='gm',
                                        config=config.RL.PPO)  # now dim(x_av = n)
        if self._am:
            self.am_encoder = WhcMapCNN(observation_space, self.gm_hidden_size, map_type='am', config=config.RL.PPO)
        # # 注意：需要分情况讨论GRU dim
        # # 在不encode visual的情况下RNN size应该为2n
        # if self.is_map_att:
        #     self.rnn_input_size = 2 * self._hidden_size
        # else:
        #     self.rnn_input_size = 4 * self._hidden_size
        # # 如果不encode visual的话，rnn的输入实际上只有一半，因为x_av只有原来的一半x_a
        # if not self.is_visual_encode:
        #     self.rnn_input_size = self.rnn_input_size // 2
        self.rnn_input_size = 0
        if self._spectrogram:
            self.rnn_input_size += self.audio_hidden_size
        if self._gm:
            self.rnn_input_size += self.gm_hidden_size
        if self._am:
            self.rnn_input_size += self.gm_hidden_size
        if self.is_visual_encode and config.ENCODE_DEPTH:
            self.rnn_input_size += self._hidden_size
        self.state_encoder = RNNStateEncoder(self.rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and self.ENCODE_RGB:
            # 注意因为如果不encode visual，会改这个私有变量，但是改不了config 里面的变量，所以这个地方需要用self.ENCODE_RGB
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces and self.ENCODE_DEPTH:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if 'spectrogram' in observation_space.spaces:
            audio_shape = observation_space.spaces['spectrogram'].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        if self._gm:
            gm_shape = observation_space.spaces['gm'].shape
            summary(self.gm_encoder.cnn, (gm_shape[2], gm_shape[0], gm_shape[1]), device='cpu')
        if self._am:
            am_shape = observation_space.spaces['am'].shape
            summary(self.am_encoder.cnn, (am_shape[2], am_shape[0], am_shape[1]), device='cpu')

            # if self.is_cross_att:
            #     self.v_a_attention = BasicAttention(self._hidden_size, self._hidden_size, self._hidden_size)
            #     self.a_v_attention = BasicAttention(self._hidden_size, self._hidden_size, self._hidden_size)

            # if self.is_map_att:
            #     if self.is_visual_encode:
            #         self.av_gm_attention = BasicAttention(2 * self._hidden_size, 2 * self._hidden_size, 2 * self._hidden_size)
            #     else:
            #         self.av_gm_attention = BasicAttention(self._hidden_size, self._hidden_size, self._hidden_size)
            '''
            self.map_attention_cnn = nn.Sequential(
                # 需要写一个 view成二维
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                Flatten(),
                nn.Linear(32 * self._hidden_size, ((int)(self.is_visual_encode) + 1) * self._hidden_size),
                # if self.is_visual_encode, dim is 2n, else n
                nn.ReLU(True),
            )
            layer_init(self.map_attention_cnn)
            '''
        if hasattr(config.RL.PPO, 'classifier_behind'):
            self.classifier_behind = config.RL.PPO.classifier_behind
        else:
            self.classifier_behind = False
        if config.RL.PPO.is_classify:
            if config.RL.PPO.is_visual_classify or self.classifier_behind:  # 同时classify visual+audio
                # assert self.is_visual_encode
                self.classifier = Sound_Classifier2(2 * self._hidden_size, sound_num,
                                                    num_layers=config.RL.PPO.classifier_num_layers)  # new classifier!
            else:  # 只classify audio
                self.classifier = Sound_Classifier2(self._hidden_size, sound_num,
                                                    num_layers=config.RL.PPO.classifier_num_layers)

        if config.RL.PPO.is_regression:
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
        if self.is_visual_encode:
            return self.visual_encoder.is_blind
        else:
            return True

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0):
        x = []
        x_gm = None
        x_am = None
        x_a = None
        x_v = None
        x_av = None  # concat for attention

        if self._spectrogram:
            x_a = self.audio_encoder(observations)
            x.append(x_a)
        if self._gm:
            x_gm = self.gm_encoder(observations)
            x.append(x_gm)

        if self._am:
            x_am = self.am_encoder(observations)
            x.append(x_am)

        if self.is_visual_encode:
            x_v = self.visual_encoder(observations)
            x.append(x_v)

        x1 = torch.cat(x, dim=1)

        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        if self.is_classify:
            if self.is_visual_classify:
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

        if self.is_regression:
            if self.reg_behind:
                predicted_rot = self.regressor(x2)
            else:
                predicted_rot = self.regressor(x_a)
        else:
            predicted_rot = None

        return x2, rnn_hidden_states1, predicted_labels, predicted_rot,


class WyxAudioNavBaselinePolicy(Policy):

    def __init__(
            self,
            observation_space,
            goal_sensor_uuid,
            masking,
            action_map_size,
            hidden_size=512,
            encode_rgb=False,
            encode_depth=False,
            sound_num=None,
    ):
        super().__init__(
            WyxAudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                encode_rgb=encode_rgb,
                encode_depth=encode_depth,
                sound_num=sound_num,
            ),
            # action_space.n,
            action_map_size ** 2,
            masking=masking)
        self.classifier = Sound_Classifier(hidden_size * 2, sound_num)
        self.regressor = X_Y_Regresor(self.net.rnn_input_size, 2)

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
    ):
        features, rnn_hidden_states, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)
        # set_trace()

        # 表示是否是确定性的，确定性的就直接按概率最大的来，不确定性的还要sample一下
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states, v_features, a_features = self.net(observations, rnn_hidden_states, prev_actions,
                                                                       masks)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        # loss在这里！！
        distribution_entropy = distribution.entropy().mean()

        v_a_features = torch.cat([v_features, a_features], dim=1)
        predicted_labels_detached = self.classifier(v_a_features.detach())
        predicted_labels = self.classifier(v_a_features)
        predicted_x_y = self.regressor(features)

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, predicted_labels_detached, predicted_labels, predicted_x_y


class XcxAudioNavBaselinePolicy(Policy):

    def __init__(
            self,
            observation_space,
            # goal_sensor_uuid,
            # masking,
            # action_map_size,
            # hidden_size=512,
            # encode_rgb=False,
            # encode_depth=False,
            sound_num=128,
            config=None):
        super().__init__(
            XcxAudioNavBaselineNet(
                observation_space=observation_space,
                # hidden_size=hidden_size,
                # goal_sensor_uuid=goal_sensor_uuid,
                # encode_rgb=encode_rgb,
                # encode_depth=encode_depth,
                # sound_num=sound_num,
                config=config,
                sound_num=sound_num),
            # action_space.n,
            (config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE) ** 2,
            masking=config.MASKING)
        self.config = config
        self.observation_space = observation_space
        # 网络留到BaselineNet类里面统一管理
        # self.classifier = Sound_Classifier(config.RL.PPO.hidden_size * 2, sound_num)
        # self.regressor = X_Y_Regresor(self.net.rnn_input_size, 2)

    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False, lambda_grad=1.0):
        features, rnn_hidden_states, predicted_labels, predicted_x_y, = self.net(observations,
                                                                                 rnn_hidden_states,
                                                                                 prev_actions, masks,
                                                                                 lambda_grad=lambda_grad)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)
        # set_trace()

        # 表示是否是确定性的，确定性的就直接按概率最大的来，不确定性的还要sample一下
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution, predicted_labels, predicted_x_y

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, lambda_grad=1.0):
        features, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks, lambda_grad=lambda_grad)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action, lambda_grad=1.0):
        features, rnn_hidden_states, predicted_labels, predicted_x_y = self.net(observations, rnn_hidden_states,
                                                                                prev_actions, masks,
                                                                                lambda_grad=lambda_grad)
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        # loss在这里！！
        distribution_entropy = distribution.entropy().mean()

        # # 定义需要返回的变量
        # predicted_labels_detached = None
        # predicted_labels = None
        # predicted_x_y = None

        # # 进行分类的情况
        # if self.net.is_classify:
        #     if self.net.is_visual_classify:
        #         v_a_features = torch.cat([v_features, a_features], dim=1)
        #     else:
        #         v_a_features = a_features

        #     predicted_labels_detached = self.net.classifier(v_a_features.detach())
        #     predicted_labels = self.net.classifier(v_a_features)

        # # 进行回归的情况
        # if self.net.is_regression:
        #     predicted_x_y = self.net.regressor(features)

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, predicted_labels, predicted_x_y
        # 注意：使用返回的predict进行计算的函数，需要检查返回值是否为None！！！


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
        self.net = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(True), nn.Linear(1024, 1024), nn.ReLU(True),
                                 nn.Linear(1024, 512), nn.ReLU(True), nn.Linear(512, output_dim), nn.Softmax())

    def forward(self, x):
        return self.net(x)


# 角度回归 (sin, cos)
class RotRegresor(nn.Module):

    def __init__(self, input_dim, output_dim):  # output_dim = 2为分类数
        super(RotRegresor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(True), nn.Linear(1024, 1024), nn.ReLU(True),
                                 nn.Linear(1024, 512), nn.ReLU(True), nn.Linear(512, output_dim),
                                 nn.Tanh())  # 把输出归一化到(-1, 1)

    def forward(self, x):
        return self.net(x)


class MLPForVisualAndAudio(nn.Module):
    '''
        用在没有av cross attention，但有gm与av的attention的情况下
        来保证a+v的维度与gm维度一致
    '''

    def __init__(self, input_dim):
        super(MLPForVisualAndAudio, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim * 2), nn.ReLU(True),
                                 nn.Linear(input_dim * 2, input_dim), nn.ReLU(True),
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
