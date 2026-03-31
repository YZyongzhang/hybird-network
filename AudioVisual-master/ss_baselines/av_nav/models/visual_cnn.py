# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from ipdb import set_trace
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from ss_baselines.common.utils import Flatten


def conv_output_dim(dimension, padding, dilation, kernel_size, stride):
    r"""Calculates the output height and width based on the input
    height and width to the convolution layer.

    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    assert len(dimension) == 2
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(int(np.floor(((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1)))
    return tuple(out_dimension)


def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)


def add_normalize(to_norm_module, normalize_config, input_shape=None):
    if normalize_config == "none":
        return to_norm_module
    elif normalize_config == "batchnorm":
        if isinstance(to_norm_module, nn.Conv2d):
            return nn.Sequential(to_norm_module, nn.BatchNorm2d(to_norm_module.out_channels))
        elif isinstance(to_norm_module, nn.Conv1d):
            return nn.Sequential(to_norm_module, nn.BatchNorm1d(to_norm_module.out_channels))
        elif isinstance(to_norm_module, nn.Linear):
            return nn.Sequential(to_norm_module, nn.BatchNorm1d(to_norm_module.out_features))
        else:
            raise NotImplementedError
    elif normalize_config == "layernorm":
        return nn.Sequential(to_norm_module, nn.LayerNorm(input_shape))
    elif normalize_config == "spectralnorm":
        return spectral_norm(to_norm_module)
    else:
        raise NotImplementedError


class VisualCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    def __init__(self, observation_space, output_size, extra_rgb):
        # output_size = 512
        # extra_rgb = False
        super().__init__()
        # 下面的n就是channel数
        if "rgb" in observation_space.spaces and not extra_rgb:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        self._n_input_depth = 0
        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0
        # self.is_blind = False
        # self.is_blind = ((self._n_input_rgb == 0) and (self._n_input_depth == 0))

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]

        if self._n_input_rgb > 0:
            # observation_space.spaces["rgb"].shape[:2] = (128, 128)
            cnn_dims = np.array(observation_space.spaces["rgb"].shape[:2], dtype=np.float32)
        elif self._n_input_depth > 0:
            cnn_dims = np.array(observation_space.spaces["depth"].shape[:2], dtype=np.float32)

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(self._cnn_layers_kernel_size, self._cnn_layers_stride):
                cnn_dims = conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )
        layer_init(self.cnn)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)


class WhcVisualCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    def __init__(self, observation_space, output_size, extra_rgb, config=None):
        # output_size = 512
        # extra_rgb = False
        super().__init__()
        # 下面的n就是channel数
        if "rgb" in observation_space.spaces and not extra_rgb:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        self._n_input_depth = 0
        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0
        # self.is_blind = ((self._n_input_rgb == 0) and (self._n_input_depth == 0))

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]

        if self._n_input_rgb > 0:
            # observation_space.spaces["rgb"].shape[:2] = (128, 128)
            cnn_dims = np.array(observation_space.spaces["rgb"].shape[:2], dtype=np.float32)
        elif self._n_input_depth > 0:
            cnn_dims = np.array(observation_space.spaces["depth"].shape[:2], dtype=np.float32)

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(self._cnn_layers_kernel_size, self._cnn_layers_stride):
                cnn_dims = conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                add_normalize(nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ), config.normalize_config),
                nn.ReLU(True),
                add_normalize(nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ), config.normalize_config),
                nn.ReLU(True),
                add_normalize(nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ), config.normalize_config),
                #  nn.ReLU(True),
                nn.ReLU(True),
                Flatten(),
                add_normalize(nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size), config.normalize_config),
                # nn.ReLU(True),
            )
        layer_init(self.cnn)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)


class NaiveVisualCNN(nn.Module):
    def __init__(self, ch):
        super(NaiveVisualCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, ch, 5, 1, 2),
            # nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 3, 1, 1),
            # nn.Conv2d(ch, ch * 2, 4, 2, 1),
            # nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            # nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.Conv2d(ch * 2, ch * 4, 3, 1, 1),
            # nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
        )

    def forward(self, visual_inputs):
        h = self.cnn(visual_inputs)
        return h


# class NaiveVisualCNNDownSample(nn.Module):
#     def __init__(self, ch):
#         super(NaiveVisualCNN, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, ch, 5, 1, 2),
#             # nn.BatchNorm2d(ch),
#             nn.ReLU(True),
#             nn.Conv2d(ch, ch * 2, 3, 1, 1),
#             # nn.Conv2d(ch, ch * 2, 4, 2, 1),
#             # nn.BatchNorm2d(ch * 2),
#             nn.ReLU(True),
#             # nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
#             nn.Conv2d(ch * 2, ch * 4, 3, 1, 1),
#             # nn.BatchNorm2d(ch * 4),
#             nn.ReLU(True),
#         )
#
#     def forward(self, visual_inputs):
#         h = self.cnn(visual_inputs)
#         return h


class VisualCNNAffordance(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    def __init__(self, ch=32):
        super().__init__()
        self._n_input_rgb = 3
        self._n_input_depth = 0

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array((128, 128), dtype=np.float32)
        elif self._n_input_depth > 0:
            cnn_dims = np.array((128, 128), dtype=np.float32)

        for kernel_size, stride in zip(self._cnn_layers_kernel_size, self._cnn_layers_stride):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_rgb + self._n_input_depth,
                out_channels=ch,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=ch,
                out_channels=ch * 2,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=ch * 2,
                out_channels=ch * 2,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            # 卷三层就够，不用再linear
            # #  nn.ReLU(True),
            # Flatten(),
            # nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        layer_init(self.cnn)

    def forward(self, rgb_observations):
        # 预处理在数据集就做好
        return self.cnn(rgb_observations)
