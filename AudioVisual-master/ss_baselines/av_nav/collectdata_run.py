#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from ipdb import set_trace

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import torch

import soundspaces
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_nav.config.default import get_config
from ss_baselines.av_wan.run import find_best_ckpt_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        # required=True,
        default='av_nav/config/pointgoal_rgb.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    ''' init a config '''
    config = get_config(args.exp_config, args.opts, args.model_dir, 'train', True)
    # config.keys()
    # ['SEED', 'BASE_TASK_CONFIG_PATH', 'TASK_CONFIG', 'CMD_TRAILING_OPTS', 'TRAINER_NAME', 'ENV_NAME',
    #  'SIMULATOR_GPU_ID', 'TORCH_GPU_ID', 'VIDEO_OPTION', 'VISUALIZATION_OPTION', 'TENSORBOARD_DIR', 'VIDEO_DIR',
    #  'TEST_EPISODE_COUNT', 'EVAL_CKPT_PATH_DIR', 'NUM_PROCESSES', 'SENSORS', 'CHECKPOINT_FOLDER', 'NUM_UPDATES',
    #  'LOG_INTERVAL', 'LOG_FILE', 'CHECKPOINT_INTERVAL', 'USE_VECENV', 'USE_SYNC_VECENV', 'EXTRA_RGB', 'DEBUG',
    #  'USE_LAST_CKPT', 'DISPLAY_RESOLUTION', 'EVAL', 'RL']
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    # Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    ''' start trainer's main func'''
    trainer.train()


if __name__ == "__main__":
    main()
