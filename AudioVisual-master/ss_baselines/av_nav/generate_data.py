import argparse
import logging
import os

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
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument("--overwrite", default=False, action='store_true', help="Modify config options from command line")
    parser.add_argument("--eval-best", default=False, action='store_true', help="Modify config options from command line")
    parser.add_argument("--moving-source", default=False, action='store_true', help="Modify config options from command line")
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--run-type",
        default='data_generation',
        help="run type of the experiment (train or eval)",
    )
    args = parser.parse_args()

    # run exp
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    trainer.data_generation()


if __name__ == "__main__":
    main()
