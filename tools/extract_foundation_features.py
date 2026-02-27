#!/usr/bin/env python3
"""
Extract features with foundation model and save offline RL shards as .pt files.

Example:
python tools/extract_foundation_features.py \
  --raw-data-path /path/to/raw_data \
  --ckpt /path/to/foundation_model.pth \
  --to-path /path/to/output_pt \
  --mode offlinetwoframe
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use foundation model to extract features and save to PT shards."
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        required=True,
        help="Input raw data directory (same as VADE LoadLmdb path).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Foundation model checkpoint path (.pth).",
    )
    parser.add_argument(
        "--to-path",
        type=str,
        required=True,
        help="Output directory for .pt shards.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offlinetwoframe",
        choices=["offline", "offlinetwoframe"],
        help="Feature extraction mode from VADE code.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from network import HybirdNetwork
    from train.VADE import LoadLmdb

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.isdir(args.raw_data_path):
        raise NotADirectoryError(f"Raw data directory not found: {args.raw_data_path}")

    os.makedirs(args.to_path, exist_ok=True)
    config = SimpleNamespace(TO_PATH=args.to_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybirdNetwork().to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.device = device

    print(f"[info] device={device}")
    print(f"[info] mode={args.mode}")
    print(f"[info] input={args.raw_data_path}")
    print(f"[info] output={args.to_path}")

    if args.mode == "offline":
        LoadLmdb.load_offline(args.raw_data_path, model=model, config=config)
    elif args.mode == "offlinetwoframe":
        LoadLmdb.load_offline_two_frame(args.raw_data_path, model=model, config=config)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print("[done] feature extraction finished.")


if __name__ == "__main__":
    main()
