#!/usr/bin/env python3
"""
Launch TensorBoard with the latest run's loss log directory.

Usage:
  python tb_last_run.py
  python tb_last_run.py --dry-run
  python tb_last_run.py --port 6007 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _iter_event_files(search_roots: Iterable[Path]):
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            for name in filenames:
                if "tfevents" in name:
                    p = Path(dirpath) / name
                    if p.is_file():
                        yield p


def find_latest_logdir(search_roots: Iterable[Path]) -> Path:
    event_files = list(_iter_event_files(search_roots))
    if not event_files:
        raise FileNotFoundError("No TensorBoard event files found in search roots.")
    latest_file = max(event_files, key=lambda p: p.stat().st_mtime)
    return latest_file.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open TensorBoard for the latest run logs.")
    p.add_argument("--port", type=int, default=6006, help="TensorBoard port (default: 6006)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="TensorBoard host (default: 127.0.0.1)")
    p.add_argument(
        "--search",
        nargs="*",
        default=["logs/tb", "logs", "media", "artifacts"],
        help="Directories to search for tfevents files.",
    )
    p.add_argument("--dry-run", action="store_true", help="Only print detected logdir.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    roots = [(cwd / s).resolve() for s in args.search]

    try:
        logdir = find_latest_logdir(roots)
    except FileNotFoundError as e:
        print(f"[tb_last_run] {e}")
        print(f"[tb_last_run] searched: {', '.join(str(p) for p in roots)}")
        return 1

    print(f"[tb_last_run] latest logdir: {logdir}")
    print(f"[tb_last_run] open: http://{args.host}:{args.port}")

    if args.dry_run:
        return 0

    cmd = [
        "tensorboard",
        "--logdir",
        str(logdir),
        "--port",
        str(args.port),
        "--host",
        args.host,
    ]

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("[tb_last_run] tensorboard command not found.")
        print("[tb_last_run] install with: pip install tensorboard")
        return 1


if __name__ == "__main__":
    sys.exit(main())

