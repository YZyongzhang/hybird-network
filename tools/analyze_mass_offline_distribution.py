#!/usr/bin/env python3
"""
Only analyze PT reward ratio distribution and draw a pie chart.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def discover_shards(root: Path) -> List[Path]:
    files = sorted(root.glob("offline_rl_shard_*.pt"))
    if files:
        return files
    return sorted(root.glob("*.pt"))


def to_1d_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).numpy()
    return np.asarray(value).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PT reward ratio and draw pie chart.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing offline_rl_shard_*.pt")
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all shards")
    parser.add_argument("--round-decimals", type=int, default=2, help="Round reward values before counting")
    parser.add_argument("--top-k", type=int, default=8, help="Keep top-k reward buckets; merge others as 'other'")
    parser.add_argument("--title", type=str, default="PT Reward Ratio Distribution", help="Pie chart title")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output report path, default: tmp/pt_reward_ratio_<timestamp>.json",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output pie figure path, default: tmp/pt_reward_ratio_<timestamp>.png",
    )
    args = parser.parse_args()

    shard_files = discover_shards(args.root)
    if args.max_shards > 0:
        shard_files = shard_files[: args.max_shards]
    if not shard_files:
        raise RuntimeError(f"No PT shards found under: {args.root}")

    counter: Counter[str] = Counter()
    total_steps = 0

    for shard in shard_files:
        data = torch.load(shard, map_location="cpu")
        if "rewards" not in data:
            raise RuntimeError(f"Missing key 'rewards' in shard: {shard}")

        rewards = to_1d_numpy(data["rewards"]).astype(np.float64)
        if rewards.size == 0:
            continue
        total_steps += int(rewards.size)

        rounded = np.round(rewards, decimals=int(args.round_decimals))
        for v in rounded.tolist():
            counter[f"{float(v):.{int(args.round_decimals)}f}"] += 1

    if total_steps <= 0:
        raise RuntimeError("No valid reward steps parsed.")

    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    top_k = max(1, int(args.top_k))
    if len(items) > top_k:
        keep = items[:top_k]
        other_cnt = int(sum(c for _, c in items[top_k:]))
        if other_cnt > 0:
            keep.append(("other", other_cnt))
        items = keep

    labels = [k for k, _ in items]
    counts = np.asarray([int(c) for _, c in items], dtype=np.int64)
    ratios = counts.astype(np.float64) / float(total_steps)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or Path("tmp") / f"pt_reward_ratio_{ts}.json"
    out_png = args.out_png or Path("tmp") / f"pt_reward_ratio_{ts}.png"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {
        "root": str(args.root),
        "num_shards": int(len(shard_files)),
        "total_steps": int(total_steps),
        "round_decimals": int(args.round_decimals),
        "top_k": int(args.top_k),
        "reward_ratio_distribution": [
            {
                "reward": label,
                "count": int(cnt),
                "ratio": float(cnt / total_steps),
            }
            for label, cnt in items
        ],
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.pie(
        counts,
        labels=labels,
        autopct="%.2f%%",
        startangle=90,
        counterclock=False,
    )
    ax.set_title(args.title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    print(f"shards={len(shard_files)} total_steps={total_steps}")
    for label, cnt, ratio in zip(labels, counts.tolist(), ratios.tolist()):
        print(f"reward={label} count={cnt} ratio={ratio:.6f}")
    print(f"report_json={out_json}")
    print(f"pie_png={out_png}")


if __name__ == "__main__":
    main()
