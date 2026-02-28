#!/usr/bin/env python3
"""
Statistics for offline collided dataset.

Metrics:
1) average steps
2) collision statistics
3) recovery-from-collision statistics
4) navigation success distribution:
   success = distance_to_goal <= success_eps AND action==0 AND done=True
"""

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

try:
    from tools.visualize_collided_topdown import load_pickle
except Exception:
    from visualize_collided_topdown import load_pickle


def flatten_1d_sequence(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                out.extend(flatten_1d_sequence(item))
            else:
                out.append(item)
        return out
    return [value]


def discover_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == "a.md":
            continue
        if p.suffix.lower() in {".pkl", ".pickle", ""}:
            files.append(p)
    files.sort()
    return files


def get_collision_mask(data: dict, rewards: Sequence[float], threshold: float) -> List[bool]:
    raw = data.get("collision", None)
    if raw is not None:
        vals = flatten_1d_sequence(raw)
        return [bool(v) for v in vals[: len(rewards)]]
    return [float(r) < threshold for r in rewards]


def run_lengths(mask: Sequence[bool]) -> List[int]:
    out: List[int] = []
    c = 0
    for v in mask:
        if v:
            c += 1
        elif c > 0:
            out.append(c)
            c = 0
    if c > 0:
        out.append(c)
    return out


def safe_dist(info_item: Any) -> Optional[float]:
    if not isinstance(info_item, dict):
        return None
    d = info_item.get("distance_to_goal", None)
    if d is None:
        return None
    try:
        return float(d)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Statistics for offline collided dataset.")
    p.add_argument("--root", type=Path, required=True, help="Root dir of offline collided pickle dataset.")
    p.add_argument("--max-files", type=int, default=0, help="0 means all files.")
    p.add_argument(
        "--collision-threshold",
        type=float,
        default=0.0,
        help="Used only when 'collision' field does not exist: reward<threshold => collision.",
    )
    p.add_argument(
        "--success-eps",
        type=float,
        default=0.0,
        help="distance_to_goal <= success_eps is treated as goal reached.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output json path. Default: tmp/offline_collided_stats_<timestamp>.json",
    )
    args = p.parse_args()

    files = discover_files(args.root)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No files found under {args.root}")

    steps_list: List[int] = []
    collision_steps_list: List[int] = []
    collision_ratio_list: List[float] = []
    longest_collision_run_list: List[int] = []
    total_collision_steps = 0
    total_steps = 0
    episodes_with_collision = 0
    recovered_episodes = 0
    stuck_episodes = 0
    all_rewards: List[float] = []
    all_collision_runs: List[int] = []
    status_counter: Counter = Counter()

    for fp in files:
        data = load_pickle(fp)
        rewards = [float(v) for v in flatten_1d_sequence(data.get("reward", []))]
        actions = flatten_1d_sequence(data.get("action_id", []))
        dones = [bool(v) for v in flatten_1d_sequence(data.get("done", []))]
        info = data.get("info", [])
        if not isinstance(info, list):
            info = []

        n = min(len(rewards), len(actions), len(dones), len(info))
        if n <= 0:
            continue
        rewards = rewards[:n]
        actions = actions[:n]
        dones = dones[:n]
        info = info[:n]
        collision_mask = get_collision_mask(data, rewards, threshold=args.collision_threshold)[:n]

        steps = n
        c_steps = int(sum(collision_mask))
        runs = run_lengths(collision_mask)
        longest_run = max(runs) if runs else 0
        has_collision = c_steps > 0

        steps_list.append(steps)
        collision_steps_list.append(c_steps)
        collision_ratio_list.append(c_steps / steps)
        longest_collision_run_list.append(longest_run)
        all_collision_runs.extend(runs)
        all_rewards.extend(rewards)
        total_steps += steps
        total_collision_steps += c_steps
        if has_collision:
            episodes_with_collision += 1

        # Recovery definition:
        # after the LAST collision, there exists a later non-collision step.
        last_col = -1
        for i, c in enumerate(collision_mask):
            if c:
                last_col = i
        recovered = bool(last_col >= 0 and last_col + 1 < n and any(not x for x in collision_mask[last_col + 1 :]))
        if recovered:
            recovered_episodes += 1
        if has_collision and last_col == n - 1:
            stuck_episodes += 1

        # Success distributions.
        distances = [safe_dist(x) for x in info]
        reached_goal = any((d is not None) and (d <= args.success_eps) for d in distances)
        stopped = any(int(a) == 0 for a in actions)
        success_stop = False
        for i in range(n):
            d = distances[i]
            if d is None:
                continue
            if d <= args.success_eps and int(actions[i]) == 0 and bool(dones[i]):
                success_stop = True
                break

        if success_stop:
            status_counter["success_stop_at_goal"] += 1
        elif reached_goal and not stopped:
            status_counter["reached_goal_not_stopped"] += 1
        elif stopped and not reached_goal:
            status_counter["stopped_not_goal"] += 1
        elif reached_goal and stopped:
            status_counter["reached_goal_and_stopped_but_not_done"] += 1
        else:
            status_counter["failed"] += 1

    if not steps_list:
        raise RuntimeError("No valid episodes parsed.")

    rewards_arr = np.array(all_rewards, dtype=np.float32)
    out = {
        "root": str(args.root),
        "episodes": len(steps_list),
        "steps": {
            "avg": float(np.mean(steps_list)),
            "median": float(np.median(steps_list)),
            "p90": float(np.percentile(steps_list, 90)),
            "p95": float(np.percentile(steps_list, 95)),
            "min": int(np.min(steps_list)),
            "max": int(np.max(steps_list)),
        },
        "collision": {
            "episodes_with_collision": int(episodes_with_collision),
            "episodes_with_collision_ratio": float(episodes_with_collision / len(steps_list)),
            "total_collision_steps": int(total_collision_steps),
            "collision_step_ratio": float(total_collision_steps / total_steps),
            "avg_collision_steps_per_episode": float(np.mean(collision_steps_list)),
            "avg_collision_ratio_per_episode": float(np.mean(collision_ratio_list)),
            "longest_collision_run": {
                "avg": float(np.mean(longest_collision_run_list)),
                "p95": float(np.percentile(longest_collision_run_list, 95)),
                "max": int(np.max(longest_collision_run_list)),
            },
            "collision_run_distribution": {
                "num_runs": int(len(all_collision_runs)),
                "mean": float(np.mean(all_collision_runs)) if all_collision_runs else 0.0,
                "p95": float(np.percentile(all_collision_runs, 95)) if all_collision_runs else 0.0,
                "max": int(np.max(all_collision_runs)) if all_collision_runs else 0,
            },
        },
        "recovery": {
            "recovered_after_collision": int(recovered_episodes),
            "recovered_after_collision_ratio": float(recovered_episodes / len(steps_list)),
            "stuck_at_end_after_collision": int(stuck_episodes),
            "stuck_at_end_after_collision_ratio": float(stuck_episodes / len(steps_list)),
        },
        "navigation_success_distribution": {
            "success_eps": float(args.success_eps),
            "counts": dict(status_counter),
            "ratios": {k: float(v / len(steps_list)) for k, v in status_counter.items()},
        },
        "reward_summary": {
            "min": float(np.min(rewards_arr)),
            "max": float(np.max(rewards_arr)),
            "mean": float(np.mean(rewards_arr)),
            "std": float(np.std(rewards_arr)),
        },
    }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_json or Path("tmp") / f"offline_collided_stats_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"files={len(steps_list)} root={args.root}")
    print(f"avg_steps={out['steps']['avg']:.3f} collision_step_ratio={out['collision']['collision_step_ratio']:.4f}")
    print(
        f"recovered={out['recovery']['recovered_after_collision']}/{len(steps_list)} "
        f"stuck={out['recovery']['stuck_at_end_after_collision']}/{len(steps_list)}"
    )
    print("navigation_success_distribution:", out["navigation_success_distribution"]["counts"])
    print(f"report_json={out_path}")


if __name__ == "__main__":
    main()
# python tools/stat_offline_collided.py --root "dataset-data3/dataset/pickle/offline(muti_collided)" --success-eps 0.0
