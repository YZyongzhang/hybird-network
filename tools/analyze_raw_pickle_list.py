#!/usr/bin/env python3
"""
Analyze raw pickle episodes from flexible path inputs.

Input supports:
1) --pickle-list <txt/json file>
2) --pickle <file> (repeatable)
3) --paths '["path1","path2"]' or --paths 'path1,path2'
4) directory paths (recursively discover pickle files)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
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


def parse_pickle_list_file(path: Path) -> List[Path]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"JSON list expected: {path}")
        return [Path(str(x)) for x in data]

    files: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        files.append(Path(s))
    return files


def parse_paths_argument(raw: str) -> List[Path]:
    s = raw.strip()
    if not s:
        return []

    # First try JSON list.
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [Path(str(x).strip()) for x in data if str(x).strip()]
    except Exception:
        pass

    # Fallback: support forms like [path1,path2] / 【path1,path2】 / path1,path2
    for ch in "[]【】":
        s = s.replace(ch, "")
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    return [Path(p) for p in parts if p]


def discover_pickles_under_dir(
    root: Path,
    include_no_ext: bool = False,
    limit: int = 0,
) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name == "a.md":
                continue
            p = Path(dirpath) / name
            suffix = p.suffix.lower()
            if suffix in {".pkl", ".pickle"}:
                files.append(p)
            elif include_no_ext and suffix == "":
                files.append(p)
            if limit > 0 and len(files) >= limit:
                files.sort()
                return files[:limit]
    files.sort()
    return files


def expand_paths_to_pickle_files(
    paths: Sequence[Path],
    include_no_ext: bool = False,
    limit: int = 0,
) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.exists() and p.is_file():
            suffix = p.suffix.lower()
            if suffix in {".pkl", ".pickle"} or (include_no_ext and suffix == ""):
                out.append(p)
            if limit > 0 and len(out) >= limit:
                break
            continue
        if p.exists() and p.is_dir():
            room = max(0, limit - len(out)) if limit > 0 else 0
            out.extend(
                discover_pickles_under_dir(
                    p,
                    include_no_ext=include_no_ext,
                    limit=room,
                )
            )
            if limit > 0 and len(out) >= limit:
                break
            continue
        # keep unresolved path in list for later skipped report
        out.append(p)
        if limit > 0 and len(out) >= limit:
            break
    # de-dup while preserving order
    uniq: List[Path] = []
    seen = set()
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        return bool(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def extract_distance_to_goal(info_item: Any) -> Optional[float]:
    if not isinstance(info_item, dict):
        return None
    for key in ("distance_to_goal", "geodesic_distance", "euclidian_distance"):
        if key not in info_item:
            continue
        value = info_item[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                continue
            value = value[0]
        dist = safe_float(value)
        if dist is not None:
            return dist
    return None


def extract_success(info_item: Any) -> Optional[bool]:
    if not isinstance(info_item, dict):
        return None
    if "success" not in info_item:
        return None
    value = info_item["success"]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    return safe_bool(value, default=False)


def infer_collision_mask(
    data: Dict[str, Any], rewards: Sequence[float], info: Sequence[Any], threshold: float
) -> List[bool]:
    n = min(len(rewards), len(info))
    raw = data.get("collision", None)
    if raw is not None:
        vals = flatten_1d_sequence(raw)
        mask = [safe_bool(v, default=False) for v in vals[:n]]
        if len(mask) < n:
            mask.extend([False] * (n - len(mask)))
        return mask

    mask: List[bool] = []
    for i in range(n):
        collided: Optional[bool] = None
        item = info[i]
        if isinstance(item, dict):
            for key in ("is_collided", "collision", "collided", "collode"):
                if key in item:
                    collided = safe_bool(item.get(key), default=False)
                    break
        if collided is None:
            collided = float(rewards[i]) <= threshold
        mask.append(collided)
    return mask


def build_hist(values: Sequence[float], bins: int) -> Dict[str, List[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(arr, bins=max(1, int(bins)))
    return {
        "edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }


def save_analysis_figures(
    report: Dict[str, Any],
    rewards: Sequence[float],
    initial_distances: Sequence[float],
    early_stop_terminal_distances: Sequence[float],
    out_dir: Path,
    distance_hist_bins: int,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = report["summary"]
    total_episodes = max(1, int(summary["total_episodes"]))
    total_steps = max(1, int(summary["total_steps"]))

    success = int(summary["success_episode_count"])
    collision = int(summary["collision_episode_count"])
    early_stop = int(summary["early_stop_episode_count"])
    collision_steps = int(summary["collision_step_count"])
    non_collision_steps = max(0, total_steps - collision_steps)

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    init_dist_arr = np.asarray(initial_distances, dtype=np.float64)
    early_dist_arr = np.asarray(early_stop_terminal_distances, dtype=np.float64)

    saved: Dict[str, str] = {}

    # 1) Episode-level ratio bars
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=140)
    ratio_names = ["success", "collision", "early_stop"]
    ratio_values = [
        success / total_episodes,
        collision / total_episodes,
        early_stop / total_episodes,
    ]
    bars = ax1.bar(ratio_names, ratio_values, color=["#2ca02c", "#d62728", "#ff7f0e"])
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("ratio")
    ax1.set_title("Episode Ratios")
    for b, v in zip(bars, ratio_values):
        ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2%}", ha="center", va="bottom", fontsize=10)
    fig1.tight_layout()
    p1 = out_dir / "episode_ratios_bar.png"
    fig1.savefig(p1)
    plt.close(fig1)
    saved["episode_ratios_bar"] = str(p1)

    # 2) Collision step ratio pie
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=140)
    ax2.pie(
        [collision_steps, non_collision_steps],
        labels=["collision_steps", "non_collision_steps"],
        autopct="%.2f%%",
        startangle=90,
        counterclock=False,
    )
    ax2.set_title("Collision Step Ratio")
    ax2.axis("equal")
    fig2.tight_layout()
    p2 = out_dir / "collision_step_ratio_pie.png"
    fig2.savefig(p2)
    plt.close(fig2)
    saved["collision_step_ratio_pie"] = str(p2)

    # 3) Reward histogram
    fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=140)
    if rewards_arr.size > 0:
        ax3.hist(rewards_arr, bins=40, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax3.set_title("Reward Distribution")
    ax3.set_xlabel("reward")
    ax3.set_ylabel("count")
    fig3.tight_layout()
    p3 = out_dir / "reward_hist.png"
    fig3.savefig(p3)
    plt.close(fig3)
    saved["reward_hist"] = str(p3)

    # 4) Initial distance histogram
    fig4, ax4 = plt.subplots(figsize=(8, 5), dpi=140)
    if init_dist_arr.size > 0:
        ax4.hist(
            init_dist_arr,
            bins=max(1, int(distance_hist_bins)),
            color="#9467bd",
            alpha=0.85,
            edgecolor="white",
        )
    ax4.set_title("Initial Distance-To-Goal Distribution")
    ax4.set_xlabel("distance_to_goal (initial)")
    ax4.set_ylabel("count")
    fig4.tight_layout()
    p4 = out_dir / "initial_distance_hist.png"
    fig4.savefig(p4)
    plt.close(fig4)
    saved["initial_distance_hist"] = str(p4)

    # 5) Early-stop terminal distance box
    fig5, ax5 = plt.subplots(figsize=(6, 5), dpi=140)
    if early_dist_arr.size > 0:
        ax5.boxplot(early_dist_arr, vert=True, labels=["early_stop_terminal_distance"])
        ax5.set_ylabel("distance_to_goal")
    else:
        ax5.text(0.5, 0.5, "No early-stop terminal distance data", ha="center", va="center")
        ax5.set_xticks([])
        ax5.set_yticks([])
    ax5.set_title("Early-Stop Terminal Distance")
    fig5.tight_layout()
    p5 = out_dir / "early_stop_terminal_distance_box.png"
    fig5.savefig(p5)
    plt.close(fig5)
    saved["early_stop_terminal_distance_box"] = str(p5)

    # 6) One-page overview (2x2) for quick sharing
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=140)
    ax = axes[0, 0]
    bars = ax.bar(ratio_names, ratio_values, color=["#2ca02c", "#d62728", "#ff7f0e"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Episode Ratios")
    for b, v in zip(bars, ratio_values):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2%}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    ax.pie(
        [collision_steps, non_collision_steps],
        labels=["collision", "non-collision"],
        autopct="%.2f%%",
        startangle=90,
        counterclock=False,
    )
    ax.set_title("Collision Step Ratio")
    ax.axis("equal")

    ax = axes[1, 0]
    if rewards_arr.size > 0:
        ax.hist(rewards_arr, bins=40, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_title("Reward Distribution")
    ax.set_xlabel("reward")
    ax.set_ylabel("count")

    ax = axes[1, 1]
    if init_dist_arr.size > 0:
        ax.hist(
            init_dist_arr,
            bins=max(1, int(distance_hist_bins)),
            color="#9467bd",
            alpha=0.85,
            edgecolor="white",
        )
    ax.set_title("Initial Distance Distribution")
    ax.set_xlabel("distance_to_goal (initial)")
    ax.set_ylabel("count")

    fig.tight_layout()
    p6 = out_dir / "analysis_overview.png"
    fig.savefig(p6)
    plt.close(fig)
    saved["analysis_overview"] = str(p6)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw pickle files from a provided list.")
    parser.add_argument("--pickle-list", type=Path, default=None, help="txt/json file containing pickle paths")
    parser.add_argument("--pickle", action="append", default=[], help="single pickle path, repeatable")
    parser.add_argument(
        "--paths",
        type=str,
        default="",
        help='paths list string, e.g. \'["path1","path2"]\' or "path1,path2"',
    )
    parser.add_argument("--include-no-ext", action="store_true", help="also include files without extension")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all discovered files")
    parser.add_argument("--stop-action-id", type=int, default=0, help="stop action id")
    parser.add_argument(
        "--success-distance-threshold",
        type=float,
        default=0,
        help="distance_to_goal <= threshold considered successful stop",
    )
    parser.add_argument(
        "--success-reward-threshold",
        type=float,
        default=0.0,
        help="fallback success threshold on terminal reward when distance/success is unavailable",
    )
    parser.add_argument(
        "--collision-reward-threshold",
        type=float,
        default=-4.0,
        help="fallback collision criterion: reward <= threshold",
    )
    parser.add_argument("--distance-hist-bins", type=int, default=20, help="bins for initial distance histogram")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output json path, default: tmp/raw_pickle_stats_<timestamp>.json",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Output figure directory, default: tmp/raw_pickle_figs_<timestamp>",
    )
    args = parser.parse_args()

    input_paths: List[Path] = [Path(p) for p in args.pickle]
    if args.pickle_list is not None:
        input_paths.extend(parse_pickle_list_file(args.pickle_list))
    if args.paths:
        input_paths.extend(parse_paths_argument(args.paths))
    if not input_paths:
        raise ValueError("No paths provided. Use --paths / --pickle-list / --pickle.")

    pickle_files = expand_paths_to_pickle_files(
        input_paths,
        include_no_ext=bool(args.include_no_ext),
        limit=int(args.max_files),
    )
    if not pickle_files:
        raise RuntimeError("No pickle files discovered from given paths.")

    all_rewards: List[float] = []
    initial_distances: List[float] = []
    early_stop_terminal_distances: List[float] = []
    early_stop_terminal_rewards: List[float] = []
    skipped_files: List[str] = []

    total_episodes = 0
    success_episodes = 0
    collision_episodes = 0
    early_stop_episodes = 0
    total_steps = 0
    total_collision_steps = 0
    from tqdm import tqdm
    for fp in tqdm(pickle_files):
        if not fp.exists() or not fp.is_file():
            skipped_files.append(f"missing:{fp}")
            continue

        try:
            data = load_pickle(fp)
        except Exception as exc:
            skipped_files.append(f"load_fail:{fp}:{exc}")
            continue

        rewards = [safe_float(v) for v in flatten_1d_sequence(data.get("reward", []))]
        rewards = [float(v) for v in rewards if v is not None]
        actions = [safe_int(v, default=-1) for v in flatten_1d_sequence(data.get("action_id", []))]
        dones = [safe_bool(v, default=False) for v in flatten_1d_sequence(data.get("done", []))]

        info = data.get("info", [])
        if isinstance(info, np.ndarray):
            info = info.reshape(-1).tolist()
        elif not isinstance(info, list):
            info = flatten_1d_sequence(info)

        n = min(len(rewards), len(actions), len(dones), len(info))
        if n <= 0:
            skipped_files.append(f"invalid_empty:{fp}")
            continue

        rewards = rewards[:n]
        actions = actions[:n]
        dones = dones[:n]
        info = info[:n]

        total_episodes += 1
        total_steps += n
        all_rewards.extend(rewards)

        collision_mask = infer_collision_mask(
            data=data,
            rewards=rewards,
            info=info,
            threshold=float(args.collision_reward_threshold),
        )[:n]
        c_steps = int(sum(1 for c in collision_mask if c))
        total_collision_steps += c_steps
        if c_steps > 0:
            collision_episodes += 1

        init_dist = extract_distance_to_goal(info[0])
        if init_dist is not None:
            initial_distances.append(float(init_dist))

        success_by_info = False
        for item in info:
            s = extract_success(item)
            if s is True:
                success_by_info = True
                break

        terminal_idx = n - 1
        terminal_action = int(actions[terminal_idx])
        terminal_reward = float(rewards[terminal_idx])
        terminal_done = bool(dones[terminal_idx])
        terminal_dist = extract_distance_to_goal(info[terminal_idx])

        success_by_stop = False
        if terminal_done and terminal_action == int(args.stop_action_id):
            if terminal_dist is not None:
                success_by_stop = terminal_dist <= float(args.success_distance_threshold)
            else:
                success_by_stop = terminal_reward >= float(args.success_reward_threshold)

        success = success_by_info or success_by_stop
        if success:
            success_episodes += 1

        early_stop = terminal_done and (terminal_action == int(args.stop_action_id)) and (not success)
        if early_stop:
            early_stop_episodes += 1
            early_stop_terminal_rewards.append(terminal_reward)
            if terminal_dist is not None:
                early_stop_terminal_distances.append(float(terminal_dist))

    if total_episodes <= 0:
        raise RuntimeError("No valid pickle episodes parsed.")

    rewards_arr = np.asarray(all_rewards, dtype=np.float64) if all_rewards else np.asarray([], dtype=np.float64)

    early_reward_range = {
        "min": float(np.min(early_stop_terminal_rewards)) if early_stop_terminal_rewards else None,
        "max": float(np.max(early_stop_terminal_rewards)) if early_stop_terminal_rewards else None,
        "mean": float(np.mean(early_stop_terminal_rewards)) if early_stop_terminal_rewards else None,
    }
    early_distance_range = {
        "min": float(np.min(early_stop_terminal_distances)) if early_stop_terminal_distances else None,
        "max": float(np.max(early_stop_terminal_distances)) if early_stop_terminal_distances else None,
        "mean": float(np.mean(early_stop_terminal_distances)) if early_stop_terminal_distances else None,
    }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_json or Path("tmp") / f"raw_pickle_stats_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "input": {
            "pickle_list": str(args.pickle_list) if args.pickle_list else None,
            "paths_arg": args.paths,
            "num_input_paths": len(input_paths),
            "num_discovered_files": len(pickle_files),
            "max_files": int(args.max_files),
            "thresholds": {
                "stop_action_id": int(args.stop_action_id),
                "success_distance_threshold": float(args.success_distance_threshold),
                "success_reward_threshold": float(args.success_reward_threshold),
                "collision_reward_threshold": float(args.collision_reward_threshold),
            },
        },
        "summary": {
            "total_episodes": int(total_episodes),
            "total_steps": int(total_steps),
            "success_episode_count": int(success_episodes),
            "success_episode_ratio": float(success_episodes / total_episodes),
            "collision_episode_count": int(collision_episodes),
            "collision_episode_ratio": float(collision_episodes / total_episodes),
            "early_stop_episode_count": int(early_stop_episodes),
            "early_stop_episode_ratio": float(early_stop_episodes / total_episodes),
            "collision_step_count": int(total_collision_steps),
            "collision_step_ratio": float(total_collision_steps / max(1, total_steps)),
        },
        "reward_range": {
            "min": float(np.min(rewards_arr)) if rewards_arr.size > 0 else None,
            "max": float(np.max(rewards_arr)) if rewards_arr.size > 0 else None,
            "mean": float(np.mean(rewards_arr)) if rewards_arr.size > 0 else None,
            "p01": float(np.percentile(rewards_arr, 1)) if rewards_arr.size > 0 else None,
            "p50": float(np.percentile(rewards_arr, 50)) if rewards_arr.size > 0 else None,
            "p95": float(np.percentile(rewards_arr, 95)) if rewards_arr.size > 0 else None,
            "p99": float(np.percentile(rewards_arr, 99)) if rewards_arr.size > 0 else None,
        },
        "early_stop_range": {
            "terminal_reward": early_reward_range,
            "terminal_distance_to_goal": early_distance_range,
        },
        "initial_distance_distribution": {
            "count": int(len(initial_distances)),
            "summary": {
                "min": float(np.min(initial_distances)) if initial_distances else None,
                "max": float(np.max(initial_distances)) if initial_distances else None,
                "mean": float(np.mean(initial_distances)) if initial_distances else None,
                "p50": float(np.percentile(initial_distances, 50)) if initial_distances else None,
                "p95": float(np.percentile(initial_distances, 95)) if initial_distances else None,
            },
            "histogram": build_hist(initial_distances, bins=int(args.distance_hist_bins)),
        },
        "skipped_files": skipped_files,
    }

    fig_dir = args.fig_dir or Path("tmp") / f"raw_pickle_figs_{ts}"
    figure_paths = save_analysis_figures(
        report=report,
        rewards=all_rewards,
        initial_distances=initial_distances,
        early_stop_terminal_distances=early_stop_terminal_distances,
        out_dir=fig_dir,
        distance_hist_bins=int(args.distance_hist_bins),
    )
    report["figures"] = figure_paths

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"episodes={total_episodes} steps={total_steps}")
    print(
        f"success={success_episodes}/{total_episodes} "
        f"collision={collision_episodes}/{total_episodes} "
        f"early_stop={early_stop_episodes}/{total_episodes}"
    )
    print(
        f"collision_steps={total_collision_steps}/{total_steps} "
        f"ratio={total_collision_steps / max(1, total_steps):.6f}"
    )
    print(f"report_json={out_path}")
    print(f"fig_dir={fig_dir}")
    print(f"overview_png={figure_paths.get('analysis_overview', '')}")


if __name__ == "__main__":
    main()
    # python tools/analyze_raw_pickle_list.py --paths '["dataset-data3/dataset/pickle/offline(1.5_muti)/level3","dataset-data3/dataset/pickle/offline(2.0_muti)/level3","dataset-data3/dataset/pickle/offline(muti_collided)","dataset-data3/dataset/pickle/hybird(heard_train_multiple)"]'
