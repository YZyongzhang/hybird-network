#!/usr/bin/env python3
"""
Analyze collided offline pickle dataset.

Focus:
1) collision duration statistics (consecutive collision run length)
2) reward design statistics (global + conditioned on collision/non-collision)
3) whether episodes recover from collision

Also generates a topdown visualization grid by reusing visualize_collided_topdown.
"""

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    # when executed as "python -m tools.analyze_collided_data"
    from tools.visualize_collided_topdown import load_pickle, render_episode_image
except Exception:
    # when executed as "python tools/analyze_collided_data.py"
    from visualize_collided_topdown import load_pickle, render_episode_image


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


def discover_pickles(root: Path) -> List[Path]:
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


def run_lengths(mask: Sequence[bool]) -> List[int]:
    runs: List[int] = []
    cnt = 0
    for v in mask:
        if v:
            cnt += 1
        elif cnt > 0:
            runs.append(cnt)
            cnt = 0
    if cnt > 0:
        runs.append(cnt)
    return runs


@dataclass
class EpisodeStats:
    path: Path
    steps: int
    has_collision: bool
    collision_steps: int
    collision_ratio: float
    collision_runs: List[int]
    longest_run: int
    recovered_after_last_collision: bool
    stuck_at_end: bool
    rewards: List[float]
    collision_mask: List[bool]


def build_collision_mask(
    data: dict, rewards: Sequence[float], source: str, threshold: float
) -> List[bool]:
    if source in ("field", "auto"):
        raw_collision = data.get("collision", None)
        if raw_collision is not None:
            arr = flatten_1d_sequence(raw_collision)
            return [bool(v) for v in arr[: len(rewards)]]
        if source == "field":
            return [False] * len(rewards)
    # reward mode (or auto fallback)
    return [float(r) < threshold for r in rewards]


def summarize_rewards(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    uniq, cnt = np.unique(values, return_counts=True)
    top_pairs = sorted(zip(cnt.tolist(), uniq.tolist()), reverse=True)[:10]
    top_map = {str(float(v)): int(c) for c, v in top_pairs}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "top_values": top_map,
    }


def analyze_episode(path: Path, source: str, threshold: float) -> EpisodeStats:
    data = load_pickle(path)
    rewards_raw = flatten_1d_sequence(data.get("reward", []))
    rewards = [float(v) for v in rewards_raw]
    collision_mask = build_collision_mask(data, rewards, source=source, threshold=threshold)

    n = min(len(rewards), len(collision_mask))
    rewards = rewards[:n]
    collision_mask = collision_mask[:n]
    runs = run_lengths(collision_mask)
    has_collision = any(collision_mask)
    longest_run = max(runs) if runs else 0

    last_collision_idx = -1
    for i, c in enumerate(collision_mask):
        if c:
            last_collision_idx = i
    recovered = False
    if last_collision_idx >= 0 and last_collision_idx + 1 < n:
        recovered = any(not x for x in collision_mask[last_collision_idx + 1 :])
    stuck_at_end = has_collision and (last_collision_idx == n - 1)

    collision_steps = int(sum(bool(x) for x in collision_mask))
    return EpisodeStats(
        path=path,
        steps=n,
        has_collision=has_collision,
        collision_steps=collision_steps,
        collision_ratio=(collision_steps / n) if n > 0 else 0.0,
        collision_runs=runs,
        longest_run=longest_run,
        recovered_after_last_collision=recovered,
        stuck_at_end=stuck_at_end,
        rewards=rewards,
        collision_mask=collision_mask,
    )


def _make_grid(images: List[Image.Image], labels: List[str], cols: int = 2) -> Image.Image:
    if not images:
        return Image.new("RGB", (640, 480), color=(255, 255, 255))
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    tile_w = max(img.width for img in images)
    tile_h = max(img.height for img in images)
    title_h = 20
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + title_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * tile_w
        y = r * (tile_h + title_h)
        canvas.paste(img, (x, y + title_h))
        draw.rectangle([(x, y), (x + tile_w, y + title_h)], fill=(255, 255, 255))
        draw.text((x + 4, y + 4), labels[idx], fill=(0, 0, 0), font=font)
    return canvas


def build_visualization(
    episode_stats: List[EpisodeStats],
    root: Path,
    collision_threshold: float,
    output: Path,
    topk: int,
) -> None:
    if not episode_stats:
        return
    by_long = sorted(episode_stats, key=lambda x: x.longest_run, reverse=True)
    recovered = [e for e in by_long if e.recovered_after_last_collision and e.has_collision]
    stuck = [e for e in by_long if e.stuck_at_end]

    selected: List[EpisodeStats] = []
    selected.extend(by_long[:topk])
    selected.extend(recovered[:topk])
    selected.extend(stuck[:topk])

    # unique by path order-preserving
    seen = set()
    uniq: List[EpisodeStats] = []
    for e in selected:
        if e.path in seen:
            continue
        uniq.append(e)
        seen.add(e.path)

    images: List[Image.Image] = []
    labels: List[str] = []
    for e in uniq:
        data = load_pickle(e.path)
        img = render_episode_image(data, e.path, collision_threshold)
        rel = e.path.relative_to(root)
        labels.append(
            f"{rel} | longest={e.longest_run} | collision_steps={e.collision_steps} | recovered={e.recovered_after_last_collision}"
        )
        images.append(img)

    grid = _make_grid(images, labels, cols=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze collided dataset quality.")
    p.add_argument("--root", type=Path, required=True, help="Dataset root containing pickle episodes.")
    p.add_argument(
        "--collision-source",
        type=str,
        default="auto",
        choices=["auto", "field", "reward"],
        help="Collision source: explicit field, reward threshold, or auto fallback.",
    )
    p.add_argument(
        "--collision-threshold",
        type=float,
        default=0.0,
        help="When collision-source=reward/auto-fallback, reward < threshold means collision.",
    )
    p.add_argument("--max-files", type=int, default=0, help="0 means analyze all files.")
    p.add_argument("--topk", type=int, default=4, help="Top-k per category for visualization.")
    p.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Default: tmp/collided_analysis_<timestamp>.json",
    )
    p.add_argument(
        "--viz-output",
        type=Path,
        default=None,
        help="Optional output image path. Default: tmp/collided_analysis_<timestamp>.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_pickles(args.root)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No pickle files found under {args.root}")

    episodes: List[EpisodeStats] = []
    for path in files:
        try:
            episodes.append(
                analyze_episode(
                    path=path,
                    source=args.collision_source,
                    threshold=args.collision_threshold,
                )
            )
        except Exception as exc:
            print(f"[WARN] failed to parse {path}: {exc}")

    if not episodes:
        raise RuntimeError("No valid episodes analyzed.")

    total_steps = sum(e.steps for e in episodes)
    total_collision_steps = sum(e.collision_steps for e in episodes)
    episodes_with_collision = sum(1 for e in episodes if e.has_collision)
    recovered_eps = sum(1 for e in episodes if e.recovered_after_last_collision)
    stuck_eps = sum(1 for e in episodes if e.stuck_at_end)
    all_runs = [r for e in episodes for r in e.collision_runs]
    longest_run_global = max((e.longest_run for e in episodes), default=0)

    rewards_all = np.array([r for e in episodes for r in e.rewards], dtype=np.float32)
    rewards_col = np.array(
        [r for e in episodes for r, c in zip(e.rewards, e.collision_mask) if c], dtype=np.float32
    )
    rewards_non = np.array(
        [r for e in episodes for r, c in zip(e.rewards, e.collision_mask) if not c], dtype=np.float32
    )

    report = {
        "root": str(args.root),
        "files_analyzed": len(episodes),
        "collision_source": args.collision_source,
        "collision_threshold": args.collision_threshold,
        "episode_stats": {
            "episodes_with_collision": episodes_with_collision,
            "episodes_with_collision_ratio": episodes_with_collision / len(episodes),
            "recovered_after_last_collision": recovered_eps,
            "recovered_after_last_collision_ratio": recovered_eps / len(episodes),
            "stuck_at_end": stuck_eps,
            "stuck_at_end_ratio": stuck_eps / len(episodes),
        },
        "step_stats": {
            "total_steps": total_steps,
            "collision_steps": total_collision_steps,
            "collision_step_ratio": (total_collision_steps / total_steps) if total_steps > 0 else 0.0,
        },
        "collision_run_stats": {
            "num_runs": len(all_runs),
            "mean": float(np.mean(all_runs)) if all_runs else 0.0,
            "median": float(np.median(all_runs)) if all_runs else 0.0,
            "p90": float(np.percentile(all_runs, 90)) if all_runs else 0.0,
            "p95": float(np.percentile(all_runs, 95)) if all_runs else 0.0,
            "max": int(max(all_runs)) if all_runs else 0,
            "longest_run_global": int(longest_run_global),
        },
        "reward_stats": {
            "all_steps": summarize_rewards(rewards_all),
            "collision_steps": summarize_rewards(rewards_col),
            "non_collision_steps": summarize_rewards(rewards_non),
        },
        "top_longest_collision_episodes": [
            {
                "path": str(e.path.relative_to(args.root)),
                "steps": e.steps,
                "collision_steps": e.collision_steps,
                "collision_ratio": e.collision_ratio,
                "longest_run": e.longest_run,
                "recovered_after_last_collision": e.recovered_after_last_collision,
                "stuck_at_end": e.stuck_at_end,
            }
            for e in sorted(episodes, key=lambda x: x.longest_run, reverse=True)[:20]
        ],
    }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report_json or (Path("tmp") / f"collided_analysis_{ts}.json")
    viz_path = args.viz_output or (Path("tmp") / f"collided_analysis_{ts}.png")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    build_visualization(
        episode_stats=episodes,
        root=args.root,
        collision_threshold=args.collision_threshold,
        output=viz_path,
        topk=max(1, int(args.topk)),
    )

    print(f"Analyzed {len(episodes)} files from {args.root}")
    print(f"Report JSON: {report_path}")
    print(f"Visualization: {viz_path}")
    print(
        "Summary: "
        f"episodes_with_collision={episodes_with_collision}/{len(episodes)} "
        f"recovered={recovered_eps} stuck={stuck_eps} "
        f"collision_step_ratio={(total_collision_steps / total_steps) if total_steps else 0.0:.4f} "
        f"longest_collision_run={longest_run_global}"
    )


if __name__ == "__main__":
    main()
