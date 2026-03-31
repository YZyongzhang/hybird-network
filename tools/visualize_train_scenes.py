#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_scene_positions(path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    scene_name = path.stem.replace(".json", "")

    starts: List[List[float]] = []
    goals: List[List[float]] = []
    for ep in episodes:
        s = ep.get("start_position", None)
        gs = ep.get("goals", [])
        if s is None or not gs:
            continue
        g0 = gs[0].get("position", None)
        if g0 is None:
            continue
        starts.append([float(s[0]), float(s[2])])  # x, z
        goals.append([float(g0[0]), float(g0[2])])  # x, z

    return scene_name, np.asarray(starts, dtype=np.float32), np.asarray(goals, dtype=np.float32)


def _plot_scene(scene: str, starts: np.ndarray, goals: np.ndarray, out_path: Path, max_lines: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if starts.size > 0:
        ax.scatter(starts[:, 0], starts[:, 1], s=10, alpha=0.45, c="#1f77b4", label="start")
    if goals.size > 0:
        ax.scatter(goals[:, 0], goals[:, 1], s=10, alpha=0.45, c="#d62728", label="goal(sound)")

    n = min(len(starts), len(goals), max_lines)
    if n > 0:
        idx = np.linspace(0, len(starts) - 1, n, dtype=int)
        for i in idx:
            ax.plot([starts[i, 0], goals[i, 0]], [starts[i, 1], goals[i, 1]], c="#999999", alpha=0.08, lw=0.8)

    ax.set_title(f"{scene} | episodes={len(starts)}")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_index(scene_imgs: List[Tuple[str, Path]], out_path: Path, cols: int = 5) -> None:
    n = len(scene_imgs)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        scene, img_path = scene_imgs[i]
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(scene, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize each train scene from AudioNav episode json.gz.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/audionav/mp3d/v1/train_multiple/content"),
        help="Directory containing per-scene json.gz files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("img/scene_vis"),
        help="Root directory to save generated scene visualizations.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=300,
        help="Max sampled start-goal lines per scene for readability.",
    )
    args = parser.parse_args()

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.output_root / f"train_multiple_{date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"No scene json.gz files found in: {args.input_dir}")

    scene_imgs: List[Tuple[str, Path]] = []
    summary = []
    for f in files:
        scene, starts, goals = _load_scene_positions(f)
        out_path = out_dir / f"{scene}.png"
        _plot_scene(scene, starts, goals, out_path, max_lines=max(0, args.max_lines))
        scene_imgs.append((scene, out_path))
        summary.append({"scene": scene, "episodes": int(len(starts)), "file": str(f)})

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot_index(scene_imgs, out_dir / "all_scenes_index.png", cols=5)
    print(f"Generated {len(scene_imgs)} scene plots in: {out_dir}")
    print(f"Index image: {out_dir / 'all_scenes_index.png'}")


if __name__ == "__main__":
    main()
