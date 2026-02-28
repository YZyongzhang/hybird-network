#!/usr/bin/env python3
"""
Visualize random trajectories from a collided pickle dataset.

For each sampled episode, draw the top-down map, plot the agent path,
highlight collision steps (inferred from negative rewards), and mark
start/end positions. Useful for sanity checking that the episode
experienced a collision and then continued moving.
"""

import argparse
import datetime as dt
import os
import pickle
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import math

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


# ---------------------------------------------------------------------------
# Pickle loading helpers
# ---------------------------------------------------------------------------
class Dummy(dict):
    """Fallback object used to satisfy habitat/soundspaces classes during pickle load."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):  # pylint: disable=unused-argument
        return Dummy()

    def __getattr__(self, name):
        return self.get(name, Dummy())

    def __setattr__(self, name, value):
        self[name] = value


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that replaces habitat/soundspaces classes with Dummy."""

    _SAFE_PREFIXES: Tuple[str, ...] = ("habitat", "soundspaces", "magnum")

    def find_class(self, module, name):  # noqa: D401
        if module.startswith(self._SAFE_PREFIXES):
            return Dummy
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return SafeUnpickler(handle).load()


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------
COLOR_PALETTE = {
    0: (245, 245, 245),  # unknown
    1: (120, 120, 120),  # obstacles
    2: (210, 210, 210),  # navigable space
    4: (80, 180, 80),  # goal regions / sounds
    10: (60, 160, 220),  # explored area (if present)
    11: (250, 170, 60),  # frontier / other annotations
}


def colorize_map(map_array: np.ndarray) -> np.ndarray:
    """Convert a single-channel topdown map into an RGB image."""
    rgb = np.zeros((*map_array.shape, 3), dtype=np.uint8)
    for value, color in COLOR_PALETTE.items():
        rgb[map_array == value] = color
    # Fallback: any remaining cells -> light gray
    rgb[(rgb == 0).all(axis=-1)] = (230, 230, 230)
    return rgb


def extract_agent_coords(info_steps: Sequence[dict]) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for step in info_steps:
        td = step.get("top_down_map")
        if not td:
            continue
        coord = td.get("agent_map_coord")
        if coord:
            coords.append(tuple(int(v) for v in coord))
    return coords


def render_episode_image(data: dict, path: Path, collision_threshold: float) -> Image.Image:
    rewards = list(data.get("reward", []))
    info_steps: Sequence[dict] = data.get("info", [])
    if not rewards or not info_steps:
        img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Empty episode", fill=(0, 0, 0), font=ImageFont.load_default())
        return img

    coords = extract_agent_coords(info_steps)
    if not coords:
        img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Missing agent coords", fill=(0, 0, 0), font=ImageFont.load_default())
        return img

    last_map = info_steps[min(len(info_steps) - 1, len(coords) - 1)]["top_down_map"]["map"]
    map_img = colorize_map(np.asarray(last_map))

    img = Image.fromarray(map_img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle([(0, 0), (img.width, 18)], fill=(255, 255, 255))
    draw.text((4, 2), f"{path.parent.name}/{path.name}", fill=(0, 0, 0), font=font)

    xs = [c[1] for c in coords]
    ys = [c[0] for c in coords]

    path_points = list(zip(xs, ys))
    if len(path_points) >= 2:
        draw.line(path_points, fill=(0, 215, 255), width=3)
    radius = 4
    draw.ellipse(
        [(xs[0] - radius, ys[0] - radius), (xs[0] + radius, ys[0] + radius)],
        fill=(51, 204, 51),
    )
    draw.ellipse(
        [(xs[-1] - radius, ys[-1] - radius), (xs[-1] + radius, ys[-1] + radius)],
        fill=(51, 102, 255),
    )

    collision_steps = [idx for idx, r in enumerate(rewards[: len(xs)]) if r < collision_threshold]
    for step in collision_steps:
        cx, cy = xs[step], ys[step]
        cross_size = 6
        draw.line([(cx - cross_size, cy - cross_size), (cx + cross_size, cy + cross_size)], fill=(255, 51, 51), width=2)
        draw.line([(cx - cross_size, cy + cross_size), (cx + cross_size, cy - cross_size)], fill=(255, 51, 51), width=2)

    post_collision = [idx for idx in collision_steps if any(r > collision_threshold for r in rewards[idx + 1 :])]
    text = f"Collisions: {len(collision_steps)}"
    if post_collision:
        text += " | Recovered"
    draw.rectangle([(4, img.height - 20), (220, img.height - 4)], fill=(255, 255, 255, 200))
    draw.text((8, img.height - 18), text, fill=(200, 0, 0), font=font)
    return img


def sample_files(dataset_root: Path, num_samples: int, seed: int) -> List[Path]:
    all_files: List[Path] = []
    for sub in dataset_root.iterdir():
        if not sub.is_dir():
            continue
        for item in sub.iterdir():
            if item.is_file() and item.suffix in {".pkl", ".pickle", ""} and item.name != "a.md":
                all_files.append(item)
    if not all_files:
        raise RuntimeError(f"No pickle files found under {dataset_root}")
    rng = random.Random(seed)
    rng.shuffle(all_files)
    return all_files[: min(num_samples, len(all_files))]


def main():
    parser = argparse.ArgumentParser(description="Visualize collided dataset topdown maps.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing collided pickle files.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of random episodes to plot.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.0,
        help="Rewards strictly below this value are treated as collisions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to tmp/collided_topdown_<timestamp>.png",
    )
    args = parser.parse_args()

    dataset_root = args.root
    samples = sample_files(dataset_root, args.num_samples, args.seed)

    cols = min(2, len(samples)) or 1
    rows = math.ceil(len(samples) / cols)
    rendered: List[Image.Image] = []
    report_lines: List[str] = []
    for path in samples:
        episode = load_pickle(path)
        rendered.append(render_episode_image(episode, path, args.collision_threshold))
        rewards = episode.get("reward", [])
        min_reward = min(rewards) if rewards else float("nan")
        max_reward = max(rewards) if rewards else float("nan")
        report_lines.append(f"{path.relative_to(dataset_root)} | steps={len(rewards)} | min_reward={min_reward:.3f} | max_reward={max_reward:.3f}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or Path("tmp") / f"collided_topdown_{timestamp}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rendered:
        tile_w = max(img.width for img in rendered)
        tile_h = max(img.height for img in rendered)
        grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(255, 255, 255))
        for idx, img in enumerate(rendered):
            row = idx // cols
            col = idx % cols
            x = col * tile_w
            y = row * tile_h
            grid.paste(img, (x, y))
        grid.save(output_path)
    else:
        Image.new("RGB", (640, 480), color=(255, 255, 255)).save(output_path)

    print(f"Wrote visualization to {output_path}")
    print("Sample summary:")
    for line in report_lines:
        print("  ", line)


if __name__ == "__main__":
    main()
