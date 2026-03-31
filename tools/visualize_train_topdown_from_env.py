#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs.default import get_config
from env import Env
from tools.eval_hybrid_sac import _overlay_sound_sources_on_map
from utils.log import logger, setup_run_logger
from utils.visualizations import plot_top_down_map


def _resolve_scene_name(scene_id: str) -> str:
    base = str(scene_id).split("/")[-1]
    if base.endswith(".glb"):
        return base[:-4]
    return base


def _scene_dataset_path(config, split: str) -> str:
    version = str(config.TASK_CONFIG.DATASET.VERSION)
    candidate_plain = os.path.join(
        "data", "datasets", "audionav", "mp3d", version, split, f"{split}.json.gz"
    )
    candidate_sample = os.path.join(
        "data", "datasets", "audionav", "mp3d", version, split, f"{split}_sample_50.json.gz"
    )
    if os.path.exists(candidate_plain):
        return candidate_plain
    if os.path.exists(candidate_sample):
        return candidate_sample
    # fallback to template from config
    return str(config.TASK_CONFIG.DATASET.DATA_PATH).format(version=version, split=split)


def build_env_config(config_path: str, split: str):
    cfg = get_config(config_paths=config_path)
    dataset_path = _scene_dataset_path(cfg, split)
    cfg.defrost()
    cfg.TASK_CONFIG.defrost()
    cfg.TASK_CONFIG.DATASET.SPLIT = split
    cfg.TASK_CONFIG.DATASET.DATA_PATH = dataset_path
    cfg.TASK_CONFIG.freeze()
    cfg.freeze()
    return cfg


def run_visualization(
    config_path: str,
    split: str,
    out_root: str,
    max_step_for_info: int = 2,
):
    cfg = build_env_config(config_path=config_path, split=split)
    env = Env(cfg)

    total_eps = int(getattr(env._env, "number_of_episodes", 0))
    if total_eps <= 0:
        raise RuntimeError(f"No episodes found for split={split}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = Path(out_root) / f"{split}_{date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_scenes = set()
    saved: List[Dict] = []

    for _ in range(total_eps):
        obs = env.reset()
        current_ep = getattr(env._env, "current_episode", None)
        if current_ep is None:
            continue
        scene = _resolve_scene_name(getattr(current_ep, "scene_id", "unknown_scene"))
        if scene in seen_scenes:
            continue

        info = {}
        done = False
        # take a few real env steps to ensure info has top_down_map
        for _step in range(max(1, max_step_for_info)):
            obs, reward, done, info = env.step(action=0)  # stop
            if isinstance(info, dict) and "top_down_map" in info:
                break
            if done:
                break

        if not isinstance(info, dict) or "top_down_map" not in info:
            logger.warning("skip scene=%s episode=%s: no top_down_map in info", scene, getattr(current_ep, "episode_id", ""))
            continue

        top = plot_top_down_map(info)
        goal_positions = [g.position for g in getattr(current_ep, "goals", [])]
        top = _overlay_sound_sources_on_map(env=env, info=info, top_down_map=top, goal_positions=goal_positions)

        episode_id = str(getattr(current_ep, "episode_id", "unknown"))
        img_path = out_dir / f"{scene}_{episode_id}.png"
        meta_path = out_dir / f"{scene}_{episode_id}.json"
        Image.fromarray(top).save(img_path)
        meta = {
            "scene": scene,
            "episode_id": episode_id,
            "scene_id": str(getattr(current_ep, "scene_id", "")),
            "goals": [list(np.asarray(g.position).tolist()) for g in getattr(current_ep, "goals", [])],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        seen_scenes.add(scene)
        saved.append({"scene": scene, "episode_id": episode_id, "image": str(img_path)})
        logger.info("saved topdown scene=%s episode=%s path=%s", scene, episode_id, img_path)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)
    logger.info("done: saved_scenes=%d/%d output=%s", len(saved), len(seen_scenes), out_dir)
    print(f"Saved {len(saved)} scene topdown maps to: {out_dir}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Pick one episode per train scene and render real env topdown map.")
    parser.add_argument("--config", type=str, default="configs/audiogoal.yaml")
    parser.add_argument("--split", type=str, default="train_multiple")
    parser.add_argument("--out-root", type=str, default="img/train_topdown_from_env")
    parser.add_argument("--max-step-for-info", type=int, default=2)
    args = parser.parse_args()

    setup_run_logger(base_dir="logs", run_name=f"visualize_train_topdown_{args.split}")
    run_visualization(
        config_path=args.config,
        split=args.split,
        out_root=args.out_root,
        max_step_for_info=args.max_step_for_info,
    )


if __name__ == "__main__":
    main()
