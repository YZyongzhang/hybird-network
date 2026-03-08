import os
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from env import Env
from network import HybirdNetwork
from utils.log import logger


def _clone_config_with_dataset_split(config, split_name: str):
    split = str(split_name).strip()
    if not split:
        return config
    cfg = config.clone()
    cfg.defrost()
    cfg.TASK_CONFIG.defrost()
    cfg.TASK_CONFIG.DATASET.SPLIT = split
    cfg.TASK_CONFIG.freeze()
    cfg.freeze()
    return cfg


def _get_audio_tensor(obs: Any) -> torch.Tensor:
    spec = obs["spectrogram"]
    if isinstance(spec, (tuple, list)):
        spec = spec[0]
    return torch.as_tensor(spec, dtype=torch.float32)


def _load_foundation_model(ckpt_path: str, device: torch.device) -> HybirdNetwork:
    if not ckpt_path:
        raise ValueError("Hybrid/Foundation ckpt path is empty. Please set EVAL.HYBRID_CKPT.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Hybrid/Foundation ckpt not found: {ckpt_path}")
    model = HybirdNetwork().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def run_foundation_online_eval(config, eval_config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_name = str(getattr(eval_config, "FOUNDATION_DATASET_SPLIT", "")).strip()
    if not split_name:
        split_name = str(getattr(eval_config, "ONLINE_DATASET_SPLIT", "")).strip()
    eval_cfg = _clone_config_with_dataset_split(config, split_name)

    # foundation == hybrid in this codebase. Prefer EVAL.HYBRID.CKPT, then legacy aliases.
    hybrid_cfg = getattr(eval_config, "HYBRID", None)
    ckpt_path = str(getattr(hybrid_cfg, "CKPT", "")).strip() if hybrid_cfg is not None else ""
    if not ckpt_path:
        ckpt_path = str(getattr(eval_config, "HYBRID_CKPT", "")).strip()
    if not ckpt_path:
        ckpt_path = str(getattr(eval_config, "FOUNDATION_CKPT", "")).strip()
    model = _load_foundation_model(ckpt_path=ckpt_path, device=device)

    env = Env(eval_cfg)
    sim = getattr(getattr(env, "_env", None), "_sim", None)
    max_steps = int(getattr(eval_config, "MAX_STEPS", 200))
    greedy = bool(getattr(eval_config, "FOUNDATION_GREEDY", True))
    log_every = int(getattr(eval_config, "FOUNDATION_LOG_EVERY", 1))
    two_frame = bool(getattr(eval_config, "FOUNDATION_TWO_FRAME", True))
    episodes = int(getattr(eval_config, "EPISODES", 0))
    if episodes <= 0:
        episodes = int(getattr(getattr(env, "_env", None), "number_of_episodes", 0))
    episodes = max(1, episodes)

    total_reward = 0.0
    total_spl = 0.0
    total_distance = 0.0
    logger.info(
        "foundation online eval begin: split=%s episodes=%s max_steps=%s greedy=%s two_frame=%s ckpt=%s",
        eval_cfg.TASK_CONFIG.DATASET.SPLIT,
        episodes,
        max_steps,
        greedy,
        two_frame,
        ckpt_path,
    )

    for ep_idx in tqdm(range(episodes), desc="foundation-online-eval"):
        obs = env.reset()
        info = {"spl": 0.0, "distance_to_goal": -1.0}
        episode_reward = 0.0
        done = False
        step = 0
        pre_rgb = None
        pre_depth = None

        current_ep = getattr(getattr(env, "_env", None), "current_episode", None)
        scene = ""
        episode_id = ""
        if current_ep is not None:
            try:
                scene = str(getattr(current_ep, "scene_id", ""))[-15:-4]
            except Exception:
                scene = ""
            try:
                episode_id = str(getattr(current_ep, "episode_id", ""))
            except Exception:
                episode_id = ""
        logger.info("scene is %s  , episodeid is %s", scene, episode_id)

        for _ in range(max_steps):
            with torch.no_grad():
                rgb = torch.as_tensor(obs["rgb"], dtype=torch.float32) / 255.0
                depth = torch.as_tensor(obs["depth"], dtype=torch.float32)
                audio = _get_audio_tensor(obs)
                if pre_rgb is None:
                    pre_rgb = torch.zeros_like(rgb)
                if pre_depth is None:
                    pre_depth = torch.zeros_like(depth)

                model_rgb = torch.cat([pre_rgb, rgb], dim=2) if two_frame else rgb
                model_depth = torch.cat([pre_depth, depth], dim=2) if two_frame else depth
                logits = model(
                    audio.to(device),
                    model_rgb.to(device),
                    model_depth.to(device),
                )
                probs = F.softmax(logits, dim=-1)
                if greedy:
                    action = int(torch.argmax(probs, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(probs).sample().item())

            obs, reward, done, info = env.step(action=action)
            episode_reward += float(reward)
            step += 1
            pre_rgb = rgb
            pre_depth = depth

            is_collided = False
            if sim is not None and hasattr(sim, "previous_step_collided"):
                try:
                    is_collided = bool(sim.previous_step_collided)
                except Exception:
                    is_collided = False
            logger.info(
                "take action foundation model %s ,reward %s , step %s , done %s , is collided %s",
                action,
                float(reward),
                step - 1,
                bool(done),
                is_collided,
            )
            if done:
                break

        total_reward += episode_reward
        total_spl += float(info.get("spl", 0.0))
        total_distance += float(info.get("distance_to_goal", -1.0))
        logger.info(
            "episode is done , distance_to_goal is %s, spl is %s \nsumreward is %s",
            float(info.get("distance_to_goal", -1.0)),
            float(info.get("spl", 0.0)),
            float(episode_reward),
        )
        if log_every > 0 and ((ep_idx + 1) % log_every == 0):
            logger.info(
                "foundation eval episode=%s/%s reward=%.4f spl=%.4f distance=%.4f steps=%s done=%s",
                ep_idx + 1,
                episodes,
                episode_reward,
                float(info.get("spl", 0.0)),
                float(info.get("distance_to_goal", -1.0)),
                step,
                bool(done),
            )

    avg_reward = total_reward / float(episodes)
    avg_spl = total_spl / float(episodes)
    avg_distance = total_distance / float(episodes)
    logger.info(
        "foundation online eval result: reward=%.4f spl=%.4f distance=%.4f",
        avg_reward,
        avg_spl,
        avg_distance,
    )
