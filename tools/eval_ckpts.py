#!/usr/bin/env python3
"""
Sweep checkpoints in a folder, evaluate one-by-one, and select the best by SPL.

Examples:
  python tools/eval_ckpts.py \
      --config configs/audiogoal.yaml \
      --ckpt-dir media/TRAIN/20260326_134342 \
      --eval-type OfflineRL_v1_5

  python tools/eval_ckpts.py \
      --config configs/audiogoal.yaml \
      --ckpt-dir media/ONLINE/yyy \
      --eval-type OnlineRL \
      --field ONLINE_MODEL_PATH
"""

import argparse
import json
import os
import random
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = None
setup_run_logger = None
Env = None
get_config = None


def _lazy_import_project_modules() -> None:
    global logger, setup_run_logger, Env, get_config
    if logger is not None:
        return
    from configs.default import get_config as _get_config
    from env import Env as _Env
    from utils.log import logger as _logger, setup_run_logger as _setup_run_logger

    get_config = _get_config
    Env = _Env
    logger = _logger
    setup_run_logger = _setup_run_logger


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_onlinerl_setup(online_cfg):
    model_name = str(getattr(online_cfg, "model", "v1")).lower().strip()
    foundation_ckpt = str(getattr(online_cfg, "FOUNDATION_CKPT", "")).strip()
    freeze_backbone = bool(getattr(online_cfg, "freeze_backbone", False))
    experiment_mode = str(getattr(online_cfg, "experiment", "custom")).lower().strip()

    if experiment_mode in {"v1_freeze", "freeze_foundation", "freezefoundationmodel"}:
        model_name = "v1"
        freeze_backbone = True
        if not foundation_ckpt:
            raise ValueError(
                "TRAIN.ONLINE.FOUNDATION_CKPT is required when TRAIN.ONLINE.experiment=v1_freeze."
            )
    elif experiment_mode in {"v2_scratch", "scratch", "from_scratch"}:
        model_name = "v2"
        foundation_ckpt = ""
        freeze_backbone = False
    elif experiment_mode in {"custom", ""}:
        pass
    else:
        raise ValueError(
            f"Unsupported TRAIN.ONLINE.experiment: {experiment_mode}. "
            "Use one of [custom, v1_freeze, v2_scratch]."
        )

    return model_name, foundation_ckpt, freeze_backbone, experiment_mode


def _obs_to_inputs(obs):
    rgb = torch.as_tensor(obs["rgb"], dtype=torch.float32) / 255.0
    depth = torch.as_tensor(obs["depth"], dtype=torch.float32)
    audio = obs["spectrogram"]
    if isinstance(audio, (tuple, list)):
        audio = audio[0]
    audio = torch.as_tensor(audio, dtype=torch.float32)
    return audio, rgb, depth


def _rotation_to_list(rotation):
    if rotation is None:
        return [0.0, 0.0, 0.0, 1.0]
    if all(hasattr(rotation, k) for k in ("x", "y", "z", "w")):
        return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
    if all(hasattr(rotation, k) for k in ("w", "x", "y", "z")):
        return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
    if isinstance(rotation, (tuple, list)):
        values = [float(v) for v in rotation]
        if len(values) >= 4:
            return values[:4]
    return [0.0, 0.0, 0.0, 1.0]


def _get_pose_from_env(env):
    sim = getattr(getattr(env, "_env", None), "_sim", None)
    if sim is None or not hasattr(sim, "get_agent_state"):
        return torch.zeros(7, dtype=torch.float32)
    try:
        state = sim.get_agent_state()
        pos = [float(v) for v in state.position]
        rot = _rotation_to_list(state.rotation)
        return torch.as_tensor(pos + rot, dtype=torch.float32)
    except Exception:
        return torch.zeros(7, dtype=torch.float32)


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


def _run_onlinerl_eval(config, eval_cfg, model_path_override: str = ""):
    from network import OnlineRLV1, OnlineRLV2

    device = _get_device()
    model_path = str(model_path_override).strip() if model_path_override else str(getattr(eval_cfg, "ONLINE_MODEL_PATH", "")).strip()
    if not model_path:
        raise ValueError("EVAL.ONLINE_MODEL_PATH is required when EVAL.TYPE=OnlineRL.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"EVAL.ONLINE_MODEL_PATH not found: {model_path}")

    seed = int(getattr(eval_cfg, "SEED", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    split_name = str(getattr(eval_cfg, "ONLINE_DATASET_SPLIT", "")).strip()
    run_cfg = _clone_config_with_dataset_split(config, split_name)

    env = Env(run_cfg)
    sim = getattr(getattr(env, "_env", None), "_sim", None)

    train_online_cfg = run_cfg.TASK_CONFIG.TRAIN.ONLINE
    model_name, foundation_ckpt, freeze_backbone, experiment_mode = _resolve_onlinerl_setup(train_online_cfg)
    model_name = str(getattr(eval_cfg, "ONLINE_MODEL", model_name)).lower().strip()
    action_dim = int(getattr(train_online_cfg, "action_dim", 4))
    hidden_dim = int(getattr(train_online_cfg, "hidden_dim", 256))
    use_pose_encoder = bool(getattr(train_online_cfg, "use_pose_encoder", False))
    pose_dim = int(getattr(train_online_cfg, "pose_dim", 7))
    pose_hidden_dim = int(getattr(train_online_cfg, "pose_hidden_dim", 64))

    if model_name == "v1":
        if not foundation_ckpt:
            raise ValueError("TRAIN.ONLINE.FOUNDATION_CKPT is required for OnlineRL v1 eval.")
        agent = OnlineRLV1(
            action_dim=action_dim,
            foundation_ckpt=foundation_ckpt,
            hidden_dim=hidden_dim,
            use_pose_encoder=use_pose_encoder,
            pose_dim=pose_dim,
            pose_hidden_dim=pose_hidden_dim,
            freeze_backbone=freeze_backbone,
            device=device,
        )
    elif model_name == "v2":
        agent = OnlineRLV2(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            foundation_ckpt=foundation_ckpt if foundation_ckpt else None,
            use_pose_encoder=use_pose_encoder,
            pose_dim=pose_dim,
            pose_hidden_dim=pose_hidden_dim,
            freeze_backbone=freeze_backbone,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported EVAL.ONLINE_MODEL: {model_name}")

    logger.info("loading onlinerl eval ckpt: %s", model_path)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    agent.eval()

    max_steps = int(getattr(eval_cfg, "MAX_STEPS", 200))
    greedy = bool(getattr(eval_cfg, "ONLINE_GREEDY", True))
    log_every = int(getattr(eval_cfg, "ONLINE_LOG_EVERY", 1))
    episodes = int(getattr(eval_cfg, "EPISODES", 0))
    if episodes <= 0:
        episodes = int(getattr(getattr(env, "_env", None), "number_of_episodes", 0))
    episodes = max(1, episodes)

    total_reward = 0.0
    total_spl = 0.0
    total_distance = 0.0
    logger.info(
        "onlinerl eval begin: model=%s experiment=%s split=%s episodes=%s max_steps=%s greedy=%s seed=%s",
        model_name,
        experiment_mode,
        run_cfg.TASK_CONFIG.DATASET.SPLIT,
        episodes,
        max_steps,
        greedy,
        seed,
    )

    try:
        for ep_idx in range(episodes):
            obs = env.reset()
            info = {"spl": 0.0, "distance_to_goal": -1.0}
            pre_rgb = None
            pre_depth = None
            episode_reward = 0.0
            done = False
            steps_taken = 0

            for step in range(max_steps):
                audio, rgb, depth = _obs_to_inputs(obs)
                pose = _get_pose_from_env(env)
                if pre_rgb is None:
                    pre_rgb = torch.zeros_like(rgb)
                if pre_depth is None:
                    pre_depth = torch.zeros_like(depth)
                trgb = torch.cat([pre_rgb, rgb], dim=2)
                tdepth = torch.cat([pre_depth, depth], dim=2)

                with torch.no_grad():
                    emb = agent.encode(
                        audio.unsqueeze(0).to(device),
                        trgb.unsqueeze(0).to(device),
                        tdepth.unsqueeze(0).to(device),
                        pose.unsqueeze(0).to(device),
                    ).float()
                    logits = agent.policy_head(emb)
                    probs = F.softmax(logits, dim=-1)
                    if greedy:
                        action = int(torch.argmax(probs, dim=-1).item())
                    else:
                        action = int(torch.distributions.Categorical(probs).sample().item())

                obs, reward, done, info = env.step(action=action)
                is_collided = False
                if sim is not None and hasattr(sim, "previous_step_collided"):
                    try:
                        is_collided = bool(sim.previous_step_collided)
                    except Exception:
                        is_collided = False
                logger.info(
                    "take action sac model %s ,reward %s , step %s , done %s , is collided %s",
                    action,
                    float(reward),
                    step,
                    bool(done),
                    is_collided,
                )
                episode_reward += float(reward)
                steps_taken = step + 1
                pre_rgb = rgb
                pre_depth = depth
                if done:
                    break

            total_reward += episode_reward
            total_spl += float(info.get("spl", 0.0))
            total_distance += float(info.get("distance_to_goal", -1.0))
            if log_every > 0 and ((ep_idx + 1) % log_every == 0):
                logger.info(
                    "onlinerl eval episode=%s/%s reward=%.4f spl=%.4f distance=%.4f steps=%s done=%s",
                    ep_idx + 1,
                    episodes,
                    episode_reward,
                    float(info.get("spl", 0.0)),
                    float(info.get("distance_to_goal", -1.0)),
                    steps_taken,
                    bool(done),
                )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        del env

    avg_reward = total_reward / float(episodes)
    avg_spl = total_spl / float(episodes)
    avg_distance = total_distance / float(episodes)
    logger.info(
        "onlinerl eval result: reward=%.4f spl=%.4f distance=%.4f",
        avg_reward,
        avg_spl,
        avg_distance,
    )
    return {
        "avg_reward": float(avg_reward),
        "avg_spl": float(avg_spl),
        "avg_distance": float(avg_distance),
        "episodes": int(episodes),
    }


def _build_two_frame(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    prev_rgb: Optional[torch.Tensor],
    prev_depth: Optional[torch.Tensor],
):
    if prev_rgb is None:
        prev_rgb = torch.zeros_like(rgb)
    if prev_depth is None:
        prev_depth = torch.zeros_like(depth)
    return torch.cat([prev_rgb, rgb], dim=2), torch.cat([prev_depth, depth], dim=2)


def _build_v1_3_model(offline_cfg, device: torch.device):
    from network import SAC_Hybird_model

    return SAC_Hybird_model(
        state_dim=offline_cfg.state_dim,
        hidden_dim=offline_cfg.hidden_dim,
        action_dim=offline_cfg.action_dim,
        actor_lr=offline_cfg.lr,
        critic_lr=offline_cfg.lr,
        alpha_lr=offline_cfg.lr,
        target_entropy=offline_cfg.target_entropy,
        tau=offline_cfg.tau,
        gamma=offline_cfg.gamma,
        beta=offline_cfg.beta,
        device=device,
    ).to(device)


def _build_v1_5_model(offline_cfg, device: torch.device):
    from network import SAC_LSTM_CQL_v1_5

    return SAC_LSTM_CQL_v1_5(
        state_dim=offline_cfg.state_dim,
        hidden_dim=offline_cfg.hidden_dim,
        action_dim=offline_cfg.action_dim,
        actor_lr=offline_cfg.actor_lr,
        critic_lr=offline_cfg.critic_lr,
        alpha_lr=offline_cfg.alpha_lr,
        target_entropy=offline_cfg.target_entropy,
        tau=offline_cfg.tau,
        gamma=offline_cfg.gamma,
        beta=offline_cfg.beta,
        device=device,
        lstm_hidden_dim=int(getattr(offline_cfg, "lstm_hidden_dim", offline_cfg.state_dim)),
        lstm_num_layers=int(getattr(offline_cfg, "lstm_num_layers", 1)),
    ).to(device)


def _load_optional_subweights(model, temporal_ckpt="", actor_ckpt="", critic1_ckpt="", critic2_ckpt=""):
    if temporal_ckpt:
        model.temporal_encoder.load_state_dict(torch.load(temporal_ckpt, map_location=model.device))
    if actor_ckpt:
        model.actor.load_state_dict(torch.load(actor_ckpt, map_location=model.device))
    if critic1_ckpt:
        model.critic_1.load_state_dict(torch.load(critic1_ckpt, map_location=model.device))
        model.target_critic_1.load_state_dict(model.critic_1.state_dict())
    if critic2_ckpt:
        model.critic_2.load_state_dict(torch.load(critic2_ckpt, map_location=model.device))
        model.target_critic_2.load_state_dict(model.critic_2.state_dict())


def _run_offline_v1_3_eval(config, eval_cfg, sac_ckpt_override: str = ""):
    device = _get_device()
    seed = int(getattr(eval_cfg, "SEED", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sac_ckpt = sac_ckpt_override if sac_ckpt_override else str(getattr(eval_cfg, "SAC_CKPT", "")).strip()
    if not sac_ckpt:
        raise ValueError("EVAL.SAC_CKPT is required for OfflineRL_v1_3.")
    if not os.path.exists(sac_ckpt):
        raise FileNotFoundError(f"SAC ckpt not found: {sac_ckpt}")

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    sac_model = _build_v1_3_model(offline_cfg, device=device)
    sac_model.load_state_dict(torch.load(sac_ckpt, map_location=device))
    sac_model.eval()

    env = Env(config)
    total_eps = int(getattr(getattr(env, "_env", None), "number_of_episodes", 0))
    episodes = int(getattr(eval_cfg, "EPISODES", 0))
    run_episodes = episodes if episodes > 0 else total_eps
    run_episodes = max(1, min(run_episodes, total_eps if total_eps > 0 else run_episodes))
    max_steps = int(getattr(eval_cfg, "MAX_STEPS", 200))
    deterministic = not bool(getattr(eval_cfg, "STOCHASTIC_SAC", False))

    reward_sum = 0.0
    spl_sum = 0.0
    success_sum = 0.0
    steps_sum = 0

    try:
        for ep_idx in range(run_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            info = {}
            prev_rgb = None
            prev_depth = None
            while (not done) and ep_steps < max_steps:
                audio, rgb, depth = _obs_to_inputs(obs)
                trgb, tdepth = _build_two_frame(rgb, depth, prev_rgb, prev_depth)
                with torch.no_grad():
                    if deterministic:
                        state = sac_model.hybird.embedding_forward(
                            audio.to(device), trgb.to(device), tdepth.to(device)
                        ).float()
                        probs = sac_model.actor(state)
                        action = int(torch.argmax(probs, dim=1).item())
                    else:
                        action = int(sac_model.get_action((audio.to(device), trgb.to(device), tdepth.to(device))))
                obs, reward, done, info = env.step(action=action)
                ep_reward += float(reward)
                ep_steps += 1
                prev_rgb, prev_depth = rgb, depth

            spl = float(info.get("spl", 0.0))
            success = float(info.get("success", 1.0 if spl > 0 else 0.0))
            reward_sum += ep_reward
            spl_sum += spl
            success_sum += success
            steps_sum += ep_steps
            logger.info(
                "[v1_3] episode %d/%d reward=%.4f spl=%.4f success=%.0f steps=%d",
                ep_idx + 1,
                run_episodes,
                ep_reward,
                spl,
                success,
                ep_steps,
            )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        del env

    return {
        "avg_reward": float(reward_sum / run_episodes),
        "avg_spl": float(spl_sum / run_episodes),
        "success_rate": float(success_sum / run_episodes),
        "avg_steps": float(steps_sum / run_episodes),
        "episodes": int(run_episodes),
    }


def _run_offline_v1_5_eval(config, eval_cfg, sac_ckpt_override: str = ""):
    from network import HybirdNetwork

    device = _get_device()
    seed = int(getattr(eval_cfg, "SEED", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hybrid_ckpt = str(getattr(eval_cfg, "HYBRID_CKPT", "")).strip()
    sac_ckpt = sac_ckpt_override if sac_ckpt_override else str(getattr(eval_cfg, "SAC_CKPT", "")).strip()
    if not hybrid_ckpt:
        raise ValueError("EVAL.HYBRID_CKPT is required for OfflineRL_v1_5.")
    if not os.path.exists(hybrid_ckpt):
        raise FileNotFoundError(f"Hybrid ckpt not found: {hybrid_ckpt}")
    if sac_ckpt and (not os.path.exists(sac_ckpt)):
        raise FileNotFoundError(f"SAC ckpt not found: {sac_ckpt}")

    hybrid_model = HybirdNetwork().to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    hybrid_model.eval()

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    sac_model = _build_v1_5_model(offline_cfg, device=device)
    if sac_ckpt:
        sac_model.load_state_dict(torch.load(sac_ckpt, map_location=device), strict=False)
    _load_optional_subweights(
        sac_model,
        temporal_ckpt=str(getattr(eval_cfg, "TEMPORAL_CKPT", "")),
        actor_ckpt=str(getattr(eval_cfg, "ACTOR_CKPT", "")),
        critic1_ckpt=str(getattr(eval_cfg, "CRITIC1_CKPT", "")),
        critic2_ckpt=str(getattr(eval_cfg, "CRITIC2_CKPT", "")),
    )
    sac_model.eval()

    env = Env(config)
    total_eps = int(getattr(getattr(env, "_env", None), "number_of_episodes", 0))
    episodes = int(getattr(eval_cfg, "EPISODES", 0))
    run_episodes = episodes if episodes > 0 else total_eps
    run_episodes = max(1, min(run_episodes, total_eps if total_eps > 0 else run_episodes))
    max_steps = int(getattr(eval_cfg, "MAX_STEPS", 200))
    deterministic = not bool(getattr(eval_cfg, "STOCHASTIC_SAC", False))

    reward_sum = 0.0
    spl_sum = 0.0
    success_sum = 0.0
    steps_sum = 0
    seq_len = 5

    try:
        for ep_idx in range(run_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            info = {}
            prev_rgb = None
            prev_depth = None
            seq_states = deque(maxlen=seq_len)

            while (not done) and ep_steps < max_steps:
                audio, rgb, depth = _obs_to_inputs(obs)
                trgb, tdepth = _build_two_frame(rgb, depth, prev_rgb, prev_depth)
                with torch.no_grad():
                    emb = hybrid_model.embedding_forward(
                        audio.to(device), trgb.to(device), tdepth.to(device)
                    ).detach().cpu()
                seq_states.append(emb)
                while len(seq_states) < seq_len:
                    seq_states.appendleft(torch.zeros_like(emb))
                seq_tensor = torch.stack(list(seq_states), dim=0).unsqueeze(0).to(device)
                action = int(sac_model.get_action(seq_tensor, eval=deterministic))

                obs, reward, done, info = env.step(action=action)
                ep_reward += float(reward)
                ep_steps += 1
                prev_rgb, prev_depth = rgb, depth

            spl = float(info.get("spl", 0.0))
            success = float(info.get("success", 1.0 if spl > 0 else 0.0))
            reward_sum += ep_reward
            spl_sum += spl
            success_sum += success
            steps_sum += ep_steps
            logger.info(
                "[v1_5] episode %d/%d reward=%.4f spl=%.4f success=%.0f steps=%d",
                ep_idx + 1,
                run_episodes,
                ep_reward,
                spl,
                success,
                ep_steps,
            )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        del env
    
    return {
        "avg_reward": float(reward_sum / run_episodes),
        "avg_spl": float(spl_sum / run_episodes),
        "success_rate": float(success_sum / run_episodes),
        "avg_steps": float(steps_sum / run_episodes),
        "episodes": int(run_episodes),
    }


def _extract_eval_spl(eval_result) -> Optional[float]:
    if eval_result is None:
        return None
    if hasattr(eval_result, "avg_spl"):
        return float(eval_result.avg_spl)
    if isinstance(eval_result, dict):
        if "avg_spl" in eval_result:
            return float(eval_result["avg_spl"])
        if "spl" in eval_result:
            return float(eval_result["spl"])
    if isinstance(eval_result, (tuple, list)) and len(eval_result) >= 2:
        return float(eval_result[1])
    return None


def _ckpt_sort_key(path_obj: Path, sort_by_epoch: bool):
    if not sort_by_epoch:
        return path_obj.name
    found = re.findall(r"(\d+)", path_obj.stem)
    epoch = int(found[-1]) if found else -1
    return (epoch, path_obj.name)


def _resolve_default_field(eval_type: str) -> str:
    if eval_type in {"OfflineRL_v1_3", "OfflineRL_v1_5"}:
        return "SAC_CKPT"
    if eval_type == "OnlineRL":
        return "ONLINE_MODEL_PATH"
    raise ValueError(f"Unsupported eval type for auto field resolve: {eval_type}")


def _run_eval_once(config, eval_cfg, eval_type: str, ckpt_override: str = "", ckpt_field: str = ""):
    if eval_type == "OfflineRL_v1_3":
        sac_ckpt = ckpt_override if ckpt_field == "SAC_CKPT" else ""
        return _run_offline_v1_3_eval(config=config, eval_cfg=eval_cfg, sac_ckpt_override=sac_ckpt)

    if eval_type == "OfflineRL_v1_5":
        sac_ckpt = ckpt_override if ckpt_field == "SAC_CKPT" else ""
        return _run_offline_v1_5_eval(config=config, eval_cfg=eval_cfg, sac_ckpt_override=sac_ckpt)

    if eval_type == "OnlineRL":
        model_path_override = ckpt_override if ckpt_field == "ONLINE_MODEL_PATH" else ""
        return _run_onlinerl_eval(config, eval_cfg, model_path_override=model_path_override)

    raise ValueError(f"Unsupported EVAL.TYPE: {eval_type}")


@dataclass
class SweepRecord:
    ckpt: str
    spl: float
    ok: bool
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints one-by-one and pick best SPL.")
    parser.add_argument("--config", type=str, default="configs/audiogoal.yaml", help="Config yaml path.")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint folder to sweep.")
    parser.add_argument("--pattern", type=str, default="*.pth", help="Glob pattern for checkpoints.")
    parser.add_argument("--recursive", action="store_true", help="Use recursive glob.")
    parser.add_argument("--sort-by-epoch", action="store_true", default=True, help="Sort by parsed epoch number.")
    parser.add_argument("--no-sort-by-epoch", dest="sort_by_epoch", action="store_false", help="Sort by filename.")
    parser.add_argument("--max-ckpts", type=int, default=0, help="0 means evaluate all.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one ckpt fails.")

    parser.add_argument(
        "--eval-type",
        type=str,
        default="OfflineRL_v1_5",
        choices=["", "OfflineRL_v1_3", "OfflineRL_v1_5", "OnlineRL"],
        help="Override EVAL.TYPE from config.",
    )
    parser.add_argument("--field", type=str, default="", help="Override ckpt target field. Usually SAC_CKPT or ONLINE_MODEL_PATH.")

    parser.add_argument("--episodes", type=int, default=None, help="Override EVAL.EPISODES.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override EVAL.MAX_STEPS.")
    parser.add_argument("--seed", type=int, default=None, help="Override EVAL.SEED.")
    parser.add_argument("--stochastic-sac", action="store_true", help="Override EVAL.STOCHASTIC_SAC=True.")
    parser.add_argument("--split", type=str, default="", help="Override dataset split for this sweep.")

    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Extra opts merged into config, e.g. --opts TASK_CONFIG.X.Y 1")
    parser.add_argument("--save-json", type=str, default="", help="Output json path. Default logs/eval_ckpts_*.json")
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    _lazy_import_project_modules()
    cfg = get_config(config_paths=args.config, opts=args.opts)
    cfg = cfg.clone()
    cfg.defrost()
    cfg.TASK_CONFIG.defrost()

    eval_cfg = cfg.TASK_CONFIG.EVAL
    if args.eval_type:
        eval_cfg.TYPE = args.eval_type
    if args.episodes is not None:
        eval_cfg.EPISODES = int(args.episodes)
    if args.max_steps is not None:
        eval_cfg.MAX_STEPS = int(args.max_steps)
    if args.seed is not None:
        eval_cfg.SEED = int(args.seed)
    if args.stochastic_sac:
        eval_cfg.STOCHASTIC_SAC = True

    if args.split:
        cfg.TASK_CONFIG.DATASET.SPLIT = args.split
        if hasattr(eval_cfg, "ONLINE_DATASET_SPLIT"):
            eval_cfg.ONLINE_DATASET_SPLIT = args.split

    cfg.TASK_CONFIG.freeze()
    cfg.freeze()
    return cfg


def collect_ckpts(ckpt_dir: Path, pattern: str, recursive: bool, sort_by_epoch: bool, max_ckpts: int) -> List[Path]:
    iterator = ckpt_dir.rglob(pattern) if recursive else ckpt_dir.glob(pattern)
    ckpts = [p for p in iterator if p.is_file()]
    ckpts = sorted(ckpts, key=lambda p: _ckpt_sort_key(p, sort_by_epoch))
    if max_ckpts > 0:
        ckpts = ckpts[:max_ckpts]
    return ckpts


def main() -> None:
    args = parse_args()
    _lazy_import_project_modules()
    log_path = setup_run_logger(base_dir="logs", run_name="eval_ckpts_sweep")
    logger.info("eval_ckpts started, log_path=%s", log_path)
    logger.info("args=%s", args)

    ckpt_dir = Path(args.ckpt_dir).expanduser()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        raise NotADirectoryError(f"ckpt_dir is not a directory: {ckpt_dir}")

    base_cfg = build_config(args)
    eval_type = str(base_cfg.TASK_CONFIG.EVAL.TYPE)
    ckpt_field = args.field.strip() if args.field.strip() else _resolve_default_field(eval_type)

    ckpts = collect_ckpts(
        ckpt_dir=ckpt_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        sort_by_epoch=args.sort_by_epoch,
        max_ckpts=int(args.max_ckpts),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: dir={ckpt_dir} pattern={args.pattern} recursive={args.recursive}")

    logger.info(
        "sweep begin: eval_type=%s field=%s split=%s total=%d strategy=coarse(step=5)+fine(best±3)",
        eval_type,
        ckpt_field,
        base_cfg.TASK_CONFIG.DATASET.SPLIT,
        len(ckpts),
    )

    stage1_indices = list(range(0, len(ckpts), 5))
    logger.info("[stage1] coarse sweep indices=%s", stage1_indices)

    records: List[SweepRecord] = []
    evaluated_indices: Set[int] = set()
    index_to_record: Dict[int, SweepRecord] = {}
    stopped_early = False

    def _eval_index(idx0: int, stage_name: str) -> bool:
        nonlocal stopped_early
        if idx0 in evaluated_indices:
            cached = index_to_record[idx0]
            logger.info(
                "[%s] reuse evaluated idx=%d/%d ckpt=%s spl=%s ok=%s",
                stage_name,
                idx0 + 1,
                len(ckpts),
                cached.ckpt,
                f"{cached.spl:.6f}" if cached.ok else "-inf",
                cached.ok,
            )
            return True

        ckpt = str(ckpts[idx0])
        logger.info("[%s] evaluating idx=%d/%d: %s", stage_name, idx0 + 1, len(ckpts), ckpt)
        try:
            iter_cfg = base_cfg.clone()
            eval_cfg = iter_cfg.TASK_CONFIG.EVAL
            result = _run_eval_once(iter_cfg, eval_cfg, eval_type=eval_type, ckpt_override=ckpt, ckpt_field=ckpt_field)
            spl = _extract_eval_spl(result)
            if spl is None:
                raise RuntimeError(f"cannot extract SPL from eval result: {result}")
            rec = SweepRecord(ckpt=ckpt, spl=float(spl), ok=True)
            records.append(rec)
            evaluated_indices.add(idx0)
            index_to_record[idx0] = rec
            logger.info("[%s] ok ckpt=%s spl=%.6f", stage_name, ckpt, float(spl))
            return True
        except Exception as exc:
            logger.exception("[%s] failed ckpt=%s error=%s", stage_name, ckpt, exc)
            rec = SweepRecord(ckpt=ckpt, spl=float("-inf"), ok=False, error=str(exc))
            records.append(rec)
            evaluated_indices.add(idx0)
            index_to_record[idx0] = rec
            if args.stop_on_error:
                stopped_early = True
                return False
            return True

    for idx0 in stage1_indices:
        if not _eval_index(idx0, stage_name="stage1"):
            break

    stage1_ok = [index_to_record[i] for i in stage1_indices if i in index_to_record and index_to_record[i].ok]
    if not stage1_ok:
        raise RuntimeError("All stage1 checkpoint evaluations failed. No valid SPL result.")

    stage1_best_idx = max(
        (i for i in stage1_indices if i in index_to_record and index_to_record[i].ok),
        key=lambda i: index_to_record[i].spl,
    )
    stage1_best = index_to_record[stage1_best_idx]
    logger.info(
        "[stage1] best idx=%d ckpt=%s spl=%.6f",
        stage1_best_idx + 1,
        stage1_best.ckpt,
        stage1_best.spl,
    )

    stage2_left = max(0, stage1_best_idx - 3)
    stage2_right = min(len(ckpts) - 1, stage1_best_idx + 3)
    stage2_indices = list(range(stage2_left, stage2_right + 1))
    logger.info("[stage2] fine sweep around best indices=%s", stage2_indices)

    if not stopped_early:
        for idx0 in stage2_indices:
            if not _eval_index(idx0, stage_name="stage2"):
                break

    ok_records = [r for r in records if r.ok]
    if not ok_records:
        raise RuntimeError("All checkpoint evaluations failed. No valid SPL result.")

    ranked = sorted(ok_records, key=lambda x: x.spl, reverse=True)
    best = ranked[0]

    logger.info("[sweep] summary begin")
    for rank, rec in enumerate(ranked, start=1):
        logger.info("[sweep] rank=%03d spl=%.6f ckpt=%s", rank, rec.spl, rec.ckpt)
    logger.info("[sweep] BEST ckpt=%s spl=%.6f", best.ckpt, best.spl)
    logger.info("[sweep] summary end")

    if args.save_json:
        save_path = Path(args.save_json)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("logs") / f"eval_ckpts_{stamp}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "eval_type": eval_type,
        "ckpt_field": ckpt_field,
        "sweep_strategy": {"stage1_stride": 5, "stage2_radius": 3},
        "stage1_indices": stage1_indices,
        "stage2_indices": stage2_indices,
        "stage1_best_index_1based": stage1_best_idx + 1,
        "stage1_best_ckpt": stage1_best.ckpt,
        "stage1_best_spl": stage1_best.spl,
        "split": str(base_cfg.TASK_CONFIG.DATASET.SPLIT),
        "ckpt_dir": str(ckpt_dir),
        "pattern": args.pattern,
        "recursive": bool(args.recursive),
        "max_ckpts": int(args.max_ckpts),
        "total_ckpts": len(ckpts),
        "evaluated_ckpts": len(records),
        "ok_count": len(ok_records),
        "fail_count": len(records) - len(ok_records),
        "best": asdict(best),
        "ranking": [asdict(r) for r in ranked],
        "all_records": [asdict(r) for r in records],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"BEST_CKPT={best.ckpt}")
    print(f"BEST_SPL={best.spl:.6f}")
    print(f"RESULT_JSON={save_path}")


if __name__ == "__main__":
    main()
