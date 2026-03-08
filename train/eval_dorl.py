import os
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from DORL.train_rl import Actor
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


def _fit_state_dim(state: torch.Tensor, state_dim: int) -> torch.Tensor:
    t = state.detach().float().view(-1).cpu()
    if t.numel() > state_dim:
        return t[:state_dim]
    if t.numel() < state_dim:
        return torch.cat([t, torch.zeros(state_dim - t.numel(), dtype=torch.float32)], dim=0)
    return t


def _load_dorl_actor(dorl_model_path: str, device: torch.device):
    if not os.path.exists(dorl_model_path):
        raise FileNotFoundError(f"EVAL.DORL_MODEL_PATH not found: {dorl_model_path}")
    ckpt = torch.load(dorl_model_path, map_location=device)
    if "actor" not in ckpt:
        raise KeyError(f"DORL checkpoint missing key 'actor': {dorl_model_path}")

    state_dim = int(ckpt.get("state_dim"))
    action_dim = int(ckpt.get("action_dim"))
    actor_sd = ckpt["actor"]
    if "net.1.weight" not in actor_sd:
        raise KeyError("DORL actor state_dict missing net.1.weight; unsupported checkpoint layout.")
    hidden_dim = int(actor_sd["net.1.weight"].shape[0])

    actor = Actor(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
    actor.load_state_dict(actor_sd)
    actor.eval()
    return actor, state_dim


def _load_hybrid_model(hybrid_ckpt: str, device: torch.device):
    if not hybrid_ckpt:
        raise ValueError("Hybrid ckpt path is empty. Please set EVAL.DORL_HYBRID_CKPT (or EVAL.HYBRID_CKPT).")
    if not os.path.exists(hybrid_ckpt):
        raise FileNotFoundError(f"Hybrid ckpt not found: {hybrid_ckpt}")
    model = HybirdNetwork().to(device)
    model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    model.eval()
    return model


def run_dorl_online_eval(config, eval_config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_name = str(getattr(eval_config, "DORL_DATASET_SPLIT", "")).strip()
    eval_cfg = _clone_config_with_dataset_split(config, split_name)

    dorl_model_path = str(getattr(eval_config, "DORL_MODEL_PATH", "")).strip()
    hybrid_ckpt = str(getattr(eval_config, "DORL_HYBRID_CKPT", "")).strip()
    if not hybrid_ckpt:
        hybrid_cfg = getattr(eval_config, "HYBRID", None)
        if hybrid_cfg is not None:
            hybrid_ckpt = str(getattr(hybrid_cfg, "CKPT", "")).strip()
    if not hybrid_ckpt:
        hybrid_ckpt = str(getattr(eval_config, "HYBRID_CKPT", "")).strip()

    actor, state_dim = _load_dorl_actor(dorl_model_path=dorl_model_path, device=device)
    hybrid_model = _load_hybrid_model(hybrid_ckpt=hybrid_ckpt, device=device)

    env = Env(eval_cfg)
    sim = getattr(getattr(env, "_env", None), "_sim", None)
    max_steps = int(getattr(eval_config, "MAX_STEPS", 200))
    greedy = bool(getattr(eval_config, "DORL_GREEDY", True))
    log_every = int(getattr(eval_config, "DORL_LOG_EVERY", 1))
    episodes = int(getattr(eval_config, "EPISODES", 0))
    if episodes <= 0:
        episodes = int(getattr(getattr(env, "_env", None), "number_of_episodes", 0))
    episodes = max(1, episodes)

    total_reward = 0.0
    total_spl = 0.0
    total_distance = 0.0
    logger.info(
        "dorl online eval begin: split=%s episodes=%s max_steps=%s greedy=%s dorl_ckpt=%s hybrid_ckpt=%s",
        eval_cfg.TASK_CONFIG.DATASET.SPLIT,
        episodes,
        max_steps,
        greedy,
        dorl_model_path,
        hybrid_ckpt,
    )

    for ep_idx in tqdm(range(episodes), desc="dorl-online-eval"):
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
                trgb = torch.cat([pre_rgb, rgb], dim=2)
                tdepth = torch.cat([pre_depth, depth], dim=2)
                state = hybrid_model.embedding_forward(
                    audio.to(device),
                    trgb.to(device),
                    tdepth.to(device),
                )
                state = _fit_state_dim(state, state_dim).to(device).unsqueeze(0)
                logits = actor(state)
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
                "take action sac model %s ,reward %s , step %s , done %s , is collided %s",
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
                "dorl eval episode=%s/%s reward=%.4f spl=%.4f distance=%.4f steps=%s done=%s",
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
        "dorl online eval result: reward=%.4f spl=%.4f distance=%.4f",
        avg_reward,
        avg_spl,
        avg_distance,
    )
