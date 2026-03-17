import sys
sys.path.append("/home/getuanhui/project/finnal_exp")
import os
import random
import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from habitat.utils.visualizations import maps

from configs.default import get_config
from env import Env
from network import HybirdNetwork, SAC_Hybird_model
from network import SAC_LSTM_CQL_v1_5
from utils.visualizations import draw_point, plot_top_down_map
from utils.log import logger, setup_run_logger

EVAL_SETTINGS = {
    "config": "configs/audiogoal.yaml",
    "hybrid_ckpt": "compare/ckpt/hybirdnetwork_RGBD_two_frame/model_epoch_100.pth",
    "sac_ckpt": "media/offline_hybird/7/sac_hybrid_model_230.pth",
    "episodes": 0,
    "max_steps": 200,
    "seed": 0,
    "stochastic_sac": False,
    "save_vis": True,
    "vis_root": "img/eval_vis",
    "save_action_probs": True,  # 新增：是否保存动作概率
    "step_log_interval": 0,  # 0表示关闭step级日志，>0时每N步输出一次
}


@dataclass
class EvalStats:
    avg_reward: float
    avg_spl: float
    success_rate: float
    avg_steps: float
    episodes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensors(obs: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.from_numpy(obs["rgb"]).float() / 255.0
    depth = torch.from_numpy(obs["depth"]).float()
    audio = torch.from_numpy(obs["spectrogram"][0]).float()
    return audio, rgb, depth


def build_two_frame(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    prev_rgb: Optional[torch.Tensor],
    prev_depth: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prev_rgb is None:
        prev_rgb = torch.zeros_like(rgb)
    if prev_depth is None:
        prev_depth = torch.zeros_like(depth)
    return torch.cat([prev_rgb, rgb], dim=2), torch.cat([prev_depth, depth], dim=2)


def build_v1_3_model(config, device: torch.device):
    return SAC_Hybird_model(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        actor_lr=config.lr,
        critic_lr=config.lr,
        alpha_lr=config.lr,
        target_entropy=config.target_entropy,
        tau=config.tau,
        gamma=config.gamma,
        beta=config.beta,
        device=device,
    ).to(device)


def hybrid_action(
    model: HybirdNetwork,
    audio: torch.Tensor,
    trgb: torch.Tensor,
    tdepth: torch.Tensor,
    device: torch.device,
) -> int:
    with torch.no_grad():
        logits = model(audio.to(device), trgb.to(device), tdepth.to(device))
        return int(torch.argmax(logits, dim=1).item())


def sac_action(
    sac_model,
    audio: torch.Tensor,
    trgb: torch.Tensor,
    tdepth: torch.Tensor,
    device: torch.device,
    deterministic: bool,
) -> int:
    with torch.no_grad():
        if deterministic:
            state = sac_model.hybird.embedding_forward(
                audio.to(device), trgb.to(device), tdepth.to(device)
            ).float()
            probs = sac_model.actor(state)
            return int(torch.argmax(probs, dim=1).item())
        return int(sac_model.get_action((audio.to(device), trgb.to(device), tdepth.to(device))))

def sac_action_with_probs(
    sac_model,
    audio: torch.Tensor,
    trgb: torch.Tensor,
    tdepth: torch.Tensor,
    device: torch.device,
    deterministic: bool,
) -> Tuple[int, np.ndarray]:
    """返回动作和动作概率分布"""
    with torch.no_grad():
        state = sac_model.hybird.embedding_forward(
            audio.to(device), trgb.to(device), tdepth.to(device)
        ).float()
        probs = sac_model.actor(state)
        probs_np = probs.cpu().numpy()[0]  # 转换为numpy数组

        if deterministic:
            action = int(torch.argmax(probs, dim=1).item())
        else:
            action_dist = torch.distributions.Categorical(probs)
            action = int(action_dist.sample().item())

        return action, probs_np


def save_observation_data(
    obs: Dict,
    step: int,
    vis_dir: str,
    action: int,
    action_probs: Optional[np.ndarray] = None,
    reward: float = 0.0,
):
    """保存观察数据：rgb, depth, spectrogram, topdown map等"""
    # 保存RGB图像
    rgb_path = os.path.join(vis_dir, f"rgb_{step:03d}.png")
    Image.fromarray(obs["rgb"]).save(rgb_path)

    # 保存Depth图像（归一化到0-255）
    # depth = obs["depth"]
    # depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    # depth_path = os.path.join(vis_dir, f"depth_{step:03d}.png")
    # Image.fromarray(depth_normalized).save(depth_path)

    # 保存Spectrogram
    # if "spectrogram" in obs and len(obs["spectrogram"]) > 0:
    #     spectrogram = obs["spectrogram"][0]  # 第一个是spectrogram
    #     # 归一化并保存
    #     spec_normalized = ((spectrogram - spectrogram.min()) /
    #                       (spectrogram.max() - spectrogram.min() + 1e-8) * 255).astype(np.uint8)
    #     spec_path = os.path.join(vis_dir, f"spectrogram_{step:03d}.png")
    #     Image.fromarray(spec_normalized).save(spec_path)

    # 保存动作信息
    action_info = {
        "step": step,
        "action": int(action),
        "reward": float(reward),
        "action_probs": action_probs.tolist() if action_probs is not None else None
    }

    action_path = os.path.join(vis_dir, f"action_{step:03d}.json")
    with open(action_path, 'w') as f:
        json.dump(action_info, f, indent=2)

    # 如果提供了动作概率，保存概率分布图
    if action_probs is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        actions = list(range(len(action_probs)))
        bars = ax.bar(actions, action_probs, color='skyblue', edgecolor='black')

        # 高亮选择的动作
        bars[action].set_color('red')

        ax.set_xlabel('Action')
        ax.set_ylabel('Probability')
        ax.set_title(f'Action Probabilities (Step {step})')
        ax.set_xticks(actions)
        ax.set_ylim(0, 1.0)

        # 在每个柱子上添加概率值
        for i, (bar, prob) in enumerate(zip(bars, action_probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

        prob_path = os.path.join(vis_dir, f"action_probs_{step:03d}.png")
        plt.tight_layout()
        plt.savefig(prob_path, dpi=150)
        plt.close(fig)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _get_collided_flag(env, info: Dict[str, Any]) -> bool:
    if isinstance(info, dict) and "collided" in info:
        try:
            return bool(info.get("collided", False))
        except Exception:
            return False

    sim = getattr(env, "_sim", None)
    if sim is None:
        sim = getattr(getattr(env, "_env", None), "_sim", None)
    if sim is not None and hasattr(sim, "previous_step_collided"):
        try:
            return bool(sim.previous_step_collided)
        except Exception:
            return False
    return False


def _max_consecutive_true(flags: List[bool]) -> int:
    best = 0
    run = 0
    for flag in flags:
        if flag:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _extract_trace_flags(
    trace: Dict[str, Any],
    collision_run_threshold: int = 3,
    goal_eps: float = 0.0,
    stop_distance_threshold: float = 3.0,
) -> Dict[str, bool]:
    steps = list(trace.get("steps", []))
    collisions = [bool(step.get("collided", False)) for step in steps]
    distances = [_safe_float(step.get("distance"), float("nan")) for step in steps]
    actions = [int(step.get("action", -1)) for step in steps]
    dones = [bool(step.get("done", False)) for step in steps]

    found_target = bool(trace.get("success", False))
    if not found_target:
        for dist in distances:
            if np.isfinite(dist) and dist <= goal_eps:
                found_target = True
                break

    collision_not_found = bool(
        (_max_consecutive_true(collisions) >= collision_run_threshold) and (not found_target)
    )
    early_stop = False
    stop_action_within_3 = False
    zero_distance_without_stop_mid = False
    for idx, (action, dist, done) in enumerate(zip(actions, distances, dones)):
        if not np.isfinite(dist):
            continue
        if action == 0 and (not done) and dist > stop_distance_threshold:
            early_stop = True
        if action == 0 and goal_eps < dist <= stop_distance_threshold:
            stop_action_within_3 = True
        if idx < len(actions) - 1 and dist <= goal_eps and action != 0:
            zero_distance_without_stop_mid = True

    return {
        "success": bool(trace.get("success", False)),
        "collision_not_found": collision_not_found,
        "early_stop": early_stop,
        "stop_action_within_3": stop_action_within_3,
        "zero_distance_without_stop_mid": zero_distance_without_stop_mid,
    }


def _trace_bucket_name(trace: Dict[str, Any]) -> str:
    flags = _extract_trace_flags(trace)
    if flags["success"]:
        return "success"
    for key in [
        "collision_not_found",
        "early_stop",
        "stop_action_within_3",
        "zero_distance_without_stop_mid",
    ]:
        if flags[key]:
            return key
    return "other_failure"


def _summarize_bucket_distribution(
    episode_traces: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    total = max(1, len(episode_traces))
    counts = {
        "success": 0,
        "collision_not_found": 0,
        "early_stop": 0,
        "stop_action_within_3": 0,
        "zero_distance_without_stop_mid": 0,
        "other": 0,
    }
    for trace in episode_traces:
        bucket = _trace_bucket_name(trace)
        if bucket == "other_failure":
            bucket = "other"
        if bucket not in counts:
            bucket = "other"
        counts[bucket] += 1
    return {
        key: {
            "count": int(val),
            "ratio": float(val / total),
        }
        for key, val in counts.items()
    }


def _dated_policy_root(vis_root: str, policy_name: str) -> str:
    date_tag = datetime.now().strftime("%Y%m%d")
    root = os.path.join(vis_root, date_tag, policy_name)
    os.makedirs(root, exist_ok=True)
    return root


def _save_episode_topdown_group(
    topdown_series: List[Dict[str, Any]],
    base_root: str,
    bucket: str,
    episode_dir_name: str,
    summary: Dict[str, Any],
) -> Optional[str]:
    if not topdown_series:
        return None
    save_dir = os.path.join(base_root, bucket, episode_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    for item in topdown_series:
        step = int(item["step"])
        Image.fromarray(item["map"]).save(os.path.join(save_dir, f"topdown_{step:03d}.png"))
        with open(os.path.join(save_dir, f"topdown_info_{step:03d}.json"), "w") as f:
            json.dump(item.get("info", {}), f, indent=2)
    with open(os.path.join(save_dir, "episode_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return save_dir


def _overlay_sound_sources_on_map(
    env,
    info: Dict[str, Any],
    top_down_map: np.ndarray,
    goal_positions: List[Any],
) -> np.ndarray:
    if top_down_map is None or not goal_positions:
        return top_down_map
    if not isinstance(info, dict) or "top_down_map" not in info:
        return top_down_map

    sim = getattr(env, "_sim", None)
    if sim is None:
        sim = getattr(getattr(env, "_env", None), "_sim", None)
    if sim is None:
        return top_down_map

    try:
        raw_map = info["top_down_map"]["map"]
        raw_h, raw_w = int(raw_map.shape[0]), int(raw_map.shape[1])
    except Exception:
        raw_h, raw_w = int(top_down_map.shape[0]), int(top_down_map.shape[1])
    rotated = raw_h > raw_w
    out = top_down_map.copy()
    marker_radius = max(2, min(out.shape[0], out.shape[1]) // 120)

    for pos in goal_positions:
        try:
            gx, gy = maps.to_grid(float(pos[2]), float(pos[0]), (raw_h, raw_w), sim=sim)
            if rotated:
                gx, gy = raw_w - 1 - int(gy), int(gx)
            else:
                gx, gy = int(gx), int(gy)
            x0 = max(0, gx - marker_radius)
            x1 = min(out.shape[0], gx + marker_radius + 1)
            y0 = max(0, gy - marker_radius)
            y1 = min(out.shape[1], gy + marker_radius + 1)
            out[x0:x1, y0:y1] = (255, 0, 0)
        except Exception:
            continue
    return out


def _summarize_online_distribution(
    episode_traces: List[Dict[str, Any]],
    collision_run_threshold: int = 3,
    goal_eps: float = 0.0,
    stop_distance_threshold: float = 3.0,
) -> Dict[str, Dict[str, float]]:
    total = max(1, len(episode_traces))
    counts = {
        "collision_not_found": 0,
        "early_stop": 0,
        "stop_action_within_3": 0,
        "zero_distance_without_stop_mid": 0,
    }

    for trace in episode_traces:
        steps = list(trace.get("steps", []))
        if not steps:
            continue
        flags = _extract_trace_flags(
            trace,
            collision_run_threshold=collision_run_threshold,
            goal_eps=goal_eps,
            stop_distance_threshold=stop_distance_threshold,
        )
        if flags["collision_not_found"]:
            counts["collision_not_found"] += 1
        if flags["early_stop"]:
            counts["early_stop"] += 1
        if flags["stop_action_within_3"]:
            counts["stop_action_within_3"] += 1
        if flags["zero_distance_without_stop_mid"]:
            counts["zero_distance_without_stop_mid"] += 1

    return {
        name: {
            "count": int(count),
            "ratio": float(count / total),
        }
        for name, count in counts.items()
    }


def _save_online_distribution_outputs(
    distribution: Dict[str, Dict[str, float]],
    bucket_distribution: Dict[str, Dict[str, float]],
    save_root: str,
    policy_name: str,
) -> Tuple[str, str]:
    os.makedirs(save_root, exist_ok=True)

    ordered = [
        ("success", "Success"),
        ("collision_not_found", "Collision No Target"),
        ("early_stop", "Early Stop"),
        ("stop_action_within_3", "Stop Within 3m"),
        ("zero_distance_without_stop_mid", "Distance=0 No Stop"),
        ("other", "Other"),
    ]
    labels = [text for key, text in ordered]
    counts = [int(bucket_distribution[key]["count"]) for key, _ in ordered]
    total_count = max(1, sum(counts))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total_count / 100.0))})",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9)
    ax.set_title(f"Online Eval Distribution ({policy_name})")
    ax.axis("equal")
    plt.tight_layout()

    fig_path = os.path.join(save_root, f"{policy_name}_online_eval_distribution.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    json_path = os.path.join(save_root, f"{policy_name}_online_eval_distribution.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "exclusive_bucket_distribution": bucket_distribution,
                "multi_label_failure_distribution": distribution,
            },
            f,
            indent=2,
        )
    return fig_path, json_path

def evaluate_policy(
    env,
    policy_name: str,
    sac_model,
    episodes: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool,
    save_vis: bool = False,
    vis_root: str = "img/eval_vis",
    save_action_probs: bool = True,  # 新增：是否保存动作概率
    step_log_interval: int = 0,
) -> EvalStats:
    reward_sum = 0.0
    spl_sum = 0.0
    success_sum = 0.0
    steps_sum = 0
    episode_traces: List[Dict[str, Any]] = []
    base_topdown_root = _dated_policy_root(vis_root, policy_name)

    for ep_idx in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        info = {}
        prev_rgb = None
        prev_depth = None
        episode_steps: List[Dict[str, Any]] = []
        topdown_series: List[Dict[str, Any]] = []
        episode = env._env.current_episode
        scene = episode.scene_id[-15:-4]
        episode_id = episode.episode_id
        goal_positions = [goal.position for goal in getattr(episode, "goals", [])]

        vis_dir = ""
        episode_data = []  # 保存整个episode的数据
        if save_vis:
            vis_dir = os.path.join(vis_root, policy_name, f"{scene}_{episode_id}")
            os.makedirs(vis_dir, exist_ok=True)
            # 保存episode信息
            episode_info = {
                "scene": scene,
                "episode_id": episode_id,
                "start_position": episode.start_position.tolist() if hasattr(episode.start_position, 'tolist') else list(episode.start_position),
                "goals": [goal.position.tolist() if hasattr(goal.position, 'tolist') else list(goal.position) for goal in episode.goals]
            }
            with open(os.path.join(vis_dir, "episode_info.json"), 'w') as f:
                json.dump(episode_info, f, indent=2)
        logger.info(
            "[%s] start episode %d/%d | id=%s scene=%s",
            policy_name,
            ep_idx + 1,
            episodes,
            episode_id,
            scene,
        )

        while (not done) and ep_steps < max_steps:
            audio, rgb, depth = to_tensors(obs)
            trgb, tdepth = build_two_frame(rgb, depth, prev_rgb, prev_depth)

            action_probs = None
            if save_action_probs:
                action, action_probs = sac_action_with_probs(
                    sac_model=sac_model,
                    audio=audio,
                    trgb=trgb,
                    tdepth=tdepth,
                    device=device,
                    deterministic=deterministic,
                )
            else:
                action = sac_action(
                    sac_model=sac_model,
                    audio=audio,
                    trgb=trgb,
                    tdepth=tdepth,
                    device=device,
                    deterministic=deterministic,
                )


            obs, reward, done, info = env.step(action=action)
            distance = _safe_float(info.get("distance_to_goal", -1.0), -1.0)
            is_collided = _get_collided_flag(env, info)
            logger.info(
                "take action sac model %s ,reward %s , step %s , done %s , is collided %s , distance %s",
                int(action),
                float(reward),
                ep_steps,
                bool(done),
                is_collided,
                distance,
            )
            ep_reward += float(reward)
            ep_steps += 1
            prev_rgb, prev_depth = rgb, depth
            if step_log_interval > 0 and ep_steps % step_log_interval == 0:
                logger.info(
                    "[%s] episode %d step=%d action=%d reward=%.4f distance=%.4f done=%s",
                    policy_name,
                    ep_idx + 1,
                    ep_steps,
                    int(action),
                    float(reward),
                    float(info.get("distance_to_goal", 0.0)),
                    bool(done),
                )
            episode_steps.append(
                {
                    "step": ep_steps,
                    "action": int(action),
                    "reward": float(reward),
                    "distance": distance,
                    "done": bool(done),
                    "collided": bool(is_collided),
                }
            )

            # 保存观察数据
            if save_vis:
                save_observation_data(
                    obs=obs,
                    step=ep_steps,
                    vis_dir=vis_dir,
                    action=action,
                    action_probs=action_probs,
                    reward=reward
                )

            # 保存topdown map（总是记录，episode结束后按分类落盘）
            try:
                top_down_map = plot_top_down_map(info)
                top_down_map = _overlay_sound_sources_on_map(
                    env=env,
                    info=info,
                    top_down_map=top_down_map,
                    goal_positions=goal_positions,
                )
                topdown_info = {
                    "step": ep_steps,
                    "agent_position": info.get("agent_position", []),
                    "agent_rotation": info.get("agent_rotation", []),
                    "distance_to_goal": info.get("distance_to_goal", 0.0),
                    "success": info.get("success", False),
                    "spl": info.get("spl", 0.0),
                }
                topdown_series.append(
                    {
                        "step": ep_steps,
                        "map": top_down_map,
                        "info": topdown_info,
                    }
                )
                if save_vis:
                    topdown_path = os.path.join(vis_dir, f"topdown_{ep_steps:03d}.png")
                    Image.fromarray(top_down_map).save(topdown_path)
                    with open(os.path.join(vis_dir, f"topdown_info_{ep_steps:03d}.json"), "w") as f:
                        json.dump(topdown_info, f, indent=2)
            except Exception as e:
                if ep_steps == 1:
                    logger.warning("[%s] visualize topdown map failed: %s", policy_name, e)

            # 保存step数据到episode_data
            step_data = {
                "step": ep_steps,
                "action": int(action),
                "reward": float(reward),
                "action_probs": action_probs.tolist() if action_probs is not None else None,
                "agent_position": info.get("agent_position", []),
                "distance_to_goal": info.get("distance_to_goal", 0.0)
            }
            episode_data.append(step_data)

        spl = float(info.get("spl", 0.0))
        success = float(info.get("success", 1.0 if spl > 0 else 0.0))

        reward_sum += ep_reward
        spl_sum += spl
        success_sum += success
        steps_sum += ep_steps
        episode_traces.append(
            {
                "episode_idx": ep_idx + 1,
                "episode_id": episode_id,
                "scene": scene,
                "success": bool(success),
                "steps": episode_steps,
            }
        )
        bucket = _trace_bucket_name(episode_traces[-1])
        topdown_summary = {
            "episode_id": episode_id,
            "scene": scene,
            "bucket": bucket,
            "total_reward": float(ep_reward),
            "spl": float(spl),
            "success": bool(success),
            "total_steps": int(ep_steps),
        }
        episode_dir_name = f"{scene}_{episode_id}"
        topdown_dir = _save_episode_topdown_group(
            topdown_series=topdown_series,
            base_root=base_topdown_root,
            bucket=bucket,
            episode_dir_name=episode_dir_name,
            summary=topdown_summary,
        )
        if topdown_dir:
            logger.info(
                "[%s] topdown saved: bucket=%s dir=%s",
                policy_name,
                bucket,
                os.path.abspath(topdown_dir),
            )

        # 保存整个episode的数据摘要
        if save_vis:
            episode_summary = {
                "episode_id": episode_id,
                "scene": scene,
                "total_reward": float(ep_reward),
                "spl": float(spl),
                "success": bool(success),
                "total_steps": ep_steps,
                "steps_data": episode_data
            }
            summary_path = os.path.join(vis_dir, "episode_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(episode_summary, f, indent=2)

        episode_id = getattr(env._env.current_episode, "episode_id", "unknown")
        logger.info(
            "[%s] episode %d/%d | id=%s reward=%.4f spl=%.4f success=%.0f steps=%d",
            policy_name,
            ep_idx + 1,
            episodes,
            episode_id,
            ep_reward,
            spl,
            success,
            ep_steps,
        )

    distribution = _summarize_online_distribution(episode_traces)
    bucket_distribution = _summarize_bucket_distribution(episode_traces)
    distribution_dir = base_topdown_root
    fig_path, json_path = _save_online_distribution_outputs(
        distribution=distribution,
        bucket_distribution=bucket_distribution,
        save_root=distribution_dir,
        policy_name=policy_name,
    )
    logger.info(
        "[%s] online eval distribution saved: fig=%s json=%s",
        policy_name,
        os.path.abspath(fig_path),
        os.path.abspath(json_path),
    )
    for key, value in distribution.items():
        logger.info(
            "[%s] distribution %-30s count=%d ratio=%.4f",
            policy_name,
            key,
            int(value["count"]),
            float(value["ratio"]),
        )
    for key, value in bucket_distribution.items():
        logger.info(
            "[%s] pie bucket %-30s count=%d ratio=%.4f",
            policy_name,
            key,
            int(value["count"]),
            float(value["ratio"]),
        )

    return EvalStats(
        avg_reward=reward_sum / episodes,
        avg_spl=spl_sum / episodes,
        success_rate=success_sum / episodes,
        avg_steps=steps_sum / episodes,
        episodes=episodes,
    )


def print_summary(name: str, stats: EvalStats) -> None:
    logger.info(
        "%-8s | episodes=%d avg_reward=%.4f avg_spl=%.4f success_rate=%.4f avg_steps=%.2f",
        name,
        stats.episodes,
        stats.avg_reward,
        stats.avg_spl,
        stats.success_rate,
        stats.avg_steps,
    )


def run_v1_3_eval(
    config,
    hybrid_ckpt: str,
    sac_ckpt: str,
    episodes: int = 0,
    max_steps: int = 200,
    seed: int = 0,
    stochastic_sac: bool = False,
    save_vis: bool = False,
    vis_root: str = "img/eval_vis",
    save_action_probs: bool = True,  # 新增：是否保存动作概率
    step_log_interval: int = 0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hybrid_model = HybirdNetwork().to(device)
    # hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    # hybrid_model.eval()

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    if getattr(offline_cfg, "model", None) != "v1_3":
        logger.warning(
            "config TRAIN.OFFLINE.model=%s, but evaluator is fixed to v1_3",
            offline_cfg.model,
        )
    sac_model = build_v1_3_model(offline_cfg, device=device)
    sac_model.load_state_dict(torch.load(sac_ckpt, map_location=device))
    sac_model.eval()

    # env_hybrid = Env(config)
    env_sac = Env(config)

    total_eps = env_sac._env.number_of_episodes
    run_episodes = episodes if episodes > 0 else total_eps
    run_episodes = min(run_episodes, total_eps)

    logger.info("device=%s episodes=%d max_steps=%d sac_type=v1_3", device, run_episodes, max_steps)
    logger.info("hybrid_ckpt=%s", hybrid_ckpt)
    logger.info("sac_ckpt=%s", sac_ckpt)
    logger.info("%s", "-" * 100)
    if save_vis:
        logger.info("visualization enabled: %s", os.path.abspath(vis_root))

    # hybrid_stats = evaluate_policy(
    #     env=env_hybrid,
    #     policy_name="hybrid",
    #     hybrid_model=hybrid_model,
    #     sac_model=None,
    #     episodes=run_episodes,
    #     max_steps=max_steps,
    #     device=device,
    #     deterministic=True,
    #     save_vis=save_vis,
    #     vis_root=vis_root,
    #     save_action_probs=False,  # hybrid模型没有动作概率
    # )

    sac_stats = evaluate_policy(
        env=env_sac,
        policy_name="sac",
        sac_model=sac_model,
        episodes=run_episodes,
        max_steps=max_steps,
        device=device,
        deterministic=not stochastic_sac,
        save_vis=save_vis,
        vis_root=vis_root,
        save_action_probs=save_action_probs,
        step_log_interval=step_log_interval,
    )

    logger.info("%s", "-" * 100)
    print_summary("sac", sac_stats)

    return  sac_stats


def build_v1_5_model(config, device: torch.device):
    return SAC_LSTM_CQL_v1_5(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        target_entropy=config.target_entropy,
        tau=config.tau,
        gamma=config.gamma,
        beta=config.beta,
        device=device,
        lstm_hidden_dim=int(getattr(config, "lstm_hidden_dim", config.state_dim)),
        lstm_num_layers=int(getattr(config, "lstm_num_layers", 1)),
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


def evaluate_policy_v1_5(
    env,
    policy_name: str,
    hybrid_model,
    sac_model,
    episodes: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool,
    step_log_interval: int = 0,
    vis_root: str = "img/eval_vis",
):
    reward_sum = 0.0
    spl_sum = 0.0
    success_sum = 0.0
    steps_sum = 0
    seq_len = 5
    episode_traces: List[Dict[str, Any]] = []
    base_topdown_root = _dated_policy_root(vis_root, policy_name)

    for ep_idx in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        info = {}
        prev_rgb = None
        prev_depth = None
        episode_steps: List[Dict[str, Any]] = []
        topdown_series: List[Dict[str, Any]] = []
        episode = env._env.current_episode
        scene = episode.scene_id[-15:-4] if hasattr(episode, "scene_id") else "unknown_scene"
        goal_positions = [goal.position for goal in getattr(episode, "goals", [])]
        seq_states = deque(maxlen=seq_len)
        episode_id = getattr(env._env.current_episode, "episode_id", "unknown")
        logger.info(
            "[%s] start episode %d/%d | id=%s",
            policy_name,
            ep_idx + 1,
            episodes,
            episode_id,
        )

        while (not done) and ep_steps < max_steps:
            audio, rgb, depth = to_tensors(obs)
            trgb, tdepth = build_two_frame(rgb, depth, prev_rgb, prev_depth)
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
            distance = _safe_float(info.get("distance_to_goal", -1.0), -1.0)
            is_collided = _get_collided_flag(env, info)
            logger.info(
                "take action sac model %s ,reward %s , step %s , done %s , is collided %s , distance %s",
                int(action),
                float(reward),
                ep_steps,
                bool(done),
                is_collided,
                distance,
            )
            ep_reward += float(reward)
            ep_steps += 1
            prev_rgb, prev_depth = rgb, depth
            if step_log_interval > 0 and ep_steps % step_log_interval == 0:
                logger.info(
                    "[%s] episode %d step=%d action=%d reward=%.4f distance=%.4f done=%s",
                    policy_name,
                    ep_idx + 1,
                    ep_steps,
                    int(action),
                    float(reward),
                    float(info.get("distance_to_goal", 0.0)),
                    bool(done),
                )
            episode_steps.append(
                {
                    "step": ep_steps,
                    "action": int(action),
                    "reward": float(reward),
                    "distance": distance,
                    "done": bool(done),
                    "collided": bool(is_collided),
                }
            )
            try:
                top_down_map = plot_top_down_map(info)
                top_down_map = _overlay_sound_sources_on_map(
                    env=env,
                    info=info,
                    top_down_map=top_down_map,
                    goal_positions=goal_positions,
                )
                topdown_series.append(
                    {
                        "step": ep_steps,
                        "map": top_down_map,
                        "info": {
                            "step": ep_steps,
                            "agent_position": info.get("agent_position", []),
                            "agent_rotation": info.get("agent_rotation", []),
                            "distance_to_goal": info.get("distance_to_goal", 0.0),
                            "success": info.get("success", False),
                            "spl": info.get("spl", 0.0),
                        },
                    }
                )
            except Exception as e:
                if ep_steps == 1:
                    logger.warning("[%s] visualize topdown map failed: %s", policy_name, e)

        spl = float(info.get("spl", 0.0))
        success = float(info.get("success", 1.0 if spl > 0 else 0.0))
        reward_sum += ep_reward
        spl_sum += spl
        success_sum += success
        steps_sum += ep_steps
        episode_traces.append(
            {
                "episode_idx": ep_idx + 1,
                "episode_id": episode_id,
                "scene": scene,
                "success": bool(success),
                "steps": episode_steps,
            }
        )
        bucket = _trace_bucket_name(episode_traces[-1])
        topdown_dir = _save_episode_topdown_group(
            topdown_series=topdown_series,
            base_root=base_topdown_root,
            bucket=bucket,
            episode_dir_name=f"{scene}_{episode_id}",
            summary={
                "episode_id": episode_id,
                "scene": scene,
                "bucket": bucket,
                "total_reward": float(ep_reward),
                "spl": float(spl),
                "success": bool(success),
                "total_steps": int(ep_steps),
            },
        )
        if topdown_dir:
            logger.info(
                "[%s] topdown saved: bucket=%s dir=%s",
                policy_name,
                bucket,
                os.path.abspath(topdown_dir),
            )
        episode_id = getattr(env._env.current_episode, "episode_id", "unknown")
        logger.info(
            "[%s] episode %d/%d | id=%s reward=%.4f spl=%.4f success=%.0f steps=%d",
            policy_name,
            ep_idx + 1,
            episodes,
            episode_id,
            ep_reward,
            spl,
            success,
            ep_steps,
        )

    distribution = _summarize_online_distribution(episode_traces)
    bucket_distribution = _summarize_bucket_distribution(episode_traces)
    distribution_dir = base_topdown_root
    fig_path, json_path = _save_online_distribution_outputs(
        distribution=distribution,
        bucket_distribution=bucket_distribution,
        save_root=distribution_dir,
        policy_name=policy_name,
    )
    logger.info(
        "[%s] online eval distribution saved: fig=%s json=%s",
        policy_name,
        os.path.abspath(fig_path),
        os.path.abspath(json_path),
    )
    for key, value in distribution.items():
        logger.info(
            "[%s] distribution %-30s count=%d ratio=%.4f",
            policy_name,
            key,
            int(value["count"]),
            float(value["ratio"]),
        )
    for key, value in bucket_distribution.items():
        logger.info(
            "[%s] pie bucket %-30s count=%d ratio=%.4f",
            policy_name,
            key,
            int(value["count"]),
            float(value["ratio"]),
        )

    return EvalStats(
        avg_reward=reward_sum / episodes,
        avg_spl=spl_sum / episodes,
        success_rate=success_sum / episodes,
        avg_steps=steps_sum / episodes,
        episodes=episodes,
    )


def run_v1_5_eval(
    config,
    hybrid_ckpt: str,
    sac_ckpt: str,
    episodes: int = 0,
    max_steps: int = 200,
    seed: int = 0,
    stochastic_sac: bool = False,
    temporal_ckpt: str = "",
    actor_ckpt: str = "",
    critic1_ckpt: str = "",
    critic2_ckpt: str = "",
    step_log_interval: int = 0,
    vis_root: str = "img/eval_vis",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hybrid_model = HybirdNetwork().to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    hybrid_model.eval()

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    if getattr(offline_cfg, "model", None) != "v1_5":
        logger.warning(
            "config TRAIN.OFFLINE.model=%s, but evaluator is fixed to v1_5",
            offline_cfg.model,
        )

    sac_model = build_v1_5_model(offline_cfg, device=device)
    if sac_ckpt:
        sac_model.load_state_dict(torch.load(sac_ckpt, map_location=device), strict=False)
    _load_optional_subweights(
        sac_model,
        temporal_ckpt=temporal_ckpt,
        actor_ckpt=actor_ckpt,
        critic1_ckpt=critic1_ckpt,
        critic2_ckpt=critic2_ckpt,
    )
    sac_model.eval()

    env = Env(config)
    total_eps = env._env.number_of_episodes
    run_episodes = episodes if episodes > 0 else total_eps
    run_episodes = min(run_episodes, total_eps)

    logger.info("device=%s episodes=%d max_steps=%d sac_type=v1_5", device, run_episodes, max_steps)
    logger.info("hybrid_ckpt=%s", hybrid_ckpt)
    logger.info("sac_ckpt=%s", sac_ckpt)
    if temporal_ckpt:
        logger.info("temporal_ckpt=%s", temporal_ckpt)
    if actor_ckpt:
        logger.info("actor_ckpt=%s", actor_ckpt)
    if critic1_ckpt:
        logger.info("critic1_ckpt=%s", critic1_ckpt)
    if critic2_ckpt:
        logger.info("critic2_ckpt=%s", critic2_ckpt)
    logger.info("%s", "-" * 100)

    sac_stats = evaluate_policy_v1_5(
        env=env,
        policy_name="sac_v1_5",
        hybrid_model=hybrid_model,
        sac_model=sac_model,
        episodes=run_episodes,
        max_steps=max_steps,
        device=device,
        deterministic=not stochastic_sac,
        step_log_interval=step_log_interval,
        vis_root=vis_root,
    )
    logger.info("%s", "-" * 100)
    print_summary("sac_v1_5", sac_stats)
    return sac_stats


def main():
    settings = EVAL_SETTINGS
    setup_run_logger(base_dir="logs", run_name="eval_hybrid_sac")
    config = get_config(config_paths=settings["config"])

    if not os.path.exists(settings["hybrid_ckpt"]):
        raise FileNotFoundError(f"Hybrid ckpt not found: {settings['hybrid_ckpt']}")
    if not os.path.exists(settings["sac_ckpt"]):
        raise FileNotFoundError(f"SAC ckpt not found: {settings['sac_ckpt']}")

    run_v1_3_eval(
        config=config,
        hybrid_ckpt=settings["hybrid_ckpt"],
        sac_ckpt=settings["sac_ckpt"],
        episodes=settings["episodes"],
        max_steps=settings["max_steps"],
        seed=settings["seed"],
        stochastic_sac=settings["stochastic_sac"],
        save_vis=settings["save_vis"],
        vis_root=settings["vis_root"],
        save_action_probs=settings["save_action_probs"],
        step_log_interval=settings["step_log_interval"],
    )


if __name__ == "__main__":
    main()
