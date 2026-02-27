import sys
sys.path.append("/home/getuanhui/project/finnal_exp")
import os
import random
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from configs.default import get_config
from env import Env
from network import HybirdNetwork, SAC_Hybird_model
from network import SAC_LSTM_CQL_v1_5
from utils.visualizations import draw_point, plot_top_down_map

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
) -> EvalStats:
    reward_sum = 0.0
    spl_sum = 0.0
    success_sum = 0.0
    steps_sum = 0

    for ep_idx in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        info = {}
        prev_rgb = None
        prev_depth = None
        episode = env._env.current_episode
        scene = episode.scene_id[-15:-4]
        episode_id = episode.episode_id

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
            ep_reward += float(reward)
            ep_steps += 1
            prev_rgb, prev_depth = rgb, depth

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

                # 保存topdown map
                try:
                    top_down_map = plot_top_down_map(info)
                    topdown_path = os.path.join(vis_dir, f"topdown_{ep_steps:03d}.png")
                    Image.fromarray(top_down_map).save(topdown_path)

                    # 保存topdown map信息
                    topdown_info = {
                        "step": ep_steps,
                        "agent_position": info.get("agent_position", []),
                        "agent_rotation": info.get("agent_rotation", []),
                        "distance_to_goal": info.get("distance_to_goal", 0.0),
                        "success": info.get("success", False),
                        "spl": info.get("spl", 0.0)
                    }
                    topdown_info_path = os.path.join(vis_dir, f"topdown_info_{ep_steps:03d}.json")
                    with open(topdown_info_path, 'w') as f:
                        json.dump(topdown_info, f, indent=2)

                except Exception as e:
                    if ep_steps == 1:
                        print(f"[{policy_name}] visualize topdown map failed: {e}")

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
        print(
            f"[{policy_name}] episode {ep_idx + 1}/{episodes} | "
            f"id={episode_id} reward={ep_reward:.4f} spl={spl:.4f} "
            f"success={success:.0f} steps={ep_steps}"
        )

    return EvalStats(
        avg_reward=reward_sum / episodes,
        avg_spl=spl_sum / episodes,
        success_rate=success_sum / episodes,
        avg_steps=steps_sum / episodes,
        episodes=episodes,
    )


def print_summary(name: str, stats: EvalStats) -> None:
    print(
        f"{name:<8} | episodes={stats.episodes} "
        f"avg_reward={stats.avg_reward:.4f} avg_spl={stats.avg_spl:.4f} "
        f"success_rate={stats.success_rate:.4f} avg_steps={stats.avg_steps:.2f}"
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
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hybrid_model = HybirdNetwork().to(device)
    # hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    # hybrid_model.eval()

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    if getattr(offline_cfg, "model", None) != "v1_3":
        print(f"warning: config TRAIN.OFFLINE.model={offline_cfg.model}, but evaluator is fixed to v1_3")
    sac_model = build_v1_3_model(offline_cfg, device=device)
    sac_model.load_state_dict(torch.load(sac_ckpt, map_location=device))
    sac_model.eval()

    # env_hybrid = Env(config)
    env_sac = Env(config)

    total_eps = env_sac._env.number_of_episodes
    run_episodes = episodes if episodes > 0 else total_eps
    run_episodes = min(run_episodes, total_eps)

    print(f"device={device} episodes={run_episodes} max_steps={max_steps} sac_type=v1_3")
    print(f"hybrid_ckpt={hybrid_ckpt}")
    print(f"sac_ckpt={sac_ckpt}")
    print("-" * 100)
    if save_vis:
        print(f"visualization enabled: {os.path.abspath(vis_root)}")

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
    )

    print("-" * 100)
    print_summary("sac", sac_stats)

    return  sac_stats


def build_v1_5_model(config, device: torch.device):
    return SAC_LSTM_CQL_v1_5(
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
):
    reward_sum = 0.0
    spl_sum = 0.0
    success_sum = 0.0
    steps_sum = 0
    seq_len = 5

    for ep_idx in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        info = {}
        prev_rgb = None
        prev_depth = None
        seq_states = deque(maxlen=seq_len)

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
            ep_reward += float(reward)
            ep_steps += 1
            prev_rgb, prev_depth = rgb, depth

        spl = float(info.get("spl", 0.0))
        success = float(info.get("success", 1.0 if spl > 0 else 0.0))
        reward_sum += ep_reward
        spl_sum += spl
        success_sum += success
        steps_sum += ep_steps
        episode_id = getattr(env._env.current_episode, "episode_id", "unknown")
        print(
            f"[{policy_name}] episode {ep_idx + 1}/{episodes} | "
            f"id={episode_id} reward={ep_reward:.4f} spl={spl:.4f} "
            f"success={success:.0f} steps={ep_steps}"
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
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hybrid_model = HybirdNetwork().to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    hybrid_model.eval()

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    if getattr(offline_cfg, "model", None) != "v1_5":
        print(f"warning: config TRAIN.OFFLINE.model={offline_cfg.model}, but evaluator is fixed to v1_5")

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

    print(f"device={device} episodes={run_episodes} max_steps={max_steps} sac_type=v1_5")
    print(f"hybrid_ckpt={hybrid_ckpt}")
    print(f"sac_ckpt={sac_ckpt}")
    if temporal_ckpt:
        print(f"temporal_ckpt={temporal_ckpt}")
    if actor_ckpt:
        print(f"actor_ckpt={actor_ckpt}")
    if critic1_ckpt:
        print(f"critic1_ckpt={critic1_ckpt}")
    if critic2_ckpt:
        print(f"critic2_ckpt={critic2_ckpt}")
    print("-" * 100)

    sac_stats = evaluate_policy_v1_5(
        env=env,
        policy_name="sac_v1_5",
        hybrid_model=hybrid_model,
        sac_model=sac_model,
        episodes=run_episodes,
        max_steps=max_steps,
        device=device,
        deterministic=not stochastic_sac,
    )
    print("-" * 100)
    print_summary("sac_v1_5", sac_stats)
    return sac_stats


def main():
    settings = EVAL_SETTINGS
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
    )


if __name__ == "__main__":
    main()
