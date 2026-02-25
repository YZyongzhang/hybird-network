import argparse
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from configs.default import get_config
from env import Env
from network import HybirdNetwork, SAC_Hybird_model


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


def evaluate_policy(
    env,
    policy_name: str,
    hybrid_model: HybirdNetwork,
    sac_model,
    episodes: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool,
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

        while (not done) and ep_steps < max_steps:
            audio, rgb, depth = to_tensors(obs)
            trgb, tdepth = build_two_frame(rgb, depth, prev_rgb, prev_depth)

            if policy_name == "hybrid":
                action = hybrid_action(hybrid_model, audio, trgb, tdepth, device)
            elif policy_name == "sac":
                action = sac_action(
                    sac_model=sac_model,
                    audio=audio,
                    trgb=trgb,
                    tdepth=tdepth,
                    device=device,
                    deterministic=deterministic,
                )
            else:
                raise ValueError(f"Unsupported policy: {policy_name}")

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
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hybrid_model = HybirdNetwork().to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    hybrid_model.eval()

    offline_cfg = config.TASK_CONFIG.TRAIN.OFFLINE
    if getattr(offline_cfg, "model", None) != "v1_3":
        print(f"warning: config TRAIN.OFFLINE.model={offline_cfg.model}, but evaluator is fixed to v1_3")
    sac_model = build_v1_3_model(offline_cfg, device=device)
    sac_model.load_state_dict(torch.load(sac_ckpt, map_location=device))
    sac_model.eval()

    env_hybrid = Env(config)
    env_sac = Env(config)

    total_eps = env_hybrid._env.number_of_episodes
    run_episodes = episodes if episodes > 0 else total_eps
    run_episodes = min(run_episodes, total_eps)

    print(f"device={device} episodes={run_episodes} max_steps={max_steps} sac_type=v1_3")
    print(f"hybrid_ckpt={hybrid_ckpt}")
    print(f"sac_ckpt={sac_ckpt}")
    print("-" * 100)

    hybrid_stats = evaluate_policy(
        env=env_hybrid,
        policy_name="hybrid",
        hybrid_model=hybrid_model,
        sac_model=None,
        episodes=run_episodes,
        max_steps=max_steps,
        device=device,
        deterministic=True,
    )

    sac_stats = evaluate_policy(
        env=env_sac,
        policy_name="sac",
        hybrid_model=hybrid_model,
        sac_model=sac_model,
        episodes=run_episodes,
        max_steps=max_steps,
        device=device,
        deterministic=not stochastic_sac,
    )

    print("-" * 100)
    print_summary("hybrid", hybrid_stats)
    print_summary("sac", sac_stats)
    print(
        "delta    | "
        f"avg_reward={sac_stats.avg_reward - hybrid_stats.avg_reward:+.4f} "
        f"avg_spl={sac_stats.avg_spl - hybrid_stats.avg_spl:+.4f} "
        f"success_rate={sac_stats.success_rate - hybrid_stats.success_rate:+.4f} "
        f"avg_steps={sac_stats.avg_steps - hybrid_stats.avg_steps:+.2f}"
    )

    try:
        env_hybrid.close()
        env_sac.close()
    except Exception:
        pass

    return hybrid_stats, sac_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid model vs v1_3 Offline SAC model on AudioNav env.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/audiogoal.yaml",
        help="Path to task config yaml.",
    )
    parser.add_argument(
        "--hybrid-ckpt",
        type=str,
        default="compare/ckpt/hybirdnetwork_RGBD_two_frame/model_epoch_100.pth",
        help="Checkpoint path for hybrid model.",
    )
    parser.add_argument(
        "--sac-ckpt",
        type=str,
        required=True,
        help="Checkpoint path for SAC model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="How many episodes to run. 0 means use env default number_of_episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max rollout steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--stochastic-sac",
        action="store_true",
        help="Use stochastic SAC sampling. Default is deterministic action selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(config_paths=args.config)
    run_v1_3_eval(
        config=config,
        hybrid_ckpt=args.hybrid_ckpt,
        sac_ckpt=args.sac_ckpt,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        stochastic_sac=args.stochastic_sac,
    )


if __name__ == "__main__":
    main()
