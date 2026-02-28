#!/usr/bin/env python3
"""
Generate synthetic but diverse OfflineRL shards compatible with this project.

Output shard keys:
- states:      [N, T, D] float32
- next_states: [N, T, D] float32
- actions:     [N, T]    int64 in [0, action_dim)
- rewards:     [N, T]    float32
- dones:       [N, T]    bool
"""

import argparse
import os
from dataclasses import dataclass

import torch


@dataclass
class PolicyCfg:
    name: str
    action_noise: float
    step_scale: float
    terminal_prob: float
    collision_prob: float
    reward_bias: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diverse OfflineRL training shards")
    parser.add_argument("--out", type=str, default="collect/diverse_offline_rl")
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--samples-per-shard", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_policies() -> list[PolicyCfg]:
    return [
        PolicyCfg("greedy", action_noise=0.03, step_scale=1.10, terminal_prob=0.18, collision_prob=0.05, reward_bias=0.30),
        PolicyCfg("near_greedy", action_noise=0.12, step_scale=1.00, terminal_prob=0.12, collision_prob=0.10, reward_bias=0.10),
        PolicyCfg("explore", action_noise=0.30, step_scale=0.95, terminal_prob=0.08, collision_prob=0.18, reward_bias=-0.02),
        PolicyCfg("recovery", action_noise=0.25, step_scale=0.85, terminal_prob=0.10, collision_prob=0.28, reward_bias=-0.05),
        PolicyCfg("random", action_noise=0.75, step_scale=0.80, terminal_prob=0.06, collision_prob=0.25, reward_bias=-0.10),
    ]


def action_transition_vectors(action_dim: int, state_dim: int, g: torch.Generator) -> torch.Tensor:
    base = torch.randn(action_dim, state_dim, generator=g) * 0.03
    # Separate action semantics in first few dims so policy differences are learnable.
    for a in range(action_dim):
        idx = a % min(action_dim, state_dim)
        base[a, idx] += 0.35
    return base


def generate_one_shard(
    n: int,
    seq_len: int,
    state_dim: int,
    action_dim: int,
    trans_vec: torch.Tensor,
    policy_mix: torch.Tensor,
    policies: list[PolicyCfg],
    g: torch.Generator,
):
    states = torch.zeros(n, seq_len, state_dim, dtype=torch.float32)
    next_states = torch.zeros(n, seq_len, state_dim, dtype=torch.float32)
    actions = torch.zeros(n, seq_len, dtype=torch.long)
    rewards = torch.zeros(n, seq_len, dtype=torch.float32)
    dones = torch.zeros(n, seq_len, dtype=torch.bool)

    policy_ids = torch.multinomial(policy_mix, num_samples=n, replacement=True, generator=g)

    for i in range(n):
        p = policies[int(policy_ids[i].item())]

        # Latent task direction and initial state.
        goal = torch.randn(state_dim, generator=g)
        goal = goal / (goal.norm() + 1e-8)
        s = torch.randn(state_dim, generator=g) * 0.25

        terminated = False
        for t in range(seq_len):
            states[i, t] = s

            # Greedy action wrt goal projection over action transition vectors.
            q_pref = torch.mv(trans_vec, goal)
            a_greedy = int(torch.argmax(q_pref).item())

            if torch.rand(1, generator=g).item() < p.action_noise:
                a = int(torch.randint(0, action_dim, (1,), generator=g).item())
            else:
                a = a_greedy

            collision = torch.rand(1, generator=g).item() < p.collision_prob
            delta = p.step_scale * trans_vec[a] + torch.randn(state_dim, generator=g) * 0.04
            if collision:
                delta = -0.20 * delta + torch.randn(state_dim, generator=g) * 0.02

            s_next = s + delta
            next_states[i, t] = s_next
            actions[i, t] = a

            # Progress-like reward + behavior bias + penalties/bonuses.
            progress = torch.dot(s_next - s, goal).item()
            reward = 0.8 * progress + p.reward_bias
            if collision:
                reward -= 0.35

            done = False
            if (not terminated) and (torch.rand(1, generator=g).item() < p.terminal_prob):
                done = True
                reward += 1.0 if a == a_greedy else -0.4
                terminated = True

            rewards[i, t] = reward
            dones[i, t] = done
            s = s_next

            if terminated:
                # Keep post-terminal transitions mostly stationary with tiny noise.
                for k in range(t + 1, seq_len):
                    states[i, k] = s
                    jitter = torch.randn(state_dim, generator=g) * 0.005
                    next_states[i, k] = s + jitter
                    actions[i, k] = 0
                    rewards[i, k] = -0.02
                    dones[i, k] = True
                break

    return {
        "states": states,
        "next_states": next_states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)

    policies = build_policies()
    # Bias toward useful-but-diverse data.
    policy_mix = torch.tensor([0.28, 0.24, 0.20, 0.16, 0.12], dtype=torch.float32)
    trans_vec = action_transition_vectors(args.action_dim, args.state_dim, g)

    total = 0
    for sid in range(args.num_shards):
        shard = generate_one_shard(
            n=args.samples_per_shard,
            seq_len=args.seq_len,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            trans_vec=trans_vec,
            policy_mix=policy_mix,
            policies=policies,
            g=g,
        )
        out_path = os.path.join(args.out, f"offline_rl_shard_{sid}.pt")
        torch.save(shard, out_path)
        total += args.samples_per_shard
        print(f"saved {out_path} samples={args.samples_per_shard}")

    print(f"done. shards={args.num_shards} total_samples={total} seq_len={args.seq_len} state_dim={args.state_dim}")


if __name__ == "__main__":
    main()
