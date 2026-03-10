import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from DORL.Dataset_env import RLEnv, build_env_from_dorl_pt

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class DORLPPOTrainConfig:
    action_dim: int = 4
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_end: float = 0.001
    entropy_decay_epochs: int = 80
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    value_clip: float = 0.2
    lr_decay_enable: bool = True
    eval_greedy_episodes: int = 64
    train_epochs: int = 50
    episodes_per_epoch: int = 0
    max_steps: int = 200
    save_every: int = 5
    ckpt_dir: str = "media/DORL/ckpt_ppo"
    tb_log_dir: str = "media/DORL/log_ppo"
    stochastic_policy: bool = True
    greedy_prob_start: float = 0.15
    greedy_prob_end: float = 0.02
    greedy_decay_epochs: int = 25
    log_interval: int = 100
    mismatch_reward: float = -10.0
    advantage_norm_eps: float = 1e-8


class DORLPPOTrainer:
    def __init__(
        self,
        env: RLEnv,
        config: Optional[DORLPPOTrainConfig] = None,
        device: Optional[torch.device] = None,
        online_eval_fn: Optional[Callable[[nn.Module, int, torch.device, int], Dict[str, float]]] = None,
    ) -> None:
        self.env = env
        self.cfg = config or DORLPPOTrainConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_eval_fn = online_eval_fn

        state0 = self._state_to_tensor(self.env.reset())
        self.state_dim = int(state0.numel())

        self.actor = Actor(self.state_dim, self.cfg.hidden_dim, self.cfg.action_dim).to(self.device)
        self.critic = ValueNet(self.state_dim, self.cfg.hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_tb_log_dir = os.path.join(self.cfg.tb_log_dir, run_ts)
        os.makedirs(self.run_tb_log_dir, exist_ok=True)
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.run_tb_log_dir)
        else:
            self.writer = None
            tqdm.write(
                "[DORL PPO] tensorboard is not installed; skip TB logging. "
                "Install with: pip install tensorboard"
            )

    def _state_to_tensor(self, state: Any) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            t = state.detach().float().view(-1).cpu()
        else:
            t = torch.as_tensor(state, dtype=torch.float32).view(-1).cpu()
        if hasattr(self, "state_dim") and t.numel() != self.state_dim:
            if t.numel() > self.state_dim:
                t = t[: self.state_dim]
            else:
                pad = torch.zeros(self.state_dim - t.numel(), dtype=torch.float32)
                t = torch.cat([t, pad], dim=0)
        return t

    def _current_greedy_prob(self, epoch_idx: int) -> float:
        if self.cfg.greedy_decay_epochs <= 1:
            return float(np.clip(self.cfg.greedy_prob_end, 0.0, 1.0))
        progress = float(epoch_idx - 1) / float(max(1, self.cfg.greedy_decay_epochs - 1))
        progress = float(np.clip(progress, 0.0, 1.0))
        prob = self.cfg.greedy_prob_start + (self.cfg.greedy_prob_end - self.cfg.greedy_prob_start) * progress
        return float(np.clip(prob, 0.0, 1.0))

    def _try_oracle_action(self) -> Optional[int]:
        try:
            if self.env.step_idx < len(self.env._active_episode):
                return int(self.env._active_episode[self.env.step_idx].action)
        except Exception:
            return None
        return None

    def _sample_action(self, state: torch.Tensor, greedy_prob: float) -> Tuple[int, float]:
        s = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s)
            dist = torch.distributions.Categorical(logits=logits)
            if random.random() < float(greedy_prob):
                oracle_action = self._try_oracle_action()
                if oracle_action is not None:
                    act = int(oracle_action)
                    log_prob = float(dist.log_prob(torch.tensor([act], device=self.device)).item())
                    return act, log_prob
            if self.cfg.stochastic_policy:
                action_t = dist.sample()
            else:
                action_t = torch.argmax(logits, dim=-1)
            action = int(action_t.item())
            log_prob = float(dist.log_prob(action_t).item())
            return action, log_prob

    def _value(self, state: torch.Tensor) -> float:
        with torch.no_grad():
            v = self.critic(state.to(self.device).unsqueeze(0)).squeeze(0).squeeze(0)
            return float(v.item())

    def _compute_gae(
        self,
        rewards: List[float],
        dones: List[float],
        values: List[float],
        last_value: float,
    ) -> Tuple[List[float], List[float]]:
        returns = [0.0 for _ in rewards]
        advantages = [0.0 for _ in rewards]
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            mask = 1.0 - float(dones[t])
            delta = float(rewards[t]) + self.cfg.gamma * next_value * mask - float(values[t])
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages[t] = float(gae)
            returns[t] = float(gae + values[t])
        return returns, advantages

    def _current_entropy_coef(self, epoch_idx: int) -> float:
        if self.cfg.entropy_decay_epochs <= 1:
            return float(max(0.0, self.cfg.entropy_coef_end))
        progress = float(epoch_idx - 1) / float(max(1, self.cfg.entropy_decay_epochs - 1))
        progress = float(np.clip(progress, 0.0, 1.0))
        coef = self.cfg.entropy_coef + (self.cfg.entropy_coef_end - self.cfg.entropy_coef) * progress
        return float(max(0.0, coef))

    def _current_lr_scale(self, epoch_idx: int) -> float:
        if not bool(self.cfg.lr_decay_enable):
            return 1.0
        progress = float(epoch_idx - 1) / float(max(1, self.cfg.train_epochs - 1))
        return float(max(0.1, 1.0 - progress))

    def _set_lrs(self, lr_scale: float) -> None:
        actor_lr = float(self.cfg.actor_lr) * float(lr_scale)
        critic_lr = float(self.cfg.critic_lr) * float(lr_scale)
        for pg in self.actor_opt.param_groups:
            pg["lr"] = actor_lr
        for pg in self.critic_opt.param_groups:
            pg["lr"] = critic_lr

    def _ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        entropy_coef: float,
    ) -> Dict[str, float]:
        if states.size(0) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "early_stop": 0.0}

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + self.cfg.advantage_norm_eps)
        n = states.size(0)
        mb = max(1, int(self.cfg.minibatch_size))
        batch_count = 0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0

        early_stop = 0.0
        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(n, device=self.device)
            epoch_kl_sum = 0.0
            epoch_kl_count = 0
            for start in range(0, n, mb):
                idx = indices[start : start + mb]
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_old_values = old_values[idx]
                b_returns = returns[idx]
                b_advs = advantages[idx]

                logits = self.actor(b_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(b_states).squeeze(-1)
                if self.cfg.value_clip > 0.0:
                    v_clip = b_old_values + torch.clamp(
                        values_pred - b_old_values,
                        -float(self.cfg.value_clip),
                        float(self.cfg.value_clip),
                    )
                    value_loss_unclip = (values_pred - b_returns) ** 2
                    value_loss_clip = (v_clip - b_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclip, value_loss_clip).mean()
                else:
                    value_loss = F.mse_loss(values_pred, b_returns)

                self.actor_opt.zero_grad()
                actor_total_loss = policy_loss - float(entropy_coef) * entropy
                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_total_loss = self.cfg.value_coef * value_loss
                critic_total_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    approx_kl = (b_old_log_probs - new_log_probs).mean()
                policy_loss_sum += float(policy_loss.item())
                value_loss_sum += float(value_loss.item())
                entropy_sum += float(entropy.item())
                kl_sum += float(approx_kl.item())
                epoch_kl_sum += float(approx_kl.item())
                epoch_kl_count += 1
                batch_count += 1

            mean_epoch_kl = epoch_kl_sum / float(max(1, epoch_kl_count))
            if self.cfg.target_kl > 0.0 and mean_epoch_kl > float(self.cfg.target_kl):
                early_stop = 1.0
                break

        denom = float(max(1, batch_count))
        return {
            "policy_loss": policy_loss_sum / denom,
            "value_loss": value_loss_sum / denom,
            "entropy": entropy_sum / denom,
            "approx_kl": kl_sum / denom,
            "early_stop": early_stop,
        }

    def _greedy_episode_success(self) -> float:
        state = self._state_to_tensor(self.env.reset())
        ep_done = False
        ep_last_info: Dict[str, Any] = {}
        for _ in range(self.cfg.max_steps):
            s = state.to(self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.actor(s)
                action = int(torch.argmax(logits, dim=-1).item())
            next_state_raw, _reward, done, info = self.env.step(action=action)
            state = self._state_to_tensor(next_state_raw)
            if done:
                ep_done = True
                ep_last_info = info
                break
        return self._episode_spl(done=ep_done, info=ep_last_info)

    def _episode_spl(self, done: bool, info: Dict[str, Any]) -> float:
        if not done:
            return 0.0
        if not bool(info.get("matched", False)):
            return 0.0
        try:
            return 1.0 if self.env.step_idx >= len(self.env._active_episode) else 0.0
        except Exception:
            return 0.0

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.cfg.ckpt_dir, f"dorl_ppo_epoch_{epoch}.pth")
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.cfg.action_dim,
                "config": asdict(self.cfg),
            },
            path,
        )

    def train(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        global_step = 0
        global_episode = 0
        episodes_per_epoch = self.cfg.episodes_per_epoch if self.cfg.episodes_per_epoch > 0 else self.env.num_episodes
        episodes_per_epoch = max(1, int(episodes_per_epoch))

        for epoch in tqdm(range(1, self.cfg.train_epochs + 1), desc="DORL PPO epochs"):
            greedy_prob = self._current_greedy_prob(epoch)
            entropy_coef = self._current_entropy_coef(epoch)
            lr_scale = self._current_lr_scale(epoch)
            self._set_lrs(lr_scale)
            epoch_reward_sum = 0.0
            epoch_len_sum = 0.0
            epoch_match_sum = 0.0
            epoch_spl_sum = 0.0

            rollout_states: List[torch.Tensor] = []
            rollout_actions: List[int] = []
            rollout_log_probs: List[float] = []
            rollout_values: List[float] = []
            rollout_returns: List[float] = []
            rollout_advs: List[float] = []

            for _ in range(episodes_per_epoch):
                state = self._state_to_tensor(self.env.reset())
                ep_reward = 0.0
                ep_len = 0
                ep_match = 0
                ep_done = False
                ep_last_info: Dict[str, Any] = {}

                ep_states: List[torch.Tensor] = []
                ep_actions: List[int] = []
                ep_log_probs: List[float] = []
                ep_rewards: List[float] = []
                ep_dones: List[float] = []
                ep_values: List[float] = []

                for _step in range(self.cfg.max_steps):
                    value = self._value(state)
                    action, log_prob = self._sample_action(state, greedy_prob=greedy_prob)
                    next_state_raw, reward, done, info = self.env.step(action=action)
                    next_state = self._state_to_tensor(next_state_raw)
                    matched = bool(info.get("matched", False))

                    ep_states.append(state)
                    ep_actions.append(int(action))
                    ep_log_probs.append(float(log_prob))
                    rollout_values.append(float(value))
                    ep_rewards.append(float(reward))
                    ep_dones.append(float(done))
                    ep_values.append(float(value))

                    global_step += 1
                    ep_reward += float(reward)
                    ep_len += 1
                    ep_match += int(matched)
                    state = next_state

                    if self.writer:
                        self.writer.add_scalar("dorl_ppo/step_reward", float(reward), global_step)
                        self.writer.add_scalar("dorl_ppo/step_done", float(done), global_step)
                        self.writer.add_scalar("dorl_ppo/step_matched", float(matched), global_step)

                    if done:
                        ep_done = True
                        ep_last_info = info
                        break

                last_value = 0.0 if ep_done else self._value(state)
                ep_returns, ep_advs = self._compute_gae(
                    rewards=ep_rewards,
                    dones=ep_dones,
                    values=ep_values,
                    last_value=last_value,
                )

                rollout_states.extend(ep_states)
                rollout_actions.extend(ep_actions)
                rollout_log_probs.extend(ep_log_probs)
                rollout_returns.extend(ep_returns)
                rollout_advs.extend(ep_advs)

                epoch_reward_sum += ep_reward
                epoch_len_sum += float(ep_len)
                if ep_len > 0:
                    epoch_match_sum += float(ep_match) / float(ep_len)
                episode_spl = self._episode_spl(done=ep_done, info=ep_last_info)
                epoch_spl_sum += episode_spl
                global_episode += 1
                if self.writer:
                    self.writer.add_scalar("dorl_ppo/episode_reward", float(ep_reward), global_episode)
                    self.writer.add_scalar("dorl_ppo/episode_len", float(ep_len), global_episode)
                    self.writer.add_scalar("dorl_ppo/episode_match_rate", float(ep_match) / float(max(ep_len, 1)), global_episode)
                    self.writer.add_scalar("dorl_ppo/episode_spl", float(episode_spl), global_episode)

            states_t = torch.stack(rollout_states, dim=0).to(self.device) if rollout_states else torch.empty(0, self.state_dim, device=self.device)
            actions_t = torch.as_tensor(rollout_actions, dtype=torch.long, device=self.device) if rollout_actions else torch.empty(0, dtype=torch.long, device=self.device)
            old_log_probs_t = torch.as_tensor(rollout_log_probs, dtype=torch.float32, device=self.device) if rollout_log_probs else torch.empty(0, device=self.device)
            old_values_t = torch.as_tensor(rollout_values, dtype=torch.float32, device=self.device) if rollout_values else torch.empty(0, device=self.device)
            returns_t = torch.as_tensor(rollout_returns, dtype=torch.float32, device=self.device) if rollout_returns else torch.empty(0, device=self.device)
            advs_t = torch.as_tensor(rollout_advs, dtype=torch.float32, device=self.device) if rollout_advs else torch.empty(0, device=self.device)

            metrics = self._ppo_update(
                states=states_t,
                actions=actions_t,
                old_log_probs=old_log_probs_t,
                old_values=old_values_t,
                returns=returns_t,
                advantages=advs_t,
                entropy_coef=entropy_coef,
            )
            online_eval_spl = float("nan")
            online_eval_reward = float("nan")
            online_eval_distance = float("nan")
            if self.online_eval_fn is not None:
                actor_prev_mode = self.actor.training
                self.actor.eval()
                with torch.no_grad():
                    eval_stats = self.online_eval_fn(self.actor, self.state_dim, self.device, epoch)
                if actor_prev_mode:
                    self.actor.train()
                online_eval_spl = float(eval_stats.get("spl", float("nan")))
                online_eval_reward = float(eval_stats.get("reward", float("nan")))
                online_eval_distance = float(eval_stats.get("distance", float("nan")))
            eval_episodes = max(1, int(self.cfg.eval_greedy_episodes))
            eval_episodes = min(eval_episodes, episodes_per_epoch)
            greedy_eval_spl_sum = 0.0
            for _ in range(eval_episodes):
                greedy_eval_spl_sum += self._greedy_episode_success()

            stats = {
                "epoch": float(epoch),
                "greedy_prob": float(greedy_prob),
                "entropy_coef": float(entropy_coef),
                "lr_scale": float(lr_scale),
                "reward_avg": float(epoch_reward_sum / episodes_per_epoch),
                "len_avg": float(epoch_len_sum / episodes_per_epoch),
                "match_rate_avg": float(epoch_match_sum / episodes_per_epoch),
                "spl_avg": float(epoch_spl_sum / episodes_per_epoch),
                "spl_greedy_eval_avg": float(greedy_eval_spl_sum / float(eval_episodes)),
                "rollout_steps": float(len(rollout_states)),
                "policy_loss": float(metrics["policy_loss"]),
                "value_loss": float(metrics["value_loss"]),
                "entropy": float(metrics["entropy"]),
                "approx_kl": float(metrics["approx_kl"]),
                "early_stop": float(metrics["early_stop"]),
                "online_eval_spl": float(online_eval_spl),
                "online_eval_reward": float(online_eval_reward),
                "online_eval_distance": float(online_eval_distance),
                "global_step": float(global_step),
            }
            history.append(stats)

            if self.writer:
                self.writer.add_scalar("dorl_ppo/epoch_reward_avg", stats["reward_avg"], epoch)
                self.writer.add_scalar("dorl_ppo/epoch_len_avg", stats["len_avg"], epoch)
                self.writer.add_scalar("dorl_ppo/epoch_match_rate_avg", stats["match_rate_avg"], epoch)
                self.writer.add_scalar("dorl_ppo/epoch_spl_avg", stats["spl_avg"], epoch)
                self.writer.add_scalar("dorl_ppo/epoch_spl_greedy_eval_avg", stats["spl_greedy_eval_avg"], epoch)
                self.writer.add_scalar("dorl_ppo/epoch_rollout_steps", stats["rollout_steps"], epoch)
                self.writer.add_scalar("dorl_ppo/policy_loss", stats["policy_loss"], epoch)
                self.writer.add_scalar("dorl_ppo/value_loss", stats["value_loss"], epoch)
                self.writer.add_scalar("dorl_ppo/entropy", stats["entropy"], epoch)
                self.writer.add_scalar("dorl_ppo/approx_kl", stats["approx_kl"], epoch)
                self.writer.add_scalar("dorl_ppo/greedy_prob", stats["greedy_prob"], epoch)
                self.writer.add_scalar("dorl_ppo/entropy_coef", stats["entropy_coef"], epoch)
                self.writer.add_scalar("dorl_ppo/lr_scale", stats["lr_scale"], epoch)
                self.writer.add_scalar("dorl_ppo/early_stop", stats["early_stop"], epoch)
                if not np.isnan(stats["online_eval_spl"]):
                    self.writer.add_scalar("dorl_ppo/online_eval_spl", stats["online_eval_spl"], epoch)
                if not np.isnan(stats["online_eval_reward"]):
                    self.writer.add_scalar("dorl_ppo/online_eval_reward", stats["online_eval_reward"], epoch)
                if not np.isnan(stats["online_eval_distance"]):
                    self.writer.add_scalar("dorl_ppo/online_eval_distance", stats["online_eval_distance"], epoch)

            if self.cfg.log_interval > 0 and epoch % self.cfg.log_interval == 0:
                tqdm.write(
                    "epoch={epoch} reward_avg={reward:.4f} len_avg={length:.2f} match_rate={match:.4f} "
                    "spl_avg={spl:.4f} spl_greedy={spl_g:.4f} steps={steps} ploss={pl:.4f} vloss={vl:.4f} "
                    "ent={ent:.4f} ent_coef={ent_c:.5f} kl={kl:.6f} early_stop={es}".format(
                        epoch=int(stats["epoch"]),
                        reward=stats["reward_avg"],
                        length=stats["len_avg"],
                        match=stats["match_rate_avg"],
                        spl=stats["spl_avg"],
                        spl_g=stats["spl_greedy_eval_avg"],
                        steps=int(stats["rollout_steps"]),
                        pl=stats["policy_loss"],
                        vl=stats["value_loss"],
                        ent=stats["entropy"],
                        ent_c=stats["entropy_coef"],
                        kl=stats["approx_kl"],
                        es=int(stats["early_stop"]),
                    )
                )
                if not np.isnan(stats["online_eval_spl"]):
                    tqdm.write(
                        "epoch={epoch} online_eval reward={reward:.4f} spl={spl:.4f} distance={dist:.4f}".format(
                            epoch=int(stats["epoch"]),
                            reward=stats["online_eval_reward"],
                            spl=stats["online_eval_spl"],
                            dist=stats["online_eval_distance"],
                        )
                    )

            if self.cfg.save_every > 0 and epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(self.cfg.train_epochs)
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return history


def run_dorl_ppo_train(
    dorl_pt_root: str,
    mismatch_reward: float = -10.0,
    config: Optional[DORLPPOTrainConfig] = None,
    online_eval_fn: Optional[Callable[[nn.Module, int, torch.device, int], Dict[str, float]]] = None,
) -> List[Dict[str, float]]:
    env = build_env_from_dorl_pt(
        root=dorl_pt_root,
        random_episode=True,
        mismatch_reward=float(mismatch_reward),
    )
    trainer = DORLPPOTrainer(env=env, config=config, online_eval_fn=online_eval_fn)
    return trainer.train()
