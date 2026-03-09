import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: float) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states, dim=0),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.stack(next_states, dim=0),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


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


class QNet(nn.Module):
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


@dataclass
class DORLTrainConfig:
    action_dim: int = 4
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 1e-4
    target_entropy: Optional[float] = None
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 200000
    warmup_steps: int = 2000
    updates_per_step: int = 1
    max_grad_norm: float = 1.0
    q_target_min: float = -100.0
    q_target_max: float = 100.0
    log_alpha_min: float = -10.0
    log_alpha_max: float = 2.0
    train_epochs: int = 50
    episodes_per_epoch: int = 0
    max_steps: int = 200
    save_every: int = 5
    ckpt_dir: str = "media/DORL/ckpt"
    stochastic_policy: bool = True
    greedy_prob_start: float = 0.15
    greedy_prob_end: float = 0.02
    greedy_decay_epochs: int = 25
    tb_log_dir: str = "media/DORL/log"
    log_interval: int = 100
    # Cancel pre-applied reward global scale in PT by dividing reward with this factor.
    reward_unscale_enable: bool = False
    reward_scale_factor: float = 1.0
    mismatch_reward: float = -10.0


class DORLSACTrainer:
    def __init__(
        self,
        env: RLEnv,
        config: Optional[DORLTrainConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.env = env
        self.cfg = config or DORLTrainConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state0 = self._state_to_tensor(self.env.reset())
        self.state_dim = int(state0.numel())

        self.actor = Actor(self.state_dim, self.cfg.hidden_dim, self.cfg.action_dim).to(self.device)
        self.q1 = QNet(self.state_dim, self.cfg.hidden_dim, self.cfg.action_dim).to(self.device)
        self.q2 = QNet(self.state_dim, self.cfg.hidden_dim, self.cfg.action_dim).to(self.device)
        self.target_q1 = QNet(self.state_dim, self.cfg.hidden_dim, self.cfg.action_dim).to(self.device)
        self.target_q2 = QNet(self.state_dim, self.cfg.hidden_dim, self.cfg.action_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.cfg.critic_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)
        if self.cfg.target_entropy is None:
            self.target_entropy = 0.98 * np.log(float(self.cfg.action_dim))
        else:
            self.target_entropy = float(self.cfg.target_entropy)

        self.replay = ReplayBuffer(self.cfg.buffer_size)
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_tb_log_dir = os.path.join(self.cfg.tb_log_dir, run_ts)
        os.makedirs(self.run_tb_log_dir, exist_ok=True)
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.run_tb_log_dir)
        else:
            self.writer = None
            tqdm.write(
                "[DORL] tensorboard is not installed; skip TB logging. "
                "Install with: pip install tensorboard"
            )

    def _alpha(self) -> torch.Tensor:
        return self.log_alpha.clamp(self.cfg.log_alpha_min, self.cfg.log_alpha_max).exp()

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

    def _policy(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs

    def _current_greedy_prob(self, epoch_idx: int) -> float:
        if self.cfg.greedy_decay_epochs <= 1:
            return float(np.clip(self.cfg.greedy_prob_end, 0.0, 1.0))
        progress = float(epoch_idx - 1) / float(max(1, self.cfg.greedy_decay_epochs - 1))
        progress = float(np.clip(progress, 0.0, 1.0))
        prob = self.cfg.greedy_prob_start + (self.cfg.greedy_prob_end - self.cfg.greedy_prob_start) * progress
        return float(np.clip(prob, 0.0, 1.0))

    def _try_oracle_action(self) -> Optional[int]:
        # For dataset env, oracle action is the recorded action at current transition.
        try:
            if self.env.step_idx < len(self.env._active_episode):
                return int(self.env._active_episode[self.env.step_idx].action)
        except Exception:
            return None
        return None

    def _sample_action(self, state: torch.Tensor, greedy_prob: float) -> int:
        s = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self._policy(s)
            if random.random() < float(greedy_prob):
                oracle_action = self._try_oracle_action()
                if oracle_action is not None:
                    return int(oracle_action)
                return int(torch.argmax(probs, dim=-1).item())
            if not self.cfg.stochastic_policy:
                return int(torch.argmax(probs, dim=-1).item())
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    # def _compensate_reward_scale(self, reward: float) -> float:
    #     if not bool(self.cfg.reward_unscale_enable):
    #         return float(reward)
    #     scale = float(self.cfg.reward_scale_factor)
    #     if abs(scale) < 1e-8:
    #         return float(reward)
    #     return float(reward) / scale

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            p_dst.data.copy_(p_dst.data * (1.0 - self.cfg.tau) + p_src.data * self.cfg.tau)

    def _update(self, global_step:int) -> Optional[Dict[str , float]]: # old online trainer which codex write alse
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)


        with torch.no_grad():
            next_probs, next_log_probs = self._policy(next_states)
            next_q1 = self.target_q1(next_states)
            next_q2 = self.target_q2(next_states)
            next_min_q = torch.min(next_q1, next_q2)
            alpha = self._alpha().detach()
            next_v = torch.sum(next_probs * (next_min_q - alpha * next_log_probs), dim=1, keepdim=True)
            q_target = rewards + self.cfg.gamma * (1.0 - dones) * next_v
            q_target = q_target.clamp(self.cfg.q_target_min, self.cfg.q_target_max)

        # Critic updates should not backprop through the shared encoder graph twice.
        # Use a detached embedding for Q updates; actor update below uses `emb`.
        emb_critic = states.detach()
        q1_pred_all = self.q1(emb_critic)
        q2_pred_all = self.q2(emb_critic)
        q1_pred = q1_pred_all.gather(1, actions)
        q2_pred = q2_pred_all.gather(1, actions)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        
        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.max_grad_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.max_grad_norm)
        self.q2_opt.step()

        probs, log_probs = self._policy(states)
        q1_detached = self.q1(states).detach()
        q2_detached = self.q2(states).detach()
        min_q = torch.min(q1_detached, q2_detached)
        alpha = self._alpha()
        actor_loss = torch.sum(probs * (alpha.detach() * log_probs - min_q), dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        alpha_loss = (alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(self.cfg.log_alpha_min, self.cfg.log_alpha_max)

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        if self.writer:
            self.writer.add_scalar("online/sac_q1_loss", float(q1_loss.item()), global_step)
            self.writer.add_scalar("online/sac_q2_loss", float(q2_loss.item()), global_step)
            self.writer.add_scalar("online/sac_actor_loss", float(actor_loss.item()), global_step)
            self.writer.add_scalar("online/sac_alpha_loss", float(alpha_loss.item()), global_step)
            self.writer.add_scalar("online/sac_alpha", float(self._alpha().item()), global_step)
            self.writer.add_scalar("online/sac_entropy", float(entropy.item()), global_step)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self._alpha().item()),
            "entropy": float(entropy.item()),
        }
    def _episode_spl(self, done: bool, info: Dict[str, Any]) -> float:
        """
        DORL 定义:
        - 只有在“动作持续匹配并到达该 episode 末尾”时 SPL=1
        - 其他情况 SPL=0
        """
        if not done:
            return 0.0
        if not bool(info.get("matched", False)):
            return 0.0
        try:
            return 1.0 if self.env.step_idx >= len(self.env._active_episode) else 0.0
        except Exception:
            return 0.0

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.cfg.ckpt_dir, f"dorl_sac_epoch_{epoch}.pth")
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "q1_opt": self.q1_opt.state_dict(),
                "q2_opt": self.q2_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
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

        for epoch in tqdm(range(1, self.cfg.train_epochs + 1), desc="DORL epochs"):
            greedy_prob = self._current_greedy_prob(epoch)
            epoch_reward_sum = 0.0
            epoch_len_sum = 0.0
            epoch_match_sum = 0.0
            epoch_spl_sum = 0.0
            epoch_updates = 0.0
            
            for _ in range(episodes_per_epoch):
                state = self._state_to_tensor(self.env.reset())
                ep_reward = 0.0
                ep_len = 0
                ep_match = 0
                ep_done = False
                ep_last_info: Dict[str, Any] = {}

                for _step in range(self.cfg.max_steps):
                    # 测试通过 ， 满足预想按照数据集进行step
                    action = self._sample_action(state, greedy_prob=greedy_prob)
                    next_state_raw, reward, done, info = self.env.step(action=action) # 满足走错路子返回done true。reward -1 。 nextstateraw 
                    next_state = self._state_to_tensor(next_state_raw)
                    matched = bool(info.get("matched", False))
                    # 测试通过
                    self.replay.push(
                        state=state,
                        action=int(action),
                        reward=float(reward),
                        next_state=next_state,
                        done=float(done),
                    )

                    global_step += 1
                    ep_reward += float(reward)
                    ep_len += 1
                    ep_match += int(matched)
                    state = next_state

                    metrics = None
                    for _ in range(self.cfg.updates_per_step):
                        metrics = self._update(global_step=global_step)
                        if metrics is not None:
                            epoch_updates += 1.0

                    if self.writer:
                        self.writer.add_scalar("dorl/step_reward", float(reward), global_step)
                        self.writer.add_scalar("dorl/step_done", float(done), global_step)
                        self.writer.add_scalar("dorl/step_matched", float(matched), global_step)
                    if metrics and self.cfg.log_interval > 0 and global_step % self.cfg.log_interval == 0:
                        tqdm.write(
                            f"step={global_step} reward={reward:.4f} done={int(done)} matched={int(matched)} "
                            f"q1={metrics['q1_loss']:.4f} q2={metrics['q2_loss']:.4f} "
                            f"actor={metrics['actor_loss']:.4f} alpha={metrics['alpha']:.4f}"
                        )

                    if done:
                        ep_done = True
                        ep_last_info = info
                        break

                epoch_reward_sum += ep_reward
                epoch_len_sum += float(ep_len)
                if ep_len > 0:
                    epoch_match_sum += float(ep_match) / float(ep_len)
                episode_spl = self._episode_spl(done=ep_done, info=ep_last_info)
                epoch_spl_sum += episode_spl
                global_episode += 1
                if self.writer:
                    self.writer.add_scalar("dorl/episode_reward", float(ep_reward), global_episode)
                    self.writer.add_scalar("dorl/episode_len", float(ep_len), global_episode)
                    self.writer.add_scalar("dorl/episode_match_rate", float(ep_match) / float(max(ep_len, 1)), global_episode)
                    self.writer.add_scalar("dorl/episode_spl", float(episode_spl), global_episode)

            stats = {
                "epoch": float(epoch),
                "greedy_prob": float(greedy_prob),
                "reward_avg": float(epoch_reward_sum / episodes_per_epoch),
                "len_avg": float(epoch_len_sum / episodes_per_epoch),
                "match_rate_avg": float(epoch_match_sum / episodes_per_epoch),
                "spl_avg": float(epoch_spl_sum / episodes_per_epoch),
                "buffer_size": float(len(self.replay)),
                "updates": float(epoch_updates),
                "global_step": float(global_step),
            }
            history.append(stats)
            if self.writer:
                self.writer.add_scalar("dorl/epoch_reward_avg", stats["reward_avg"], epoch)
                self.writer.add_scalar("dorl/epoch_len_avg", stats["len_avg"], epoch)
                self.writer.add_scalar("dorl/epoch_match_rate_avg", stats["match_rate_avg"], epoch)
                self.writer.add_scalar("dorl/epoch_spl_avg", stats["spl_avg"], epoch)
                self.writer.add_scalar("dorl/epoch_updates", stats["updates"], epoch)
                self.writer.add_scalar("dorl/greedy_prob", stats["greedy_prob"], epoch)
            tqdm.write(
                "epoch={epoch} greedy_prob={greedy:.4f} reward_avg={reward:.4f} len_avg={length:.2f} "
                "match_rate={match:.4f} spl_avg={spl:.4f} buffer={buf} updates={upd}".format(
                    epoch=int(stats["epoch"]),
                    greedy=stats["greedy_prob"],
                    reward=stats["reward_avg"],
                    length=stats["len_avg"],
                    match=stats["match_rate_avg"],
                    spl=stats["spl_avg"],
                    buf=int(stats["buffer_size"]),
                    upd=int(stats["updates"]),
                )
            )

            if self.cfg.save_every > 0 and epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(self.cfg.train_epochs)
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return history


def run_dorl_train(
    dorl_pt_root,
    mismatch_reward: float = -10.0,
    config: Optional[DORLTrainConfig] = None,
) -> List[Dict[str, float]]:
    env = build_env_from_dorl_pt(
        root=dorl_pt_root,
        random_episode=True,
        mismatch_reward=float(mismatch_reward),
    )
    trainer = DORLSACTrainer(env=env, config=config)
    return trainer.train()
