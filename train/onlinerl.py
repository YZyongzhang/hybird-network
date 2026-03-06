import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        audio = torch.stack([s[0] for s in states], dim=0)
        trgb = torch.stack([s[1] for s in states], dim=0)
        tdepth = torch.stack([s[2] for s in states], dim=0)
        pose = torch.stack([s[3] for s in states], dim=0)
        next_audio = torch.stack([s[0] for s in next_states], dim=0)
        next_trgb = torch.stack([s[1] for s in next_states], dim=0)
        next_tdepth = torch.stack([s[2] for s in next_states], dim=0)
        next_pose = torch.stack([s[3] for s in next_states], dim=0)

        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        return (audio, trgb, tdepth, pose), actions, rewards, (next_audio, next_trgb, next_tdepth, next_pose), dones

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class OnlineRLTrain:
    def __init__(
        self,
        env,
        agent,
        writer,
        epoch,
        save_dir,
        config,
        device=None,
    ):
        self.env = env
        self.agent = agent
        self.writer = writer
        self.epoch = int(epoch)
        self.save_dir = save_dir
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dim = int(getattr(config, "action_dim", 4))
        self.hidden_dim = int(getattr(config, "hidden_dim", 256))
        self.actor_lr = float(getattr(config, "lr", 3e-4))
        self.critic_lr = float(getattr(config, "critic_lr", self.actor_lr))
        self.alpha_lr = float(getattr(config, "alpha_lr", self.actor_lr))
        self.gamma = float(getattr(config, "gamma", 0.99))
        self.tau = float(getattr(config, "tau", 0.005))
        self.max_steps = int(getattr(config, "MAX_STEPS", 200))
        self.save_every = int(getattr(config, "save_every", 10))
        self.log_interval = int(getattr(config, "log_interval", 100))
        self.batch_size = int(getattr(config, "batch_size", 128))
        self.buffer_size = int(getattr(config, "buffer_size", 50000))
        self.warmup_steps = int(getattr(config, "warmup_steps", 2000))
        self.updates_per_step = int(getattr(config, "updates_per_step", 1))
        self.max_grad_norm = float(getattr(config, "max_grad_norm", 1.0))
        self.log_alpha_min = float(getattr(config, "log_alpha_min", -10.0))
        self.log_alpha_max = float(getattr(config, "log_alpha_max", 2.0))
        self.q_target_min = float(getattr(config, "q_target_min", -100.0))
        self.q_target_max = float(getattr(config, "q_target_max", 100.0))
        env_episode_count = int(getattr(getattr(self.env, "_env", None), "number_of_episodes", 0))
        config_episode_count = int(getattr(config, "EPISODES_PER_EPOCH", 0))
        self.episodes_per_epoch = env_episode_count if env_episode_count > 0 else config_episode_count
        if self.episodes_per_epoch <= 0:
            self.episodes_per_epoch = 1

        greedy_prob_start_default = float(getattr(config, "POLICY_MIX_ORACLE", 0.15))
        self.greedy_prob_start = float(getattr(config, "GREEDY_PROB_START", greedy_prob_start_default))
        self.greedy_prob_end = float(getattr(config, "GREEDY_PROB_END", 0.02))
        self.greedy_decay_epochs = int(
            getattr(config, "GREEDY_DECAY_EPOCHS", max(1, int(self.epoch * 0.5)))
        )
        self.greedy_prob_start = min(1.0, max(0.0, self.greedy_prob_start))
        self.greedy_prob_end = min(1.0, max(0.0, self.greedy_prob_end))
        if self.greedy_prob_end > self.greedy_prob_start:
            self.greedy_prob_end = self.greedy_prob_start

        default_target_entropy = 0.98 * torch.log(torch.tensor(float(self.action_dim))).item()
        self.target_entropy = float(getattr(config, "target_entropy", default_target_entropy))
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.agent.to(self.device)
        self.agent.train()
        self.embedding_dim = int(self.agent.embedding_dim)

        self.q1 = QNet(self.embedding_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.q2 = QNet(self.embedding_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_q1 = QNet(self.embedding_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_q2 = QNet(self.embedding_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        actor_params = (
            list(self.agent.policy_parameters())
            + list(self.agent.backbone_parameters())
            + list(self.agent.pose_parameters())
        )
        self.actor_optimizer = torch.optim.Adam(
            [p for p in actor_params if p.requires_grad],
            lr=self.actor_lr,
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.sim = getattr(getattr(self.env, "_env", None), "_sim", None)

        self.reward_shaping_enable = bool(getattr(config, "REWARD_SHAPING_ENABLE", True))
        self.reward_global_scale = float(getattr(config, "REWARD_GLOBAL_SCALE", 1.0))
        self.reward_collision_penalty = float(getattr(config, "REWARD_COLLISION_PENALTY", -0.5))
        self.reward_escape_collision_bonus = float(
            getattr(config, "REWARD_ESCAPE_COLLISION_BONUS", 1.0)
        )
        self.reward_bad_stop_penalty = float(getattr(config, "REWARD_BAD_STOP_PENALTY", -8.0))
        self.stop_action_id = int(getattr(config, "STOP_ACTION_ID", 0))
        self.stop_success_distance = float(getattr(config, "STOP_SUCCESS_DISTANCE", 0.0))
        reward_min_clip = getattr(config, "REWARD_MIN_CLIP", None)
        reward_max_clip = getattr(config, "REWARD_MAX_CLIP", None)
        self.reward_min_clip = float(reward_min_clip) if reward_min_clip is not None else None
        self.reward_max_clip = float(reward_max_clip) if reward_max_clip is not None else None

    def _alpha(self):
        return self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max).exp()

    def _extract_distance_to_goal(self, info):
        if not isinstance(info, dict):
            return None
        for key in ("distance_to_goal", "geodesic_distance", "euclidian_distance"):
            if key not in info:
                continue
            try:
                value = info[key]
                if isinstance(value, (tuple, list)):
                    value = value[0]
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _is_collided(self, info):
        if isinstance(info, dict):
            for key in ("is_collided", "collision", "collided", "collode"):
                if key in info:
                    value = info[key]
                    if isinstance(value, (tuple, list)):
                        value = value[0]
                    try:
                        return bool(value)
                    except Exception:
                        pass
        if self.sim is not None and hasattr(self.sim, "previous_step_collided"):
            try:
                return bool(self.sim.previous_step_collided)
            except Exception:
                return False
        return False

    def _shape_reward(self, reward, action, info, collided, prev_collided):
        shaped_reward = float(reward) * self.reward_global_scale
        if not self.reward_shaping_enable:
            return shaped_reward

        if collided:
            shaped_reward += self.reward_collision_penalty

        if prev_collided and (not collided):
            shaped_reward += self.reward_escape_collision_bonus

        if int(action) == self.stop_action_id:
            dist = self._extract_distance_to_goal(info)
            if dist is not None and dist > self.stop_success_distance:
                shaped_reward += self.reward_bad_stop_penalty

        if self.reward_min_clip is not None:
            shaped_reward = max(shaped_reward, self.reward_min_clip)
        if self.reward_max_clip is not None:
            shaped_reward = min(shaped_reward, self.reward_max_clip)
        return shaped_reward

    def _obs_to_inputs(self, obs):
        rgb = torch.as_tensor(obs["rgb"], dtype=torch.float32) / 255.0
        depth = torch.as_tensor(obs["depth"], dtype=torch.float32)
        audio = obs["spectrogram"]
        if isinstance(audio, (tuple, list)):
            audio = audio[0]
        audio = torch.as_tensor(audio, dtype=torch.float32)
        return audio, rgb, depth

    def _rotation_to_list(self, rotation):
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

    def _get_pose(self):
        if self.sim is None or not hasattr(self.sim, "get_agent_state"):
            return torch.zeros(7, dtype=torch.float32)
        try:
            state = self.sim.get_agent_state()
            pos = [float(v) for v in state.position]
            rot = self._rotation_to_list(state.rotation)
            return torch.as_tensor(pos + rot, dtype=torch.float32)
        except Exception:
            return torch.zeros(7, dtype=torch.float32)

    def _stack_two_frame(self, pre_rgb, pre_depth, rgb, depth):
        trgb = torch.cat([pre_rgb, rgb], dim=2)
        tdepth = torch.cat([pre_depth, depth], dim=2)
        return trgb, tdepth

    def _encode(self, audio, trgb, tdepth, pose):
        return self.agent.encode(
            audio.to(self.device),
            trgb.to(self.device),
            tdepth.to(self.device),
            pose.to(self.device),
        ).float()

    def _policy(self, embedding):
        logits = self.agent.policy_head(embedding)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs

    def _soft_update(self, online, target):
        for t, s in zip(target.parameters(), online.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

    def _current_greedy_prob(self, epoch_idx):
        if self.greedy_decay_epochs <= 1:
            return self.greedy_prob_end
        progress = float(epoch_idx - 1) / float(max(1, self.greedy_decay_epochs - 1))
        progress = min(1.0, max(0.0, progress))
        prob = self.greedy_prob_start + (self.greedy_prob_end - self.greedy_prob_start) * progress
        return min(1.0, max(0.0, prob))

    def _sample_action(self, state, greedy_prob):
        audio, trgb, tdepth, pose = state

        with torch.no_grad():
            emb = self._encode(
                audio.unsqueeze(0),
                trgb.unsqueeze(0),
                tdepth.unsqueeze(0),
                pose.unsqueeze(0),
            )
            probs, _ = self._policy(emb)

            if random.random() < float(greedy_prob):
                if self.sim is not None and hasattr(self.sim, "compute_oracle_actions"):
                    try:
                        oracle_actions = self.sim.compute_oracle_actions() or []
                        if isinstance(oracle_actions, (list, tuple)) and len(oracle_actions) > 0:
                            return int(oracle_actions[0])
                    except Exception:
                        pass
                action = int(torch.argmax(probs, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())
        return action

    def _update(self, global_step):
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        (audio, trgb, tdepth, pose), actions, rewards, (next_audio, next_trgb, next_tdepth, next_pose), dones = self.replay_buffer.sample(
            self.batch_size
        )
        audio = audio.to(self.device)
        trgb = trgb.to(self.device)
        tdepth = tdepth.to(self.device)
        pose = pose.to(self.device)
        next_audio = next_audio.to(self.device)
        next_trgb = next_trgb.to(self.device)
        next_tdepth = next_tdepth.to(self.device)
        next_pose = next_pose.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        dones = dones.to(self.device).unsqueeze(1)

        emb = self._encode(audio, trgb, tdepth, pose)
        next_emb = self._encode(next_audio, next_trgb, next_tdepth, next_pose)

        with torch.no_grad():
            next_probs, next_log_probs = self._policy(next_emb)
            next_q1 = self.target_q1(next_emb)
            next_q2 = self.target_q2(next_emb)
            next_min_q = torch.min(next_q1, next_q2)
            alpha = self._alpha().detach()
            next_v = torch.sum(next_probs * (next_min_q - alpha * next_log_probs), dim=1, keepdim=True)
            q_target = rewards + self.gamma * (1.0 - dones) * next_v
            q_target = q_target.clamp(self.q_target_min, self.q_target_max)

        # Critic updates should not backprop through the shared encoder graph twice.
        # Use a detached embedding for Q updates; actor update below uses `emb`.
        emb_critic = emb.detach()
        q1_pred_all = self.q1(emb_critic)
        q2_pred_all = self.q2(emb_critic)
        q1_pred = q1_pred_all.gather(1, actions)
        q2_pred = q2_pred_all.gather(1, actions)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        probs, log_probs = self._policy(emb)
        q1_detached = self.q1(emb).detach()
        q2_detached = self.q2(emb).detach()
        min_q = torch.min(q1_detached, q2_detached)
        alpha = self._alpha()
        actor_loss = torch.sum(probs * (alpha.detach() * log_probs - min_q), dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        alpha_loss = (alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(self.log_alpha_min, self.log_alpha_max)

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

    def _save_checkpoint(self, epoch):
        agent_ckpt = os.path.join(self.save_dir, f"onlinerl_epoch_{epoch}.pth")
        torch.save(self.agent.state_dict(), agent_ckpt)
        aux_ckpt = os.path.join(self.save_dir, f"onlinerl_sac_state_epoch_{epoch}.pth")
        torch.save(
            {
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "q1_opt": self.q1_optimizer.state_dict(),
                "q2_opt": self.q2_optimizer.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "alpha_opt": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            aux_ckpt,
        )

    def train(self):
        global_step = 0
        global_episode = 0
        for ep in tqdm(range(1, self.epoch + 1), desc="online epochs"):
            greedy_prob = self._current_greedy_prob(ep)
            epoch_reward_sum = 0.0
            epoch_spl_sum = 0.0
            epoch_distance_sum = 0.0

            for _ in range(self.episodes_per_epoch):
                obs = self.env.reset()
                episode_reward = 0.0
                info = {"spl": 0.0, "distance_to_goal": -1.0}
                pre_rgb = None
                pre_depth = None
                prev_collided = False

                for step in range(self.max_steps):
                    audio, rgb, depth = self._obs_to_inputs(obs)
                    pose = self._get_pose()
                    if pre_rgb is None:
                        pre_rgb = torch.zeros_like(rgb)
                    if pre_depth is None:
                        pre_depth = torch.zeros_like(depth)
                    trgb, tdepth = self._stack_two_frame(pre_rgb, pre_depth, rgb, depth)
                    state = (audio, trgb, tdepth, pose)

                    action = self._sample_action(state, greedy_prob=greedy_prob)
                    next_obs, reward, done, info = self.env.step(action=action)
                    collided = self._is_collided(info)
                    shaped_reward = self._shape_reward(
                        reward=reward,
                        action=action,
                        info=info,
                        collided=collided,
                        prev_collided=prev_collided,
                    )

                    next_audio, next_rgb, next_depth = self._obs_to_inputs(next_obs)
                    next_pose = self._get_pose()
                    next_trgb, next_tdepth = self._stack_two_frame(rgb, depth, next_rgb, next_depth)
                    next_state = (next_audio, next_trgb, next_tdepth, next_pose)

                    self.replay_buffer.push(
                        state=tuple(x.cpu() for x in state),
                        action=int(action),
                        reward=float(shaped_reward),
                        next_state=tuple(x.cpu() for x in next_state),
                        done=float(done),
                    )

                    global_step += 1
                    episode_reward += float(shaped_reward)

                    metrics = None
                    for _ in range(self.updates_per_step):
                        metrics = self._update(global_step)

                    if self.writer:
                        self.writer.add_scalar("online/step_reward_raw", float(reward), global_step)
                        self.writer.add_scalar("online/step_reward_shaped", float(shaped_reward), global_step)
                        self.writer.add_scalar("online/is_collided", float(collided), global_step)

                    if metrics and self.log_interval > 0 and global_step % self.log_interval == 0:
                        tqdm.write(
                            f"step={global_step} reward_raw={reward:.4f} reward_shaped={shaped_reward:.4f} "
                            f"q1={metrics['q1_loss']:.4f} q2={metrics['q2_loss']:.4f} "
                            f"actor={metrics['actor_loss']:.4f} alpha={metrics['alpha']:.4f}"
                        )

                    obs = next_obs
                    pre_rgb = rgb
                    pre_depth = depth
                    prev_collided = collided
                    if done:
                        break

                global_episode += 1
                epoch_reward_sum += episode_reward
                epoch_spl_sum += float(info.get("spl", 0.0))
                epoch_distance_sum += float(info.get("distance_to_goal", -1.0))

                if self.writer:
                    self.writer.add_scalar("online/episode_reward_shaped", episode_reward, global_episode)
                    self.writer.add_scalar("online/spl", float(info.get("spl", 0.0)), global_episode)
                    self.writer.add_scalar(
                        "online/distance_to_goal",
                        float(info.get("distance_to_goal", -1.0)),
                        global_episode,
                    )

            epoch_reward_avg = epoch_reward_sum / float(self.episodes_per_epoch)
            epoch_spl_avg = epoch_spl_sum / float(self.episodes_per_epoch)
            epoch_distance_avg = epoch_distance_sum / float(self.episodes_per_epoch)
            if self.writer:
                self.writer.add_scalar("online/epoch_reward_shaped_avg", epoch_reward_avg, ep)
                self.writer.add_scalar("online/epoch_spl_avg", epoch_spl_avg, ep)
                self.writer.add_scalar("online/epoch_distance_to_goal_avg", epoch_distance_avg, ep)
                self.writer.add_scalar("online/greedy_prob", float(greedy_prob), ep)
            tqdm.write(
                f"epoch={ep} episodes={self.episodes_per_epoch} greedy_prob={greedy_prob:.4f} "
                f"reward_avg={epoch_reward_avg:.4f} spl_avg={epoch_spl_avg:.4f} "
                f"distance_avg={epoch_distance_avg:.4f}"
            )

            if self.save_every > 0 and ep % self.save_every == 0:
                self._save_checkpoint(ep)

        self._save_checkpoint(self.epoch)
