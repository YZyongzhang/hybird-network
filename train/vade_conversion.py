import os
import pickle
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class LoadLmdb:
    @classmethod
    def _normalize_raw_paths(cls, path):
        if isinstance(path, (str, os.PathLike)):
            paths = [os.fspath(path)]
        else:
            try:
                paths = [os.fspath(p) for p in list(path)]
            except TypeError as exc:
                raise ValueError(
                    "RAW_DATA_PATH must be a directory path or a sequence of directory paths."
                ) from exc

        if not paths:
            raise ValueError("RAW_DATA_PATH is empty.")

        return paths

    @classmethod
    def _flatten_sequence(cls, value):
        if isinstance(value, np.ndarray):
            return value.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            out = []
            for v in value:
                if isinstance(v, (list, tuple, np.ndarray)):
                    out.extend(cls._flatten_sequence(v))
                else:
                    out.append(v)
            return out
        return [value]

    @classmethod
    def _read_transition_arrays(cls, data):
        action_id = cls._flatten_sequence(data.get("action_id", []))
        rewards = cls._flatten_sequence(data.get("reward", []))
        dones = cls._flatten_sequence(data.get("done", []))
        n = min(len(action_id), len(rewards), len(dones))
        return action_id[:n], rewards[:n], dones[:n]

    @classmethod
    def _safe_to_bool(cls, value, default=False):
        if value is None:
            return default
        try:
            return bool(value)
        except Exception:
            return default

    @classmethod
    def _safe_to_int(cls, value, default=None):
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    @classmethod
    def _safe_to_float(cls, value, default=None):
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @classmethod
    def _get_info_list(cls, data, n):
        info = data.get("info", [])
        if isinstance(info, np.ndarray):
            info = info.reshape(-1).tolist()
        elif not isinstance(info, list):
            info = cls._flatten_sequence(info)
        if len(info) < n:
            info = list(info) + [None] * (n - len(info))
        return info[:n]

    @classmethod
    def _infer_collision_mask(cls, data, rewards, info_list, n, config):
        raw = data.get("collision", None)
        if raw is not None:
            vals = cls._flatten_sequence(raw)
            mask = [cls._safe_to_bool(v, default=False) for v in vals[:n]]
            if len(mask) < n:
                mask.extend([False] * (n - len(mask)))
            return mask

        # 从 info 中兜底推断碰撞标记
        mask = []
        for i in range(n):
            item = info_list[i]
            collided = None
            if isinstance(item, dict):
                for key in ("is_collided", "collision", "collided", "collode"):
                    if key in item:
                        collided = cls._safe_to_bool(item.get(key), default=False)
                        break
            if collided is None:
                threshold = float(getattr(config, "COLLISION_REWARD_THRESHOLD", -4.0))
                collided = float(rewards[i]) <= threshold
            mask.append(collided)
        return mask

    @classmethod
    def _extract_distance_to_goal(cls, info_item: Any):
        if not isinstance(info_item, dict):
            return None
        for key in ("distance_to_goal", "geodesic_distance", "euclidian_distance"):
            if key not in info_item:
                continue
            value = info_item[key]
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 0:
                    continue
                value = value[0]
            dist = cls._safe_to_float(value, default=None)
            if dist is not None:
                return dist
        return None

    @classmethod
    def _extract_success(cls, info_item: Any):
        if not isinstance(info_item, dict):
            return None
        if "success" not in info_item:
            return None
        value = info_item["success"]
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            value = value[0]
        return cls._safe_to_bool(value, default=False)

    @classmethod
    def _reshape_rewards(cls, data, action_id, rewards, dones, config):
        n = min(len(action_id), len(rewards), len(dones))
        if n <= 0:
            return rewards
        if not bool(getattr(config, "REWARD_SCALE_ENABLE", False)):
            return rewards

        rewards = [float(v) for v in rewards[:n]]
        action_id = action_id[:n]
        dones = dones[:n]
        info_list = cls._get_info_list(data, n)
        collision_mask = cls._infer_collision_mask(
            data=data,
            rewards=rewards,
            info_list=info_list,
            n=n,
            config=config,
        )

        reward_global_scale = float(getattr(config, "REWARD_GLOBAL_SCALE", 1.0))
        reward_collision_penalty = float(getattr(config, "REWARD_COLLISION_PENALTY", -0.2))
        reward_escape_collision_bonus = float(
            getattr(config, "REWARD_ESCAPE_COLLISION_BONUS", 0.3)
        )
        reward_bad_stop_penalty = float(getattr(config, "REWARD_BAD_STOP_PENALTY", -5.0))
        stop_action_id = int(getattr(config, "STOP_ACTION_ID", 0))
        stop_success_distance = float(getattr(config, "STOP_SUCCESS_DISTANCE", 1.0))
        reward_min_clip = getattr(config, "REWARD_MIN_CLIP", None)
        reward_max_clip = getattr(config, "REWARD_MAX_CLIP", None)
        clip_lo = float(reward_min_clip) if reward_min_clip is not None else -np.inf
        clip_hi = float(reward_max_clip) if reward_max_clip is not None else np.inf

        shaped = []
        for i in range(n):
            r = rewards[i] * reward_global_scale
            collided = collision_mask[i]
            prev_collided = collision_mask[i - 1] if i > 0 else False

            if collided:
                # 加大碰撞惩罚
                r += reward_collision_penalty
            if prev_collided and not collided:
                # 鼓励脱离连续碰撞
                r += reward_escape_collision_bonus

            action = cls._safe_to_int(action_id[i], default=None)
            done = cls._safe_to_bool(dones[i], default=False)
            if done and action == stop_action_id:
                dist = cls._extract_distance_to_goal(info_list[i])
                succ = cls._extract_success(info_list[i])
                bad_stop = False
                if dist is not None:
                    bad_stop = dist > stop_success_distance
                elif succ is not None:
                    bad_stop = not succ
                if bad_stop:
                    r += reward_bad_stop_penalty

            r = float(np.clip(r, clip_lo, clip_hi))
            shaped.append(r)
        return shaped

    @classmethod
    def Generate_DORL_PT(cls, path, model, config):
        """
        生成 DORL 可直接读取的 episode shards。

        每个 episode 结构:
            [
                (state_embed, next_state_embed, reward, done, action),
                ...
            ]

        输出文件:
            dorl_episode_shard_{id}.pt
        每个 pt 保存一个 episode 列表（list[episode]）。
        """
        files = None
        # 支持 RAW_DATA_PATH 为:
        # 1) 单个 pkl 文件（内部保存原始样本文件路径列表）
        # 2) 单个目录
        # 3) 目录列表
        if isinstance(path, (str, os.PathLike)):
            path_str = os.fspath(path)
            if os.path.isfile(path_str):
                with open(path_str, "rb") as f:
                    files = pickle.load(f)
            else:
                files = cls.get_files(path=path_str)
        else:
            paths = cls._normalize_raw_paths(path)
            # 兼容列表里只给了一个 pkl 文件清单路径的场景
            if len(paths) == 1 and os.path.isfile(paths[0]):
                with open(paths[0], "rb") as f:
                    files = pickle.load(f)
            else:
                files = cls.get_files(path=paths)

        if not isinstance(files, list) or len(files) == 0:
            raise ValueError(
                "Generate_DORL_PT failed to resolve input files from RAW_DATA_PATH. "
                f"Got type={type(files).__name__}, size={len(files) if hasattr(files, '__len__') else 'N/A'}."
            )

        random.shuffle(files)
        os.makedirs(config.TO_PATH, exist_ok=True)

        episodes_per_shard = int(getattr(config, "EPISODES_PER_SHARD", 200))
        if episodes_per_shard <= 0:
            episodes_per_shard = 200

        embed_batch_size = int(getattr(config, "EMBED_BATCH_SIZE", 64))
        if embed_batch_size <= 0:
            embed_batch_size = 64

        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        shard_id = 0
        total_episodes = 0
        shard_episodes = []

        def flush_shard():
            nonlocal shard_id, total_episodes
            if not shard_episodes:
                return
            shard_path = os.path.join(config.TO_PATH, f"dorl_episode_shard_{shard_id}.pt")
            torch.save(shard_episodes, shard_path)
            print(f"保存 DORL shard {shard_id}, episodes={len(shard_episodes)}")
            total_episodes += len(shard_episodes)
            shard_id += 1
            shard_episodes.clear()

        for file in tqdm(files, desc="Generate DORL PT"):
            with open(file, "rb") as f:
                data = pickle.load(f)

            obs = data.get("obs", [])
            action_id, rewards, dones = cls._read_transition_arrays(data)
            rewards = cls._reshape_rewards(
                data=data,
                action_id=action_id,
                rewards=rewards,
                dones=dones,
                config=config,
            )
            n = min(len(action_id), len(rewards), len(dones), max(0, len(obs) - 1))
            if n <= 0:
                continue

            # 逐 step 构造 state 输入，按 batch 编码，避免显存峰值过高
            encoded_states = []
            chunk_audio, chunk_trgb, chunk_tdepth = [], [], []
            pre_rgb, pre_depth = None, None

            with torch.no_grad():
                for i, v in enumerate(obs):
                    rgb = torch.from_numpy(v["rgb"]).float() / 255.0
                    depth = torch.from_numpy(v["depth"]).float()
                    audio = torch.from_numpy(v["spectrogram"][0]).float()

                    if i == 0:
                        pre_rgb = torch.zeros_like(rgb)
                        pre_depth = torch.zeros_like(depth)

                    trgb = torch.cat([pre_rgb, rgb], dim=2)
                    tdepth = torch.cat([pre_depth, depth], dim=2)
                    pre_rgb = rgb
                    pre_depth = depth

                    chunk_audio.append(audio)
                    chunk_trgb.append(trgb)
                    chunk_tdepth.append(tdepth)

                    if len(chunk_audio) >= embed_batch_size:
                        emb = model.embedding_forward(
                            torch.stack(chunk_audio).to(device),
                            torch.stack(chunk_trgb).to(device),
                            torch.stack(chunk_tdepth).to(device),
                        )
                        if emb.dim() == 1:
                            emb = emb.unsqueeze(0)
                        emb = emb.detach().cpu()
                        encoded_states.extend([emb[j] for j in range(emb.shape[0])])
                        chunk_audio.clear()
                        chunk_trgb.clear()
                        chunk_tdepth.clear()

                if chunk_audio:
                    emb = model.embedding_forward(
                        torch.stack(chunk_audio).to(device),
                        torch.stack(chunk_trgb).to(device),
                        torch.stack(chunk_tdepth).to(device),
                    )
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)
                    emb = emb.detach().cpu()
                    encoded_states.extend([emb[j] for j in range(emb.shape[0])])

            if len(encoded_states) < n + 1:
                continue

            episode = []
            for i in range(n):
                episode.append(
                    (
                        encoded_states[i],
                        encoded_states[i + 1],
                        float(rewards[i]),
                        bool(dones[i]),
                        int(action_id[i]),
                    )
                )

            shard_episodes.append(episode)
            if len(shard_episodes) >= episodes_per_shard:
                flush_shard()

        flush_shard()
        print(f"DORL PT 写入完成，总 episodes: {total_episodes}, 输出目录: {config.TO_PATH}")
        return {
            "output_dir": config.TO_PATH,
            "total_episodes": total_episodes,
            "num_shards": shard_id,
            "episodes_per_shard": episodes_per_shard,
            "embed_batch_size": embed_batch_size,
        }

    @classmethod
    def get_files(cls, path):
        # 获取 raw path 下各 scene 中的 pkl 文件。
        # 支持单路径和路径列表，便于混合多个 raw dataset 一起生成 PT。
        files = []
        for root in cls._normalize_raw_paths(path):
            if not os.path.isdir(root):
                raise ValueError(f"RAW_DATA_PATH is not a valid directory: {root}")

            object_name_files = [
                os.path.join(root, i)
                for i in os.listdir(path=root)
                if i != "a.md"
            ]

            for scene in object_name_files:
                if os.path.isdir(scene):
                    files.extend(
                        [
                            os.path.join(scene, i)
                            for i in os.listdir(scene)
                            if os.path.isfile(os.path.join(scene, i))
                        ]
                    )
                elif os.path.isfile(scene):
                    files.append(scene)

        if not files:
            raise ValueError(f"No files found from RAW_DATA_PATH={path}")
        return files
    @classmethod
    def load_pt(cls, path, config):
        files = cls.get_files(path=path)
        random.shuffle(files)
        os.makedirs(config.TO_PATH , exist_ok=True)
        shard_size = 10000  # 每个 shard 1w 样本
        shard_id = 0

        buffer_rgb, buffer_depth, buffer_audios, buffer_actions , buffer_angles = [],[], [], [] , []

        for file in tqdm(files):
            tqdm.write(file)
            with open(file, 'rb') as f:
                data = pickle.load(f)
         
            obs = data['obs']
            

            action_id = data['action_id']
            action_id = np.array(action_id).reshape(-1).tolist()
            for v, a  , info in zip(obs[:-1], action_id , data['info']):
                # if info['distance_to_goal'] > 8.0:
                #     tqdm.write(f"distence is {info['distance_to_goal']} , drop")
                #     continue
                rgb = torch.from_numpy(v['rgb']).float() / 255.0
                depth = torch.from_numpy(v['depth']).float()                
                audio = torch.from_numpy(v['spectrogram'][0]).float()
                
                action = torch.tensor(a, dtype=torch.long)
                angel = np.degrees(v['angle'][1])
                buffer_rgb.append(rgb)
                buffer_depth.append(depth)
                buffer_audios.append(audio)
                buffer_actions.append(action)
                buffer_angles.append(torch.tensor(angel))
                # 写一个 shard
                if len(buffer_rgb) >= shard_size:
                    torch.save({
                        'rgb': torch.stack(buffer_rgb),
                        'depth':torch.stack(buffer_depth),
                        'audios': torch.stack(buffer_audios),
                        'actions': torch.stack(buffer_actions),
                        'angles':torch.stack(buffer_angles)
                    }, f"{config.TO_PATH}/foundation_model_shard_{shard_id}.pt")

                    print(f"保存 shard {shard_id}, size={len(buffer_rgb)}")

                    buffer_rgb, buffer_depth , buffer_audios, buffer_actions , buffer_angles = [],[], [], [] , []
                    shard_id += 1

        # 保存最后一个不满 shard 的数据
        if buffer_rgb:
            torch.save({
                'rgb': torch.stack(buffer_rgb),
                'depth':torch.stack(buffer_depth),
                'audios': torch.stack(buffer_audios),
                'actions': torch.stack(buffer_actions),
                'angles':torch.stack(buffer_angles)
            }, f"{config.TO_PATH}/foundation_model_shard_{shard_id}.pt")
            print(f"保存 shard {shard_id}, size={len(buffer_rgb)}")
    
    @classmethod
    def load_two_frame_pt(cls, path, config):
        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.TO_PATH, exist_ok=True)

        shard_size = 10000  # 每个shard包含的样本数
        shard_id = 0

        buffer_rgb, buffer_depth, buffer_audios, buffer_actions , buffer_angles = [],[], [], [] , []

        for file in tqdm(files, desc="Loading offline RL data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist() # 取出reset的时候的done。后续要删除这个地方。更改数据收集策略
    
            for i in range(len(obs) - 1): # 去除最后没有动作的终态
                v_now = obs[i]
                
                if i != 0:
                    v_pre = obs[i - 1]
                a = action_id[i]
                r = rewards[i]
                d = dones[i]

                # 提取特征
                rgb_now = torch.from_numpy(v_now['rgb']).float() / 255.0
                depth_now = torch.from_numpy(v_now['depth']).float() 
                if i == 0:
                    rgb_pre = torch.zeros_like(rgb_now)
                    depth_pre = torch.zeros_like(depth_now)
                else:
                    rgb_pre = torch.from_numpy(v_pre['rgb']).float() / 255.0
                    depth_pre = torch.from_numpy(v_pre['depth']).float()
                audio = torch.from_numpy(v_now['spectrogram'][0]).float()

                # 去除上一步的audio，将两帧img拼接一起
                rgb = torch.cat([rgb_pre , rgb_now] , dim=2)
                depth = torch.cat([depth_pre , depth_now] , dim=2)

                action = torch.tensor(a, dtype=torch.long)
                angel = np.degrees(obs[i]['angle'][1])
                buffer_rgb.append(rgb)
                buffer_depth.append(depth)
                buffer_audios.append(audio)
                buffer_actions.append(action)
                buffer_angles.append(torch.tensor(angel))
                # 写一个 shard
                if len(buffer_rgb) >= shard_size:
                    torch.save({
                        'rgb': torch.stack(buffer_rgb),
                        'depth':torch.stack(buffer_depth),
                        'audios': torch.stack(buffer_audios),
                        'actions': torch.stack(buffer_actions),
                        'angles':torch.stack(buffer_angles)
                    }, f"{config.TO_PATH}/foundation_model_shard_{shard_id}.pt")

                    print(f"保存 shard {shard_id}, size={len(buffer_rgb)}")

                    buffer_rgb, buffer_depth , buffer_audios, buffer_actions , buffer_angles = [],[], [], [] , []
                    shard_id += 1

        # 保存最后一个不满 shard 的数据
        if buffer_rgb:
            torch.save({
                'rgb': torch.stack(buffer_rgb),
                'depth':torch.stack(buffer_depth),
                'audios': torch.stack(buffer_audios),
                'actions': torch.stack(buffer_actions),
                'angles':torch.stack(buffer_angles)
            }, f"{config.TO_PATH}/foundation_model_shard_{shard_id}.pt")
            print(f"保存 shard {shard_id}, size={len(buffer_rgb)}")
    @classmethod
    def load_offline_lstm_level_audio_visual(cls, path, model, config, seq_len=5):
        """
        构建 LSTM 时序输入的 offline 数据。
        每个样本是一个长度为 seq_len 的序列：
            - states: [seq_len, feature_dim]
            - actions: [seq_len]
            - rewards: [seq_len]
            - dones: [seq_len]
        """

        # files = cls.get_files(path=path)
        with open(path , 'rb') as f:
            files = pickle.load(f)
            
        random.shuffle(files)

        os.makedirs(config.TO_PATH, exist_ok=True)

        seq_len = int(getattr(config, "SEQ_LEN", seq_len))
        shard_size = int(getattr(config, "SHARD_SIZE", 2000))
        shard_id = 0

        buffer_audio_states, buffer_visual_audio_states , buffer_next_audio_states ,  buffer_next_visual_audio_states= [], [] , [] , []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading LSTM offline data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id, rewards, dones = cls._read_transition_arrays(data)
            rewards = cls._reshape_rewards(
                data=data,
                action_id=action_id,
                rewards=rewards,
                dones=dones,
                config=config,
            )

            # 编码整个轨迹
            encoded_audio_states = []
            encoded_visual_audio_states = []
            with torch.no_grad():
                for  i , v in enumerate(obs):
                    
                    rgb = torch.from_numpy(v['rgb']).float() / 255.0
                    depth = torch.from_numpy(v['depth']).float()
                    audio = torch.from_numpy(v['spectrogram'][0]).float()
                    if i == 0:
                        pre_rgb = torch.zeros_like(rgb)
                        pre_depth = torch.zeros_like(depth)
                    audio , visual_audio = model.embedding_forward_attention(
                        audio.to(model.device),
                        torch.cat([pre_rgb , rgb] , dim = 2).to(model.device),
                        torch.cat([pre_depth , depth] , dim = 2).to(model.device)
                    )

                    encoded_audio_states.append(audio.squeeze(0).cpu()) # 去除batch
                    encoded_visual_audio_states.append(visual_audio.squeeze(0).cpu())
                    pre_rgb = rgb
                    pre_depth = depth

            # 构造时序样本（滑动窗口）
            traj_len = len(encoded_audio_states)
            for i in range(traj_len - seq_len):
                state_audio_seq = torch.stack(encoded_audio_states[i:i+seq_len])              # 当前状态序列
                state_visual_audio_seq = torch.stack(encoded_visual_audio_states[i:i+seq_len])
                next_state_audio_seq = torch.stack(encoded_audio_states[i+1:i+1+seq_len])    # 下一个状态序列
                next_state_visual_audio_seq = torch.stack(encoded_visual_audio_states[i+1:i+1+seq_len])
                a_seq = torch.tensor(action_id[i:i+seq_len], dtype=torch.long)
                r_seq = torch.tensor(rewards[i:i+seq_len], dtype=torch.float)
                d_seq = torch.tensor(dones[i:i+seq_len], dtype=torch.bool)
                buffer_audio_states.append(state_audio_seq)
                buffer_visual_audio_states.append(state_visual_audio_seq)
                buffer_next_audio_states.append(next_state_audio_seq)
                buffer_next_visual_audio_states.append(next_state_visual_audio_seq)
                buffer_actions.append(a_seq)
                buffer_rewards.append(r_seq)
                buffer_dones.append(d_seq)

                # 存 shard
                if len(buffer_audio_states) >= shard_size:
                    shard_path = os.path.join(config.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
                    torch.save({
                        'states_audio': torch.stack(buffer_audio_states) , 
                        'states_visual_audio': torch.stack(buffer_visual_audio_states),
                        'next_states_audio': torch.stack(buffer_next_audio_states),
                        'next_states_visual_audio': torch.stack(buffer_next_visual_audio_states),
                        'actions': torch.stack(buffer_actions),
                        'rewards': torch.stack(buffer_rewards),
                        'dones': torch.stack(buffer_dones)
                    }, shard_path)
                    print(f"保存 shard {shard_id}, size={len(buffer_audio_states)}")

                    buffer_audio_states, buffer_visual_audio_states , buffer_next_audio_states ,  buffer_next_visual_audio_states= [], [] , [] , []
                    buffer_actions, buffer_rewards, buffer_dones = [], [], []
                    shard_id += 1

        # 保存最后一批
        if buffer_audio_states:
            shard_path = os.path.join(config.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
            torch.save({
                'states_audio': torch.stack(buffer_audio_states) , 
                'states_visual_audio': torch.stack(buffer_visual_audio_states),
                'next_states_audio': torch.stack(buffer_next_audio_states),
                'next_states_visual_audio': torch.stack(buffer_next_visual_audio_states),
                'actions': torch.stack(buffer_actions),
                'rewards': torch.stack(buffer_rewards),
                'dones': torch.stack(buffer_dones)
            }, shard_path)
            print(f"保存 shard {shard_id}, size={len(buffer_audio_states)})")
    @classmethod
    def load_offline_lstm_level(cls, path, model, config, seq_len=5):
        """
        构建 LSTM 时序输入的 offline 数据。
        每个样本是一个长度为 seq_len 的序列：
            - states: [seq_len, feature_dim]
            - actions: [seq_len]
            - rewards: [seq_len]
            - dones: [seq_len]
        """

        # files = cls.get_files(path=path)
        with open(path , 'rb') as f:
            files = pickle.load(f)
            
        random.shuffle(files)

        os.makedirs(config.TO_PATH, exist_ok=True)

        seq_len = int(getattr(config, "SEQ_LEN", seq_len))
        shard_size = int(getattr(config, "SHARD_SIZE", 2000))
        shard_id = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading LSTM offline data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id, rewards, dones = cls._read_transition_arrays(data)
            rewards = cls._reshape_rewards(
                data=data,
                action_id=action_id,
                rewards=rewards,
                dones=dones,
                config=config,
            )

            # 编码整个轨迹
            encoded_states = []
            with torch.no_grad():
                for  i , v in enumerate(obs):
                    
                    rgb = torch.from_numpy(v['rgb']).float() / 255.0
                    depth = torch.from_numpy(v['depth']).float()
                    audio = torch.from_numpy(v['spectrogram'][0]).float()
                    if i == 0:
                        pre_rgb = torch.zeros_like(rgb)
                        pre_depth = torch.zeros_like(depth)
                    state = model.embedding_forward(
                        audio.to(model.device),
                        torch.cat([pre_rgb , rgb] , dim = 2).to(model.device),
                        torch.cat([pre_depth , depth] , dim = 2).to(model.device)
                    )

                    encoded_states.append(state.squeeze(0).cpu()) # 去除batch
                    pre_rgb = rgb
                    pre_depth = depth

            # 构造时序样本（滑动窗口）
            traj_len = len(encoded_states)
            for i in range(traj_len - seq_len):
                state_seq = torch.stack(encoded_states[i:i+seq_len])              # 当前状态序列
                next_state_seq = torch.stack(encoded_states[i+1:i+1+seq_len])    # 下一个状态序列
                a_seq = torch.tensor(action_id[i:i+seq_len], dtype=torch.long)
                r_seq = torch.tensor(rewards[i:i+seq_len], dtype=torch.float)
                d_seq = torch.tensor(dones[i:i+seq_len], dtype=torch.bool)
                buffer_states.append(state_seq)
                buffer_next_states.append(next_state_seq)
                buffer_actions.append(a_seq)
                buffer_rewards.append(r_seq)
                buffer_dones.append(d_seq)

                # 存 shard
                if len(buffer_states) >= shard_size:
                    shard_path = os.path.join(config.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
                    torch.save({
                        'states': torch.stack(buffer_states),
                        'next_states': torch.stack(buffer_next_states),
                        'actions': torch.stack(buffer_actions),
                        'rewards': torch.stack(buffer_rewards),
                        'dones': torch.stack(buffer_dones)
                    }, shard_path)
                    print(f"保存 shard {shard_id}, size={len(buffer_states)}")

                    buffer_states, buffer_next_states = [], []
                    buffer_actions, buffer_rewards, buffer_dones = [], [], []
                    shard_id += 1

        # 保存最后一批
        if buffer_states:
            shard_path = os.path.join(config.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
            torch.save({
                'states': torch.stack(buffer_states),
                'next_states': torch.stack(buffer_next_states),
                'actions': torch.stack(buffer_actions),
                'rewards': torch.stack(buffer_rewards),
                'dones': torch.stack(buffer_dones)
            }, shard_path)
            print(f"保存 shard {shard_id}, size={len(buffer_states)})")
    @classmethod
    def load_offline_lstm(cls, path, model, config, seq_len=5):
        """
        构建 LSTM 时序输入的 offline 数据。
        每个样本是一个长度为 seq_len 的序列：
            - states: [seq_len, feature_dim]
            - actions: [seq_len]
            - rewards: [seq_len]
            - dones: [seq_len]
        """

        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.TO_PATH, exist_ok=True)

        seq_len = int(getattr(config, "SEQ_LEN", seq_len))
        shard_size = int(getattr(config, "SHARD_SIZE", 2000))
        shard_id = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading LSTM offline data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id, rewards, dones = cls._read_transition_arrays(data)
            rewards = cls._reshape_rewards(
                data=data,
                action_id=action_id,
                rewards=rewards,
                dones=dones,
                config=config,
            )

            # 编码整个轨迹
            encoded_states = []
            with torch.no_grad():
                for  i , v in enumerate(obs):
                    
                    rgb = torch.from_numpy(v['rgb']).float() / 255.0
                    depth = torch.from_numpy(v['depth']).float()
                    audio = torch.from_numpy(v['spectrogram'][0]).float()
                    if i == 0:
                        pre_rgb = torch.zeros_like(rgb)
                        pre_depth = torch.zeros_like(depth)
                    state = model.embedding_forward(
                        audio.to(model.device),
                        torch.cat([pre_rgb , rgb] , dim = 2).to(model.device),
                        torch.cat([pre_depth , depth] , dim = 2).to(model.device)
                    )

                    encoded_states.append(state.squeeze(0).cpu()) # 去除batch
                    pre_rgb = rgb
                    pre_depth = depth

            # 构造时序样本（滑动窗口）
            traj_len = len(encoded_states)
            for i in range(traj_len - seq_len):
                state_seq = torch.stack(encoded_states[i:i+seq_len])              # 当前状态序列
                next_state_seq = torch.stack(encoded_states[i+1:i+1+seq_len])    # 下一个状态序列
                a_seq = torch.tensor(action_id[i:i+seq_len], dtype=torch.long)
                r_seq = torch.tensor(rewards[i:i+seq_len], dtype=torch.float)
                d_seq = torch.tensor(dones[i:i+seq_len], dtype=torch.bool)
                buffer_states.append(state_seq)
                buffer_next_states.append(next_state_seq)
                buffer_actions.append(a_seq)
                buffer_rewards.append(r_seq)
                buffer_dones.append(d_seq)

                # 存 shard
                if len(buffer_states) >= shard_size:
                    shard_path = os.path.join(config.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
                    torch.save({
                        'states': torch.stack(buffer_states),
                        'next_states': torch.stack(buffer_next_states),
                        'actions': torch.stack(buffer_actions),
                        'rewards': torch.stack(buffer_rewards),
                        'dones': torch.stack(buffer_dones)
                    }, shard_path)
                    print(f"保存 shard {shard_id}, size={len(buffer_states)}")

                    buffer_states, buffer_next_states = [], []
                    buffer_actions, buffer_rewards, buffer_dones = [], [], []
                    shard_id += 1

        # 保存最后一批
        if buffer_states:
            shard_path = os.path.join(config.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
            torch.save({
                'states': torch.stack(buffer_states),
                'next_states': torch.stack(buffer_next_states),
                'actions': torch.stack(buffer_actions),
                'rewards': torch.stack(buffer_rewards),
                'dones': torch.stack(buffer_dones)
            }, shard_path)
            print(f"保存 shard {shard_id}, size={len(buffer_states)})")

    @classmethod
    def load_offline_lstm_v15(cls, path, model, config, seq_len=5):
        """
        v1_5 使用的 LSTM 离线编码存储：
        shard keys: states, next_states, actions, rewards, dones
        """
        return cls.load_offline_lstm(path=path, model=model, config=config, seq_len=seq_len)

    @classmethod
    def load_offline_two_frame(cls, path, model, config):
        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.TO_PATH, exist_ok=True)

        shard_size = int(getattr(config, "SHARD_SIZE", 10000))
        embed_batch_size = int(getattr(config, "EMBED_BATCH_SIZE", 64))
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        shard_id = 0
        total_samples = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        def flush_shard():
            nonlocal shard_id, total_samples
            if not buffer_states:
                return
            shard_path = os.path.join(config.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
            torch.save(
                {
                    "states": torch.stack(buffer_states),
                    "next_states": torch.stack(buffer_next_states),
                    "actions": torch.stack(buffer_actions),
                    "rewards": torch.stack(buffer_rewards),
                    "dones": torch.stack(buffer_dones),
                },
                shard_path,
            )
            print(f"保存 shard {shard_id}, size={len(buffer_states)}")
            total_samples += len(buffer_states)
            shard_id += 1
            buffer_states.clear()
            buffer_next_states.clear()
            buffer_actions.clear()
            buffer_rewards.clear()
            buffer_dones.clear()

        for file in tqdm(files, desc="Loading offline RL data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist() # 取出reset的时候的done。后续要删除这个地方。更改数据收集策略
            rewards = cls._reshape_rewards(
                data=data,
                action_id=action_id,
                rewards=rewards,
                dones=dones,
                config=config,
            )
    
            chunk_audio_now, chunk_trgb_now, chunk_tdepth_now = [], [], []
            chunk_audio_next, chunk_trgb_next, chunk_tdepth_next = [], [], []
            chunk_action, chunk_reward, chunk_done = [], [], []

            for i in range(len(obs) - 1):
                v_now = obs[i]
                v_next = obs[i + 1]
                a = action_id[i]
                r = rewards[i]
                d = dones[i]

                # 提取特征
                rgb_now = torch.from_numpy(v_now['rgb']).float() / 255.0
                depth_now = torch.from_numpy(v_now['depth']).float() 
                audio_now = torch.from_numpy(v_now['spectrogram'][0]).float()
                rgb_next = torch.from_numpy(v_next['rgb']).float() / 255.0
                depth_next = torch.from_numpy(v_next['depth']).float()
                audio_next = torch.from_numpy(v_next['spectrogram'][0]).float()
                if i == 0:
                    # 如果是第一个step ， 则在前面填充0
                    pre_rgb = torch.zeros_like(rgb_now)
                    pre_depth = torch.zeros_like(depth_now)
                    
                    trgb = torch.cat([pre_rgb , rgb_now] , dim = 2)
                    tdepth = torch.cat([pre_depth , depth_now] , dim = 2)
                    trgb_next =  torch.cat([rgb_now , rgb_next] , dim = 2 )
                    tdepth_next = torch.cat([depth_now , depth_next] , dim = 2)
                    pre_rgb = rgb_now
                    pre_depth = depth_now
                else:
                    trgb = torch.cat([pre_rgb , rgb_now] , dim = 2)
                    tdepth = torch.cat([pre_depth , depth_now] , dim = 2)
                    trgb_next =  torch.cat([rgb_now , rgb_next] , dim = 2 )
                    tdepth_next = torch.cat([depth_now , depth_next] , dim = 2)
                    pre_rgb = rgb_now
                    pre_depth = depth_now
                
                chunk_audio_now.append(audio_now)
                chunk_trgb_now.append(trgb)
                chunk_tdepth_now.append(tdepth)
                chunk_audio_next.append(audio_next)
                chunk_trgb_next.append(trgb_next)
                chunk_tdepth_next.append(tdepth_next)
                chunk_action.append(torch.tensor(a, dtype=torch.long))
                chunk_reward.append(torch.tensor(r, dtype=torch.float))
                chunk_done.append(torch.tensor(d, dtype=torch.bool))

                if len(chunk_audio_now) >= embed_batch_size:
                    with torch.no_grad():
                        states = model.embedding_forward(
                            torch.stack(chunk_audio_now).to(device),
                            torch.stack(chunk_trgb_now).to(device),
                            torch.stack(chunk_tdepth_now).to(device),
                        )
                        next_states = model.embedding_forward(
                            torch.stack(chunk_audio_next).to(device),
                            torch.stack(chunk_trgb_next).to(device),
                            torch.stack(chunk_tdepth_next).to(device),
                        )
                    if states.dim() == 1:
                        states = states.unsqueeze(0)
                    if next_states.dim() == 1:
                        next_states = next_states.unsqueeze(0)
                    states = states.detach().cpu()
                    next_states = next_states.detach().cpu()
                    for j in range(states.shape[0]):
                        buffer_states.append(states[j])
                        buffer_next_states.append(next_states[j])
                        buffer_actions.append(chunk_action[j])
                        buffer_rewards.append(chunk_reward[j])
                        buffer_dones.append(chunk_done[j])
                        if len(buffer_states) >= shard_size:
                            flush_shard()
                    chunk_audio_now.clear()
                    chunk_trgb_now.clear()
                    chunk_tdepth_now.clear()
                    chunk_audio_next.clear()
                    chunk_trgb_next.clear()
                    chunk_tdepth_next.clear()
                    chunk_action.clear()
                    chunk_reward.clear()
                    chunk_done.clear()

            if chunk_audio_now:
                with torch.no_grad():
                    states = model.embedding_forward(
                        torch.stack(chunk_audio_now).to(device),
                        torch.stack(chunk_trgb_now).to(device),
                        torch.stack(chunk_tdepth_now).to(device),
                    )
                    next_states = model.embedding_forward(
                        torch.stack(chunk_audio_next).to(device),
                        torch.stack(chunk_trgb_next).to(device),
                        torch.stack(chunk_tdepth_next).to(device),
                    )
                if states.dim() == 1:
                    states = states.unsqueeze(0)
                if next_states.dim() == 1:
                    next_states = next_states.unsqueeze(0)
                states = states.detach().cpu()
                next_states = next_states.detach().cpu()
                for j in range(states.shape[0]):
                    buffer_states.append(states[j])
                    buffer_next_states.append(next_states[j])
                    buffer_actions.append(chunk_action[j])
                    buffer_rewards.append(chunk_reward[j])
                    buffer_dones.append(chunk_done[j])
                    if len(buffer_states) >= shard_size:
                        flush_shard()

        flush_shard()
        print(f"PT 写入完成，总样本数: {total_samples}, 输出目录: {config.TO_PATH}")
        return {
            "output_dir": config.TO_PATH,
            "total_samples": total_samples,
            "num_shards": shard_id,
            "shard_size": shard_size,
            "embed_batch_size": embed_batch_size,
        }
    @classmethod
    def load_offline_with_hybrid(cls, path, config):
        files = cls.get_files(path=path)
        random.shuffle(files)
        os.makedirs(config.TO_PATH, exist_ok=True)

        shard_size = int(getattr(config, "SHARD_SIZE", 10000))
        shard_id = 0
        total_samples = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        def flush_shard():
            nonlocal shard_id, total_samples
            if len(buffer_states) == 0:
                return
            shard_path = os.path.join(config.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
            torch.save(
                {
                    # state 是 (audio, trgb, tdepth) tuple，按 list 保存避免错误 stack
                    "states": list(buffer_states),
                    "next_states": list(buffer_next_states),
                    "actions": torch.stack(buffer_actions),
                    "rewards": torch.stack(buffer_rewards),
                    "dones": torch.stack(buffer_dones),
                },
                shard_path,
                _use_new_zipfile_serialization=True
            )
            total_samples += len(buffer_states)
            print(f"保存 shard {shard_id}, size={len(buffer_states)}")
            shard_id += 1
            buffer_states.clear()
            buffer_next_states.clear()
            buffer_actions.clear()
            buffer_rewards.clear()
            buffer_dones.clear()

        for file in tqdm(files, desc="Loading offline RL data to PT"):
            with open(file, "rb") as f:
                data = pickle.load(f)

            obs = data["obs"]
            action_id = np.array(data["action_id"]).reshape(-1).tolist()
            rewards = np.array(data["reward"]).reshape(-1).tolist()
            dones = np.array(data["done"]).reshape(-1).tolist()
            rewards = cls._reshape_rewards(
                data=data,
                action_id=action_id,
                rewards=rewards,
                dones=dones,
                config=config,
            )

            for i in range(len(obs) - 1):
                v_now = obs[i]
                v_next = obs[i + 1]
                a = action_id[i]
                r = rewards[i]
                d = dones[i]

                rgb_now = torch.from_numpy(v_now["rgb"]).float() / 255.0
                depth_now = torch.from_numpy(v_now["depth"]).float()
                audio_now = torch.from_numpy(v_now["spectrogram"][0]).float()
                rgb_next = torch.from_numpy(v_next["rgb"]).float() / 255.0
                depth_next = torch.from_numpy(v_next["depth"]).float()
                audio_next = torch.from_numpy(v_next["spectrogram"][0]).float()

                if i == 0:
                    pre_rgb = torch.zeros_like(rgb_now)
                    pre_depth = torch.zeros_like(depth_now)

                trgb = torch.cat([pre_rgb, rgb_now], dim=2)
                tdepth = torch.cat([pre_depth, depth_now], dim=2)
                trgb_next = torch.cat([rgb_now, rgb_next], dim=2)
                tdepth_next = torch.cat([depth_now, depth_next], dim=2)
                pre_rgb = rgb_now
                pre_depth = depth_now

                state = (audio_now, trgb, tdepth)
                next_state = (audio_next, trgb_next, tdepth_next)

                buffer_states.append(state)
                buffer_next_states.append(next_state)
                buffer_actions.append(torch.tensor(a, dtype=torch.long))
                buffer_rewards.append(torch.tensor(r, dtype=torch.float))
                buffer_dones.append(torch.tensor(d, dtype=torch.bool))

                if len(buffer_states) >= shard_size:
                    flush_shard()

        flush_shard()
        print(f"PT 写入完成，总样本数: {total_samples}, 输出目录: {config.TO_PATH}")
