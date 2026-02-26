# dataloader
from torch.utils.data import Dataset
import os
import pickle
from tqdm import tqdm
import pdb
import torch
import numpy as np
import random
import glob
class LoadLmdb:

    def __init__(self, path):

        self.paths = self._deal_path(path)
        
        
    @classmethod
    def get_files(cls , path):
        import os
        # 获取到object的路径
        object_name_files = []
        # parents = [os.path.join(path , i) for i in os.listdir(path = path)]
        # for path in parents:
        object_name_files.extend([os.path.join(path , i ) for i in os.listdir(path=path) if i != "a.md"])
        files = []
        
        for scene in object_name_files:
            files.extend([os.path.join(scene , i) for i in os.listdir(scene)])
        return files
    @classmethod
    def load_pt(cls , path , config):
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
    def load_two_frame_pt(cls , path , config):
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

        shard_size = 2000  # 每个shard包含的序列数
        shard_id = 0

        buffer_audio_states, buffer_visual_audio_states , buffer_next_audio_states ,  buffer_next_visual_audio_states= [], [] , [] , []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading LSTM offline data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist()

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

        shard_size = 2000  # 每个shard包含的序列数
        shard_id = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading LSTM offline data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist()

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

        shard_size = 2000  # 每个shard包含的序列数
        shard_id = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading LSTM offline data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist()

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
    def load_offline_two_frame(cls , path , model , config):
        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.TO_PATH, exist_ok=True)

        shard_size = 10000  # 每个shard包含的样本数
        shard_id = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading offline RL data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist() # 取出reset的时候的done。后续要删除这个地方。更改数据收集策略
    
            # if data['info'][0]['distance_to_goal'] > 5 :
            #     tqdm.write(f"{data['info'][0]['distance_to_goal']} drop")
            #     continue
            # 遍历每一对 (state, next_state)
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
                
                # 编码成状态向量
                with torch.no_grad():
                    state = model.embedding_forward(audio_now.to(model.device), trgb.to(model.device) , tdepth.to(model.device))
                    next_state = model.embedding_forward(audio_next.to(model.device), trgb_next.to(model.device) , tdepth_next.to(model.device))
                # state = (audio_now , trgb , tdepth)
                # next_state = (audio_next , trgb_next , tdepth_next)

                buffer_states.append(state)
                buffer_next_states.append(next_state)
                buffer_actions.append(torch.tensor(a, dtype=torch.long))
                buffer_rewards.append(torch.tensor(r, dtype=torch.float))
                buffer_dones.append(torch.tensor(d, dtype=torch.bool))

                # 存 shard
                if len(buffer_states) >= shard_size:
                    shard_path = os.path.join(config.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
                    torch.save({
                        'states': torch.stack(buffer_states),
                        'next_states': torch.stack(buffer_next_states),
                        # 'states':buffer_states,
                        # 'next_states':buffer_next_states,
                        'actions': torch.stack(buffer_actions),
                        'rewards': torch.stack(buffer_rewards),
                        'dones': torch.stack(buffer_dones)
                    }, shard_path)
                    print(f"保存 shard {shard_id}, size={len(buffer_states)}")

                    # 清空缓存
                    buffer_states, buffer_next_states = [], []
                    buffer_actions, buffer_rewards, buffer_dones = [], [], []
                    shard_id += 1

        # 保存最后一个不满的 shard
        if buffer_states:
            shard_path = os.path.join(config.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
            torch.save({
                'states': torch.stack(buffer_states),
                'next_states': torch.stack(buffer_next_states),
                'actions': torch.stack(buffer_actions),
                'rewards': torch.stack(buffer_rewards),
                'dones': torch.stack(buffer_dones)
            }, shard_path)
            print(f"保存 shard {shard_id}, size={len(buffer_states)}")
    @classmethod
    def load_offline_with_hybrid(cls , path , config):
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
    @classmethod
    def load_offline(cls, path, model, config):
        """

        每个样本包含：
            - state: 当前状态编码特征
            - next_state: 下一状态编码特征
            - action: 动作 (long tensor)
            - reward: 奖励 (float tensor)
            - done: 终止标志 (bool tensor)
        """

        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.TO_PATH, exist_ok=True)

        shard_size = 10000  # 每个shard包含的样本数
        shard_id = 0

        buffer_states, buffer_next_states = [], []
        buffer_actions, buffer_rewards, buffer_dones = [], [], []

        for file in tqdm(files, desc="Loading offline RL data"):
            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist() # 取出reset的时候的done。后续要删除这个地方。更改数据收集策略
    
            # if data['info'][0]['distance_to_goal'] > 5 :
            #     tqdm.write(f"{data['info'][0]['distance_to_goal']} drop")
            #     continue
            # 遍历每一对 (state, next_state)
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

                # 编码成状态向量
                with torch.no_grad():
                    state = model.embedding_forward(audio_now.to(model.device), rgb_now.to(model.device) , depth_now.to(model.device))
                    next_state = model.embedding_forward(audio_next.to(model.device), rgb_next.to(model.device) , depth_next.to(model.device))

                buffer_states.append(state)
                buffer_next_states.append(next_state)
                buffer_actions.append(torch.tensor(a, dtype=torch.long))
                buffer_rewards.append(torch.tensor(r, dtype=torch.float))
                buffer_dones.append(torch.tensor(d, dtype=torch.bool))

                # 存 shard
                if len(buffer_states) >= shard_size:
                    shard_path = os.path.join(config.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
                    torch.save({
                        'states': torch.stack(buffer_states),
                        'next_states': torch.stack(buffer_next_states),
                        'actions': torch.stack(buffer_actions),
                        'rewards': torch.stack(buffer_rewards),
                        'dones': torch.stack(buffer_dones)
                    }, shard_path)
                    print(f"保存 shard {shard_id}, size={len(buffer_states)}")

                    # 清空缓存
                    buffer_states, buffer_next_states = [], []
                    buffer_actions, buffer_rewards, buffer_dones = [], [], []
                    shard_id += 1

        # 保存最后一个不满的 shard
        if buffer_states:
            shard_path = os.path.join(config.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
            torch.save({
                'states': torch.stack(buffer_states),
                'next_states': torch.stack(buffer_next_states),
                'actions': torch.stack(buffer_actions),
                'rewards': torch.stack(buffer_rewards),
                'dones': torch.stack(buffer_dones)
            }, shard_path)
            print(f"保存 shard {shard_id}, size={len(buffer_states)}")   
    @classmethod
    def get_values_tuple(cls, data):
        import pdb;pdb.set_trace()
        obs = data['obs']
        audio = data[0][0]['audio'][:-1]
        visual = data[0][0]['camera'][:-1]
        action = data[0][0]['rl_pred']
        return audio ,visual , action

    @classmethod
    def get_values_offline(cls, data):
        """
        处理数据的函数
        """
        audio = data[0][0]['audio'][:-1]
        visual = data[0][0]['camera'][:-1]
        next_audio = data[0][0]['audio'][1:]
        next_visual = data[0][0]['camera'][1:]
        action = data[0][0]['rl_pred']
        rewards = data[0][0]['reward']
        dones = data[1]
        return audio , visual, next_audio, next_visual, action, rewards, dones
    @classmethod
    def _deal_path(cls,path):

        # 简单情况的deal，复杂的话估计要重复好几次，不如重新写一个def
        files = os.listdir(path)
        paths = [os.path.join(path , i) for i in files]
        return paths

class AVtrans:
    @classmethod
    def deal_data(cls, audio, visual):
        """
        处理音频和视觉数据
        """
        audio = audio.astype(float) / 32768.0
        audio_tensor = cls.mel_audio(audio)
        visual = visual[:,:,:-1]
        image_tensor = torch.tensor(np.array(visual), dtype=torch.float32) # 确保视觉数据是float32类型
        return  audio_tensor , image_tensor
    @classmethod
    def mel_audio(cls, audio):
        """
        处理音频数据，转换为梅尔频谱图
        """
        import librosa
        left_audio = audio[0]
        right_audio = audio[1]
        
        sr = 48000
        mel_spec_left = librosa.feature.melspectrogram(y=left_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spec_left = librosa.power_to_db(mel_spec_left, ref=np.max)

        mel_spec_right = librosa.feature.melspectrogram(y=right_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spec_right = librosa.power_to_db(mel_spec_right, ref=np.max)

        mel_spec = np.stack([mel_spec_left, mel_spec_right], axis=0)
        audio_tensor = torch.tensor(mel_spec, dtype=torch.float32)

        return audio_tensor

class ShardedPTDataset(Dataset):
    def __init__(self, shard_pattern, preload=True):
        super().__init__()
        self.shard_files = []
        for pattern in shard_pattern:
            self.shard_files.extend(sorted(glob.glob(pattern))[:7])
        assert len(self.shard_files) > 0, f"No shards found at {shard_pattern}"
        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_map = []    # 全局 index → (shard_id, local_index)

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"scan shard id {shard_id}")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.total_size = sum(self.shard_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]

        rgb = data["rgb"][local_idx]
        depth = data['depth'][local_idx]
        audio = data["audios"][local_idx]
        action = data["actions"][local_idx]
        angle = data['angles'][local_idx]
        std_audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        return  std_audio, rgb , depth , angle , action
      
class ShardedPTDatasetOffline(Dataset):
    def __init__(self, train_shard_dir, attention=False, preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/pt/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        self.shard_files = []
        self.use_attention = attention
        self.get_files(train_shard_dir)
        # for pattern in shard_pattern:
        #     lists_ = glob.glob(pattern)
        #     random.shuffle(lists_)
        #     import pdb;pdb.set_trace()
        #     self.shard_files.extend(lists_[:40])
        assert len(self.shard_files) > 0, f"No shards found at {train_shard_dir}"
        print(
            f"[ShardedPTDatasetOffline] found {len(self.shard_files)} shards in {train_shard_dir}"
        )

        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_map = []    # 全局 index → (shard_id, local_index)

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"[ShardedPTDatasetOffline] loading shard {shard_id}: {shard_file}")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)
            print(f"[ShardedPTDatasetOffline] shard {shard_id} samples={size}")

            # 构建 index map
            for i in range(size):
                self.index_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.total_size = sum(self.shard_sizes)
        print(f"[ShardedPTDatasetOffline] total samples: {self.total_size}")

    def get_files(self, train_shard_dir):
        if not os.path.isdir(train_shard_dir):
            raise ValueError(f"train_shard_pattern must be a directory path, got: {train_shard_dir}")
        files = sorted(glob.glob(os.path.join(train_shard_dir, "offline_rl_shard_*.pt")))
        if not files:
            files = sorted(glob.glob(os.path.join(train_shard_dir, "*.pt")))
        self.shard_files.extend(files)
        print(f"[ShardedPTDatasetOffline] shard glob matched {len(files)} files")

    def replay(self):
        pass

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]


        if self.use_attention:
            states_audio = data["states_audio"][local_idx]
            states_visual_audio = data["states_visual_audio"][local_idx]
            next_states_audio = data['next_states_audio'][local_idx]
            next_states_visual_audio = data['next_states_visual_audio'][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            states_audio = states_audio.squeeze(0)
            states_visual_audio = states_visual_audio.squeeze(0)
            next_states_audio = next_states_audio.squeeze(0)
            next_states_visual_audio = next_states_visual_audio.squeeze(0)
            return (states_audio , states_visual_audio), (next_states_audio , next_states_visual_audio) , action, reward, done
        else:
            # 取出一个 transition
            state       = data["states"][local_idx]
            next_state  = data["next_states"][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            # state = state.squeeze(0)
            # next_state = next_state.squeeze(0)
            return state, next_state , action, reward, done

class RandomReloadShardedPTDatasetOffline(Dataset):
    def __init__(
        self,
        train_shard_dir,
        attention=False,
        shards_per_epoch=8,
        reload_every_epochs=10,
        seed=None,
    ):
        super().__init__()
        self.shard_files = []
        self.use_attention = attention
        self._rng = random.Random(seed) if seed is not None else random
        self.get_files(train_shard_dir)
        assert len(self.shard_files) > 0, f"No shards found at {train_shard_dir}"

        self.shards_per_epoch = max(1, int(shards_per_epoch))
        self.reload_every_epochs = max(1, int(reload_every_epochs))
        self._epoch_counter = 0

        self.active_shard_files = []
        self.active_shards = []
        self.index_map = []
        self.total_size = 0
        self._reload_shards(initial=True)

    def get_files(self, train_shard_dir):
        if not os.path.isdir(train_shard_dir):
            raise ValueError(f"train_shard_pattern must be a directory path, got: {train_shard_dir}")
        files = sorted(glob.glob(os.path.join(train_shard_dir, "offline_rl_shard_*.pt")))
        if not files:
            files = sorted(glob.glob(os.path.join(train_shard_dir, "*.pt")))
        self.shard_files.extend(files)

    def _rng_shuffle(self, values):
        if hasattr(self._rng, "shuffle"):
            self._rng.shuffle(values)
        else:
            random.shuffle(values)

    def _rng_sample(self, values, k):
        if hasattr(self._rng, "sample"):
            return self._rng.sample(values, k)
        return random.sample(values, k)

    def _reload_shards(self, initial=False):
        target = min(self.shards_per_epoch, len(self.shard_files))
        if target == len(self.shard_files):
            selected = list(self.shard_files)
        else:
            selected = self._rng_sample(self.shard_files, target)

        self.active_shard_files = selected
        self.active_shards = []
        self.index_map = []

        total = 0
        for active_id, shard_file in enumerate(self.active_shard_files):
            print(
                f"[RandomReloadShardedPTDatasetOffline] loading active shard "
                f"{active_id+1}/{len(self.active_shard_files)}: {shard_file}"
            )
            data = torch.load(shard_file, map_location="cpu")
            self.active_shards.append(data)
            size = len(data["actions"])
            total += size
            print(
                f"[RandomReloadShardedPTDatasetOffline] active shard {active_id} samples={size}"
            )
            for i in range(size):
                self.index_map.append((active_id, i))

        self.total_size = total
        phase = "initial" if initial else "reload"
        print(
            f"[RandomReloadShardedPTDatasetOffline] {phase}: "
            f"loaded_shards={len(self.active_shard_files)} total_samples={self.total_size}"
        )

    def on_epoch_end(self):
        self._epoch_counter += 1
        if self._epoch_counter % self.reload_every_epochs == 0:
            self._reload_shards(initial=False)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        active_id, local_idx = self.index_map[index]
        data = self.active_shards[active_id]

        if self.use_attention:
            states_audio = data["states_audio"][local_idx]
            states_visual_audio = data["states_visual_audio"][local_idx]
            next_states_audio = data["next_states_audio"][local_idx]
            next_states_visual_audio = data["next_states_visual_audio"][local_idx]
            action = data["actions"][local_idx]
            reward = data["rewards"][local_idx]
            done = data["dones"][local_idx]
            states_audio = states_audio.squeeze(0)
            states_visual_audio = states_visual_audio.squeeze(0)
            next_states_audio = next_states_audio.squeeze(0)
            next_states_visual_audio = next_states_visual_audio.squeeze(0)
            return (
                (states_audio, states_visual_audio),
                (next_states_audio, next_states_visual_audio),
                action,
                reward,
                done,
            )

        state = data["states"][local_idx]
        next_state = data["next_states"][local_idx]
        action = data["actions"][local_idx]
        reward = data["rewards"][local_idx]
        done = data["dones"][local_idx]
        return state, next_state, action, reward, done
        
class ShardedPTDatasetOfflineBuffer(Dataset):
    def __init__(self, train_json,  attention = False , preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/pt/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        # self.shard_files = []
        self.use_attention = attention
        self.get_file_buffer(train_json)
        # for pattern in shard_pattern:
        #     lists_ = glob.glob(pattern)
        #     random.shuffle(lists_)
        #     import pdb;pdb.set_trace()
        #     self.shard_files.extend(lists_[:40])
        # assert len(self.shard_files) > 0, f"No shards found at {train_json}"

        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_greedy_map = []    # 全局 index → (shard_id, local_index)
        self.index_1_map = []
        self.index_2_map = []

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.buffer_greedy):
            print(f"scan shard id {shard_id} and {shard_file} ...")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_greedy_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位


        for shard_id, shard_file in enumerate(self.buffer_1):
            print(f"scan shard id {shard_id} ...")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_1_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        for shard_id, shard_file in enumerate(self.buffer_2):
            print(f"scan shard id {shard_id} ...")
            data = torch.load(shard_file, map_location="cpu")
            size = len(data["actions"])
            self.shard_sizes.append(size)

            # 构建 index map
            for i in range(size):
                self.index_2_map.append((shard_id, i))

            if preload:
                self.shards.append(data)  # 直接放内存
            else:
                self.shards.append(None)  # 占位

        self.index_map = self.index_greedy_map
        self.total_size = sum(self.shard_sizes)
        print(f"Total samples: {self.total_size}")

    def get_files(self , train_json):
        import json
        files = []
        with open(train_json , 'r') as f:
            train_json_data = json.load(f) 
        path = train_json_data['path']
        split = train_json_data['split']
        for beta in path.keys():
            for distance in path[beta].keys():
                print(path[beta][distance])
                lists_1 = glob.glob(path[beta][distance])
                random.shuffle(lists_1)
                files.extend(lists_1[:split[beta][distance]])
                # self.shard_files.extend(lists_1)
        return files
    def get_file_buffer(self , train_json):
        import json
        with open(train_json , 'r') as f:
            train_json_data = json.load(f)
        self.buffer_greedy = self.get_files(train_json=train_json_data['greedy'])
        self.buffer_1 = self.get_files(train_json=train_json_data['1.5'])
        self.buffer_2 = self.get_files(train_json=train_json_data['2.0'])

    def replay(self , logger):
        # 每一次replay greedy减少10k。相应的1.5和2.0各加5k
        # 当greedy减少到50k时 ， 不断sample1.5和2.0，仍确保1.5和2.0始终总体占据50k
        logger.info(f"开始进行replay buffer")
        random.shuffle(self.index_greedy_map)
        random.shuffle(self.index_1_map)
        random.shuffle(self.index_2_map)
        logger.info(f"打乱所有的列表,目前buffer长度{len(self.index_map)}")
        self.index_map = self.index_map[10000:] # 去除前10kgreedy数据
        cache_1_map = self.index_1_map[:5000]
        self.index_1_map = self.index_1_map[5000:]
        cache_2_map = self.index_2_map[:5000]
        self.index_2_map = self.index_2_map[5000:]
        logger.info(f"cache 1 map {len(cache_1_map)}")
        self.index_map.extend(cache_1_map)
        self.index_map.extend(cache_2_map)
        logger.info(f"打乱所有的列表,目前buffer长度{len(self.index_map)} , 1 map {len(self.index_1_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        
        shard_id, local_idx = self.index_map[index]

        # 如果没预加载，就临时加载这个 shard
        if self.shards[shard_id] is None:
            data = torch.load(self.shard_files[shard_id], map_location="cpu")
            self.shards[shard_id] = data
        else:
            data = self.shards[shard_id]


        if self.use_attention:
            states_audio = data["states_audio"][local_idx]
            states_visual_audio = data["states_visual_audio"][local_idx]
            next_states_audio = data['next_states_audio'][local_idx]
            next_states_visual_audio = data['next_states_visual_audio'][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            states_audio = states_audio.squeeze(0)
            states_visual_audio = states_visual_audio.squeeze(0)
            next_states_audio = next_states_audio.squeeze(0)
            next_states_visual_audio = next_states_visual_audio.squeeze(0)
            return (states_audio , states_visual_audio), (next_states_audio , next_states_visual_audio) , action, reward, done
        else:
            # 取出一个 transition
            state       = data["states"][local_idx]
            next_state  = data["next_states"][local_idx]
            action      = data["actions"][local_idx]
            reward      = data["rewards"][local_idx]
            done        = data["dones"][local_idx]
            state = state.squeeze(0)
            next_state = next_state.squeeze(0)

            return state, next_state , action, reward, done
