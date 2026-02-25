# dataloader
from torch.utils.data import DataLoader , Dataset, Sampler
import os
import lmdb , pickle
from tqdm import tqdm
import sys
 
import pdb
import librosa
import time
import torch
import numpy as np
from PIL import Image
import random
from network.hybird.foundation_model import Network
import glob
from collections import OrderedDict, defaultdict
CHUNK_SIZE = 2048
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
    def load_lmdb(cls,path , config):
        
        # 这里看需不需要使用这个dealpath ，也可以在外部处理，如果文件层级简单的话推荐使用内部的
        # paths = cls._deal_path(paths)
        
        paths = cls.get_files(path=path)
        os.makedirs(config.LMDB.TO_PATH , exist_ok=True)
        env = lmdb.open(config.LMDB.TO_PATH, map_size=1024*1014*1024*30)
        sum_index = 0
        txn = env.begin(write=True) 
        for path_i in tqdm(paths):
            with open(path_i ,'rb') as f:
                
                data = pickle.load(f)
            obs = data['obs'][:-1] # drop -1 step
            sound_id = data['sound_id']
            action_id = data['action_id'][0]
            
            for index in range(len(obs)):
                
                key = f'index_{sum_index}'.encode('utf-8')
                # spectrogram , audio , visual , angle , sound_name , action
                values = (obs[index]["spectrogram"][0] , obs[index]['spectrogram'][1] , obs[index]['rgb'] , obs[index]['angle'] , sound_id , action_id[index])
                values = pickle.dumps(values)
                txn.put(key, values)
                sum_index+=1

                if sum_index % 5000 == 0: 
                    txn.commit()
                    txn = env.begin(write=True)
        
        # 在内部设置了长度大小
        key = f'__len__'.encode('utf-8')
        len_ = pickle.dumps(sum_index)
        txn.put(key , len_)
        txn.commit()
        env.close()
        print(f"dataset location the {config.LMDB.TO_PATH}")
    @classmethod
    def load_offline_two_frame_lmdb(cls, path, model, config):

        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

        # --------- 打开 LMDB ----------
        map_size = 1024 ** 3 * 20  # 1TB 上限（必须给足，不够会报错）
        env = lmdb.open(
            config.LMDB.TO_PATH,
            map_size=map_size,
            subdir=True,
            meminit=False,
            map_async=True
        )

        txn = env.begin(write=True)

        global_index = 0
        commit_interval = 1000   # 每1000条 commit 一次（非常重要）

        for file in tqdm(files, desc="Loading offline RL data"):

            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist()

            for i in range(len(obs) - 1):

                v_now = obs[i]
                v_next = obs[i + 1]

                a = action_id[i]
                r = rewards[i]
                d = dones[i]

                # ---------- 提取 ----------
                rgb_now = torch.from_numpy(v_now['rgb']).float() / 255.0
                depth_now = torch.from_numpy(v_now['depth']).float()
                audio_now = torch.from_numpy(v_now['spectrogram'][0]).float()

                rgb_next = torch.from_numpy(v_next['rgb']).float() / 255.0
                depth_next = torch.from_numpy(v_next['depth']).float()
                audio_next = torch.from_numpy(v_next['spectrogram'][0]).float()

                if i == 0:
                    pre_rgb = torch.zeros_like(rgb_now)
                    pre_depth = torch.zeros_like(depth_now)

                trgb = torch.cat([pre_rgb, rgb_now], dim=2)
                tdepth = torch.cat([pre_depth, depth_now], dim=2)
                trgb_next = torch.cat([rgb_now, rgb_next], dim=2)
                tdepth_next = torch.cat([depth_now, depth_next], dim=2)

                pre_rgb = rgb_now
                pre_depth = depth_now

                # ---------- embedding ----------
                with torch.no_grad():
                    state = model.embedding_forward(
                        audio_now.to(model.device),
                        trgb.to(model.device),
                        tdepth.to(model.device)
                    ).cpu()

                    next_state = model.embedding_forward(
                        audio_next.to(model.device),
                        trgb_next.to(model.device),
                        tdepth_next.to(model.device)
                    ).cpu()

                sample = {
                    "state": state,
                    "next_state": next_state,
                    "action": torch.tensor(a, dtype=torch.long),
                    "reward": torch.tensor(r, dtype=torch.float),
                    "done": torch.tensor(d, dtype=torch.bool)
                }

                key = f"{global_index:08d}".encode()
                txn.put(key, pickle.dumps(sample))

                global_index += 1

                # ---------- 定期 commit ----------
                if global_index % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)

        txn.commit()
        env.sync()
        env.close()

        print(f"LMDB 写入完成，总样本数: {global_index}")
    @classmethod
    def load_offline_with_hybrid_lmdb(cls, path, config):

        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

        # ⚠️ 必须给足 map_size
        map_size = 1024 ** 3 * 200  # 200G 
        env = lmdb.open(
            config.LMDB.TO_PATH,
            map_size=map_size,
            subdir=True,
            meminit=False,
            map_async=True
        )

        txn = env.begin(write=True)

        global_index = 0
        commit_interval = 1000  # 每1000条 commit 一次

        for file in tqdm(files, desc="Loading offline RL data"):

            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1).tolist()
            rewards = np.array(data['reward']).reshape(-1).tolist()
            dones = np.array(data['done']).reshape(-1).tolist()

            for i in range(len(obs) - 1):

                v_now = obs[i]
                v_next = obs[i + 1]

                a = action_id[i]
                r = rewards[i]
                d = dones[i]

                # ---------- 提取特征 ----------
                rgb_now = torch.from_numpy(v_now['rgb']).float() / 255.0
                depth_now = torch.from_numpy(v_now['depth']).float()
                audio_now = torch.from_numpy(v_now['spectrogram'][0]).float()

                rgb_next = torch.from_numpy(v_next['rgb']).float() / 255.0
                depth_next = torch.from_numpy(v_next['depth']).float()
                audio_next = torch.from_numpy(v_next['spectrogram'][0]).float()

                if i == 0:
                    pre_rgb = torch.zeros_like(rgb_now)
                    pre_depth = torch.zeros_like(depth_now)

                trgb = torch.cat([pre_rgb, rgb_now], dim=2)
                tdepth = torch.cat([pre_depth, depth_now], dim=2)
                trgb_next = torch.cat([rgb_now, rgb_next], dim=2)
                tdepth_next = torch.cat([depth_now, depth_next], dim=2)

                pre_rgb = rgb_now
                pre_depth = depth_now

                # ⚠️ 必须保证是 CPU tensor
                sample = {
                    "state": (
                        audio_now.cpu(),
                        trgb.cpu(),
                        tdepth.cpu()
                    ),
                    "next_state": (
                        audio_next.cpu(),
                        trgb_next.cpu(),
                        tdepth_next.cpu()
                    ),
                    "action": torch.tensor(a, dtype=torch.long),
                    "reward": torch.tensor(r, dtype=torch.float),
                    "done": torch.tensor(d, dtype=torch.bool)
                }

                key = f"{global_index:08d}".encode()
                txn.put(key, pickle.dumps(sample))

                global_index += 1

                # ---------- 定期 commit ----------
                if global_index % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)

        txn.commit()
        env.sync()
        env.close()

        print(f"LMDB 写入完成，总样本数: {global_index}")
    
    @classmethod
    def load_offline_with_hybrid_lmdb_chunked_store(cls, path, config):

        CHUNK_SIZE = config.LMDB.CHUNK_SIZE

        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

        map_size = 1024 ** 3 * 200  # 200G

        env = lmdb.open(
            config.LMDB.TO_PATH,
            map_size=map_size,
            subdir=True,
            meminit=False,
            map_async=True
        )

        txn = env.begin(write=True)

        # ===== 全局 buffer =====
        buffer = {
            "state_audio": [],
            "state_rgb": [],
            "state_depth": [],
            "next_audio": [],
            "next_rgb": [],
            "next_depth": [],
            "action": [],
            "reward": [],
            "done": []
        }

        chunk_id = 0

        for file in tqdm(files, desc="Building Cross-Episode Chunked LMDB"):

            with open(file, 'rb') as f:
                data = pickle.load(f)

            obs = data['obs']
            action_id = np.array(data['action_id']).reshape(-1)
            rewards = np.array(data['reward']).reshape(-1)
            dones = np.array(data['done']).reshape(-1)

            T = len(obs)

            # ---------- 构造 episode tensor ----------

            rgb = torch.stack([
                torch.from_numpy(o['rgb']).float() / 255.0
                for o in obs
            ])

            depth = torch.stack([
                torch.from_numpy(o['depth']).float()
                for o in obs
            ])

            audio = torch.stack([
                torch.from_numpy(o['spectrogram'][0]).float()
                for o in obs
            ])

            rgb_prev = torch.zeros_like(rgb)
            depth_prev = torch.zeros_like(depth)

            rgb_prev[1:] = rgb[:-1]
            depth_prev[1:] = depth[:-1]

            trgb = torch.cat([rgb_prev, rgb], dim=3)
            tdepth = torch.cat([depth_prev, depth], dim=3)

            trgb_next = trgb[1:]
            tdepth_next = tdepth[1:]
            audio_next = audio[1:]

            trgb = trgb[:-1]
            tdepth = tdepth[:-1]
            audio = audio[:-1]

            action = torch.from_numpy(action_id).long()
            reward = torch.from_numpy(rewards).float()
            done = torch.from_numpy(dones).bool()

            N = trgb.shape[0]
            # ===== 逐 step 加入全局 buffer =====
            for i in range(N):

                buffer["state_audio"].append(audio[i])
                buffer["state_rgb"].append(trgb[i])
                buffer["state_depth"].append(tdepth[i])

                buffer["next_audio"].append(audio_next[i])
                buffer["next_rgb"].append(trgb_next[i])
                buffer["next_depth"].append(tdepth_next[i])

                buffer["action"].append(action[i])
                buffer["reward"].append(reward[i])
                buffer["done"].append(done[i])

                # ===== 满 chunk 才写 =====
                if len(buffer["action"]) >= CHUNK_SIZE:

                    chunk = {
                        k: torch.stack(v).cpu()
                        for k, v in buffer.items()
                    }

                    key = f"{chunk_id:08d}".encode()
                    txn.put(key, pickle.dumps(chunk))

                    chunk_id += 1

                    # 清空 buffer
                    for k in buffer:
                        buffer[k] = []

                    if chunk_id % 100 == 0:
                        txn.commit()
                        txn = env.begin(write=True)

        # ===== 写入剩余不足 CHUNK_SIZE 的部分 =====
        if len(buffer["action"]) > 0:

            chunk = {
                k: torch.stack(v).cpu()
                for k, v in buffer.items()
            }

            key = f"{chunk_id:08d}".encode()
            txn.put(key, pickle.dumps(chunk))

        txn.commit()
        env.sync()
        env.close()
    
    @classmethod
    def load_lmdb_offline_rl(cls, paths):
        """
        离线RL的LMDB数据加载
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_name = input("please input the dataset name:")
        env = lmdb.open(f"./experiment/dataset/{dataset_name}", map_size=1024*1024*1024*30)
        foundation_model = Network().to(device)
        checkpoint = torch.load('./experiment/train/ckpt/model_epoch_1000.pth')
        foundation_model.load_state_dict(checkpoint)
        foundation_model.eval()
        sum_index = 0
        txn = env.begin(write=True) 
        for path_i in tqdm(paths):
            with open(path_i ,'rb') as f:
                data = pickle.load(f)

                audio , visual, next_audio, next_visual, action, rewards, dones = cls.get_values_offline(data)
                dones = [int(i[0]) for i in dones]

                for index in range(len(audio)):
                    key = f'index_{sum_index}'.encode('utf-8')
                    audio_t , visual_t = AVtrans.deal_data(audio[index], visual[index])
                    next_audio_t, next_visual_t = AVtrans.deal_data(next_audio[index], next_visual[index])
                    state = foundation_model(audio_t.unsqueeze(0).to(device), visual_t.unsqueeze(0).to(device))
                    next_state = foundation_model(next_audio_t.unsqueeze(0).to(device), next_visual_t.unsqueeze(0).to(device))
                    value = (state.squeeze(0).detach().cpu(), next_state.squeeze(0).detach().cpu(), action[index], rewards[index], dones[index])
                    values = pickle.dumps(value)
                    txn.put(key, values)
                    sum_index+=1

                    if sum_index % 5000 == 0: 
                        txn.commit()
                        txn = env.begin(write=True)
        
        # 在内部设置了长度大小
        key = f'__len__'.encode('utf-8')
        len_ = pickle.dumps(sum_index)
        txn.put(key , len_)
        txn.commit()
        env.close()
        print(f"dataset location the ./experiment/dataset/{dataset_name}")

    @classmethod
    def load_pt(cls , path , config):
        files = cls.get_files(path=path)
        random.shuffle(files)
        os.makedirs(config.LMDB.TO_PATH , exist_ok=True)
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
                    }, f"{config.LMDB.TO_PATH}/foundation_model_shard_{shard_id}.pt")

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
            }, f"{config.LMDB.TO_PATH}/foundation_model_shard_{shard_id}.pt")
            print(f"保存 shard {shard_id}, size={len(buffer_rgb)}")
    
    @classmethod
    def load_two_frame_pt(cls , path , config):
        files = cls.get_files(path=path)
        random.shuffle(files)

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

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
                    }, f"{config.LMDB.TO_PATH}/foundation_model_shard_{shard_id}.pt")

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
            }, f"{config.LMDB.TO_PATH}/foundation_model_shard_{shard_id}.pt")
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

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

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
                    shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
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
            shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
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

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

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
                    shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
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
            shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
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

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

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
                    shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
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
            shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_lstm_shard_{shard_id}.pt")
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

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

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
                    shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
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
            shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
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

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

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
                # with torch.no_grad():
                #     state = model.embedding_forward(audio_now.to(model.device), trgb.to(model.device) , tdepth.to(model.device))
                #     next_state = model.embedding_forward(audio_next.to(model.device), trgb_next.to(model.device) , tdepth_next.to(model.device))
                state = (audio_now , trgb , tdepth)
                next_state = (audio_next , trgb_next , tdepth_next)

                buffer_states.append(state)
                buffer_next_states.append(next_state)
                buffer_actions.append(torch.tensor(a, dtype=torch.long))
                buffer_rewards.append(torch.tensor(r, dtype=torch.float))
                buffer_dones.append(torch.tensor(d, dtype=torch.bool))

                # 存 shard
                if len(buffer_states) >= shard_size:
                    shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
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
            shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
            torch.save({
                'states': torch.stack(buffer_states),
                'next_states': torch.stack(buffer_next_states),
                'actions': torch.stack(buffer_actions),
                'rewards': torch.stack(buffer_rewards),
                'dones': torch.stack(buffer_dones)
            }, shard_path)
            print(f"保存 shard {shard_id}, size={len(buffer_states)}")
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

        os.makedirs(config.LMDB.TO_PATH, exist_ok=True)

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
                    shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
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
            shard_path = os.path.join(config.LMDB.TO_PATH, f"offline_rl_shard_{shard_id}.pt")
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

class VADE(Dataset):
    def __init__(self, path , transform = None):
        super().__init__()

        self.env = lmdb.open(
            path,
            readonly=True,
        )
        self.txn = self.env.begin()
        

        self.transform = transform

    def __getitem__(self, index):
        key = f"index_{index}".encode("utf-8")
        value = self.txn.get(key)
        if value is None:
            raise IndexError(f"Index {index} not found in LMDB dataset")

        sample = pickle.loads(value)
        spectrogram , audio , visual , angle , sound_name , action = sample
        angle = np.degrees(angle[1])
        # visual , audio= self._deal_datas(visual,audio)

        return spectrogram , angle

    def __len__(self):
        length_key = b'__len__'
        length = self.txn.get(length_key)
        r = pickle.loads(length)
        return r
    
    def _deal_datas(self,audio , visual):
        
        audio = audio.astype(float) / 32768.0
        audio_tensor = self.mel_audio(audio)
        visual = visual[:,:,:-1]
        image_tensor = torch.tensor(np.array(visual))

        return image_tensor , audio_tensor

    def mel_audio(self,audio):
        # 更改收集数据代码后，这个地方要改变
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

class VADE_Offline(Dataset):
    def __init__(self, path , transform = None):
        super().__init__()

        self.env = lmdb.open(
            path,
            readonly=True,
        )
        self.txn = self.env.begin()
        

        self.transform = transform

    def __getitem__(self, index):
        key = f"index_{index}".encode("utf-8")
        value = self.txn.get(key)

        if value is None:
            raise IndexError(f"Index {index} not found in LMDB dataset")

        sample = pickle.loads(value)
        state, next_state, action, rewards, done= sample
        
        return state, next_state, action, rewards, done

    def __len__(self):
        length_key = b'__len__'
        length = self.txn.get(length_key)
        r = pickle.loads(length)
        return r

# class HybridOfflineDataset(Dataset):

#     def __init__(self, lmdb_path):
#         self.env = lmdb.open(
#             lmdb_path,
#             readonly=True,
#             lock=False,
#             readahead=False
#         )
#         with self.env.begin() as txn:
#             self.length = txn.stat()["entries"]

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         with self.env.begin() as txn:
#             key = f"{idx:08d}".encode()
#             sample = pickle.loads(txn.get(key))
#         return (
#                 sample["state"],
#                 sample["next_state"],
#                 sample["action"],
#                 sample["reward"],
#                 sample["done"]
#             )
class HybridOfflineDataset(Dataset):

    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        self.txn = None
        self.length = None

    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=512
        )
        self.txn = self.env.begin()
        self.length = self.txn.stat()["entries"]

    def __len__(self):
        if self.length is None:
            self._init_db()
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        key = f"{idx:08d}".encode()
        sample = pickle.loads(self.txn.get(key))

        return (
            sample["state"],
            sample["next_state"],
            sample["action"],
            sample["reward"],
            sample["done"]
        )


class ChunkedHybridOfflineDataset(Dataset):
    """
    LMDB dataset for chunked store produced by:
    LoadLmdb.load_offline_with_hybrid_lmdb_chunked_store

    Each LMDB value is a dict with keys:
      state_audio/state_rgb/state_depth/next_audio/next_rgb/next_depth/action/reward/done
    where each field is a tensor of shape [chunk_size, ...].
    """

    def __init__(
        self,
        lmdb_path,
        chunks_per_epoch=8,
        readahead=True,
        shuffle_chunks=True,
        log_chunk_loading=True,
        chunk_keys_cache_path=None,
        profile_chunk_time=False,
        chunk_replace_ratio=0.2,
    ):
        self.lmdb_path = lmdb_path
        self.chunks_per_epoch = int(chunks_per_epoch)
        self.readahead = bool(readahead)
        self.shuffle_chunks = bool(shuffle_chunks)
        self.log_chunk_loading = bool(log_chunk_loading)
        self.profile_chunk_time = bool(profile_chunk_time)
        self.chunk_replace_ratio = float(chunk_replace_ratio)
        self.chunk_keys_cache_path = (
            chunk_keys_cache_path
            if chunk_keys_cache_path is not None
            else os.path.join(self.lmdb_path, "__chunk_index_cache.pkl")
        )
        self.env = None
        self.txn = None
        self._owner_pid = None

        # all lmdb chunk keys and epoch pointer
        self._all_chunk_keys = []
        self._chunk_cursor = 0

        # in-memory working set for current epoch
        self._loaded_chunks = {}
        self._index_map = []  # list[(chunk_key, local_idx)]
        self._loaded_size = 0
        self._current_epoch_keys = []

    def _open_env(self):
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=self.readahead,
            max_readers=512,
        )
        self.txn = self.env.begin()
        self._owner_pid = os.getpid()

    def _init_db(self):
        self._open_env()
        self._all_chunk_keys = self._load_chunk_keys_cache()
        
        if self._all_chunk_keys is None:
            self._all_chunk_keys = self._scan_chunk_keys()
            self._save_chunk_keys_cache(self._all_chunk_keys)
        if len(self._all_chunk_keys) == 0:
            raise RuntimeError(f"No valid chunk entries found in lmdb: {self.lmdb_path}")
        if self.shuffle_chunks:
            random.shuffle(self._all_chunk_keys)
        self._chunk_cursor = 0
        init_keys = self._next_epoch_keys()
        self._apply_loaded_epoch(*self._load_chunks_by_keys(init_keys))

    def _chunk_keys_signature(self):
        sig = {"lmdb_path": os.path.abspath(self.lmdb_path)}
        if self.txn is not None:
            stat = self.txn.stat()
            sig["entries"] = int(stat.get("entries", 0))
        data_mdb = os.path.join(self.lmdb_path, "data.mdb")
        lock_mdb = os.path.join(self.lmdb_path, "lock.mdb")
        if os.path.exists(data_mdb):
            st = os.stat(data_mdb)
            sig["data_mdb_size"] = int(st.st_size)
            sig["data_mdb_mtime_ns"] = int(st.st_mtime_ns)
        if os.path.exists(lock_mdb):
            st = os.stat(lock_mdb)
            sig["lock_mdb_size"] = int(st.st_size)
            sig["lock_mdb_mtime_ns"] = int(st.st_mtime_ns)
        return sig

    def _load_chunk_keys_cache(self):
        if not os.path.exists(self.chunk_keys_cache_path):
            return None
        try:
            with open(self.chunk_keys_cache_path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("signature") != self._chunk_keys_signature():
                return None
            keys = payload.get("keys")
            if keys is None:
                # backward compatibility for old cache:
                # {'signature', 'index_map', 'chunk_to_indices', 'total_size'}
                if "chunk_to_indices" in payload:
                    keys = list(payload["chunk_to_indices"].keys())
                elif "index_map" in payload:
                    seen = OrderedDict()
                    for item in payload["index_map"]:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            seen[item[0]] = True
                    keys = list(seen.keys())
            return keys
        except Exception:
            return None

    def _save_chunk_keys_cache(self, keys):
        try:
            payload = {
                "signature": self._chunk_keys_signature(),
                "keys": keys,
            }
            with open(self.chunk_keys_cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _scan_chunk_keys(self):
        keys = []
        cursor = self.txn.cursor()
        for key, _ in cursor:
            keys.append(key)
        return keys

    def _load_chunk(self, txn, chunk_key):
        t0 = time.perf_counter()
        raw = txn.get(chunk_key)
        get_s = time.perf_counter() - t0
        if raw is None:
            return None, get_s, 0.0
        t1 = time.perf_counter()
        chunk = pickle.loads(raw)
        loads_s = time.perf_counter() - t1
        if "action" not in chunk:
            return None, get_s, loads_s
        return chunk, get_s, loads_s

    def _next_epoch_keys(self):
        if self.chunks_per_epoch <= 0:
            n = len(self._all_chunk_keys)
        else:
            n = min(self.chunks_per_epoch, len(self._all_chunk_keys))

        keys = []
        while len(keys) < n:
            if self._chunk_cursor >= len(self._all_chunk_keys):
                self._chunk_cursor = 0
                if self.shuffle_chunks:
                    random.shuffle(self._all_chunk_keys)
            keys.append(self._all_chunk_keys[self._chunk_cursor])
            self._chunk_cursor += 1
        return keys

    def _draw_next_keys_excluding(self, exclude_keys, k):
        if k <= 0:
            return []
        exclude = set(exclude_keys)
        out = []
        max_scan = max(len(self._all_chunk_keys) * 2, 1)
        scan = 0
        while len(out) < k and scan < max_scan:
            if self._chunk_cursor >= len(self._all_chunk_keys):
                self._chunk_cursor = 0
                if self.shuffle_chunks:
                    random.shuffle(self._all_chunk_keys)
            cand = self._all_chunk_keys[self._chunk_cursor]
            self._chunk_cursor += 1
            scan += 1
            if cand in exclude or cand in out:
                continue
            out.append(cand)
        return out

    def _build_next_epoch_keys(self):
        if len(self._current_epoch_keys) == 0:
            return self._next_epoch_keys()
        n = len(self._current_epoch_keys)
        replace_n = int(round(n * self.chunk_replace_ratio))
        replace_n = min(max(replace_n, 1), n)
        keep_n = n - replace_n
        if self.shuffle_chunks:
            keep_keys = random.sample(self._current_epoch_keys, keep_n)
        else:
            keep_keys = self._current_epoch_keys[:keep_n]
        new_keys = self._draw_next_keys_excluding(keep_keys, replace_n)
        next_keys = keep_keys + new_keys
        if self.shuffle_chunks:
            random.shuffle(next_keys)
        return next_keys

    def _load_chunks_by_keys(self, selected_keys):
        loaded_chunks = {}
        index_map = []
        total_get_s = 0.0
        total_pickle_s = 0.0
        total_extend_s = 0.0
        if self.log_chunk_loading:
            print(
                f"[ChunkLoad] selected_chunks={len(selected_keys)} "
                f"cursor={self._chunk_cursor}/{len(self._all_chunk_keys)}"
            )

        for i, chunk_key in enumerate(selected_keys, start=1):
            chunk, get_s, loads_s = self._load_chunk(self.txn, chunk_key)
            if chunk is None:
                if self.log_chunk_loading:
                    k = chunk_key.decode("utf-8", errors="ignore")
                    print(f"[ChunkLoad] skip {i}/{len(selected_keys)} key={k}")
                continue
            loaded_chunks[chunk_key] = chunk
            import pdb;pdb.set_trace()
            chunk_len = int(chunk["action"].shape[0])
            t2 = time.perf_counter()
            index_map.extend((chunk_key, local_idx) for local_idx in range(chunk_len))
            extend_s = time.perf_counter() - t2
            total_get_s += get_s
            total_pickle_s += loads_s
            total_extend_s += extend_s

            if self.log_chunk_loading:
                k = chunk_key.decode("utf-8", errors="ignore")
                if self.profile_chunk_time:
                    print(
                        f"[ChunkLoad] load {i}/{len(selected_keys)} key={k} samples={chunk_len} "
                        f"txn_get={get_s:.4f}s pickle={loads_s:.4f}s index_extend={extend_s:.4f}s"
                    )
                else:
                    print(f"[ChunkLoad] load {i}/{len(selected_keys)} key={k} samples={chunk_len}")

        loaded_size = len(index_map)
        if self.log_chunk_loading:
            if self.profile_chunk_time:
                print(
                    f"[ChunkLoad] ready chunks={len(loaded_chunks)} samples={loaded_size} "
                    f"sum_txn_get={total_get_s:.4f}s sum_pickle={total_pickle_s:.4f}s "
                    f"sum_index_extend={total_extend_s:.4f}s"
                )
            else:
                print(f"[ChunkLoad] ready chunks={len(loaded_chunks)} samples={loaded_size}")
        return selected_keys, loaded_chunks, index_map, loaded_size

    def _apply_loaded_epoch(self, selected_keys, loaded_chunks, index_map, loaded_size):
        self._current_epoch_keys = list(selected_keys)
        self._loaded_chunks = loaded_chunks
        self._index_map = index_map
        self._loaded_size = loaded_size

    def on_epoch_end(self):
        if self.env is None:
            self._init_db()
            return
        next_keys = self._build_next_epoch_keys()
        self._apply_loaded_epoch(*self._load_chunks_by_keys(next_keys))

    def get_chunk_groups(self):
        if self.env is None:
            self._init_db()
        groups = defaultdict(list)
        for gidx, (chunk_key, _) in enumerate(self._index_map):
            groups[chunk_key].append(gidx)
        return groups

    def __len__(self):
        if self.env is None:
            self._init_db()
        return self._loaded_size

    def __getitem__(self, idx):
        if self.env is None or self._owner_pid != os.getpid():
            self._open_env()
            if len(self._all_chunk_keys) == 0:
                self._all_chunk_keys = self._load_chunk_keys_cache()
                if self._all_chunk_keys is None:
                    self._all_chunk_keys = self._scan_chunk_keys()
                if self.shuffle_chunks:
                    random.shuffle(self._all_chunk_keys)
                self._chunk_cursor = 0
            if self._loaded_size == 0:
                init_keys = self._next_epoch_keys()
                self._apply_loaded_epoch(*self._load_chunks_by_keys(init_keys))

        chunk_key, local_idx = self._index_map[idx]
        chunk = self._loaded_chunks.get(chunk_key)
        if chunk is None:
            # Fallback: should rarely happen unless external mutation.
            chunk, _, _ = self._load_chunk(self.txn, chunk_key)
            if chunk is None:
                raise IndexError(f"Chunk key not found in lmdb: {chunk_key!r}")
            self._loaded_chunks[chunk_key] = chunk

        state = (
            chunk["state_audio"][local_idx],
            chunk["state_rgb"][local_idx],
            chunk["state_depth"][local_idx],
        )
        next_state = (
            chunk["next_audio"][local_idx],
            chunk["next_rgb"][local_idx],
            chunk["next_depth"][local_idx],
        )
        action = chunk["action"][local_idx]
        reward = chunk["reward"][local_idx]
        done = chunk["done"][local_idx]
        return state, next_state, action, reward, done


class ChunkAwareBatchSampler(Sampler):
    """
    Batch sampler that groups indices by chunk to reduce txn.get frequency.
    """

    def __init__(self, dataset: ChunkedHybridOfflineDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

    def __iter__(self):
        chunk_groups = self.dataset.get_chunk_groups()
        chunk_keys = list(chunk_groups.keys())
        random.shuffle(chunk_keys)

        for chunk_key in chunk_keys:
            indices = list(chunk_groups[chunk_key])
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        chunk_groups = self.dataset.get_chunk_groups()
        total_batches = 0
        for indices in chunk_groups.values():
            n = len(indices)
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += (n + self.batch_size - 1) // self.batch_size
        return total_batches


class ChunkWindowBatchSampler(Sampler):
    """
    Keep only a small window of chunks "active" in memory during an epoch.
    All samples are still consumed once per epoch, but chunk switching becomes
    smoother and memory usage is bounded by window size.
    """

    def __init__(
        self,
        dataset: ChunkedHybridOfflineDataset,
        batch_size: int,
        window_chunks: int = 8,
        drop_last: bool = False,
        log_window: bool = False,
        log_every_batches: int = 100,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.window_chunks = int(window_chunks)
        self.drop_last = drop_last
        self.log_window = bool(log_window)
        self.log_every_batches = int(log_every_batches)

    def __iter__(self):
        chunk_groups = self.dataset.get_chunk_groups()
        chunk_keys = list(chunk_groups.keys())
        random.shuffle(chunk_keys)

        if self.window_chunks <= 0:
            self.window_chunks = 1

        active = []
        next_chunk_ptr = 0
        emitted_batches = 0
        finished_chunks = 0

        def _push_next_chunk():
            nonlocal next_chunk_ptr
            if next_chunk_ptr >= len(chunk_keys):
                return
            key = chunk_keys[next_chunk_ptr]
            next_chunk_ptr += 1
            idxs = list(chunk_groups[key])
            random.shuffle(idxs)
            active.append([key, idxs, 0])  # key, indices, cursor
            if self.log_window:
                print(
                    f"[ChunkWindow] add chunk={key.decode('utf-8', errors='ignore')} "
                    f"samples={len(idxs)} active={len(active)}"
                )

        while len(active) < self.window_chunks and next_chunk_ptr < len(chunk_keys):
            _push_next_chunk()

        if self.log_window:
            print(
                f"[ChunkWindow] start epoch chunks={len(chunk_keys)} "
                f"window={self.window_chunks} batch_size={self.batch_size}"
            )

        while active:
            # Randomly sample from current active window for better mixing.
            slot = random.randrange(len(active))
            key, idxs, cursor = active[slot]
            batch = idxs[cursor: cursor + self.batch_size]

            if len(batch) < self.batch_size and self.drop_last:
                # Mark this chunk as exhausted and replace it.
                active.pop(slot)
                _push_next_chunk()
                continue

            if len(batch) > 0:
                emitted_batches += 1
                if self.log_window and self.log_every_batches > 0 and emitted_batches % self.log_every_batches == 0:
                    print(
                        f"[ChunkWindow] batches={emitted_batches} "
                        f"active={len(active)} next_chunk_ptr={next_chunk_ptr}/{len(chunk_keys)} "
                        f"finished_chunks={finished_chunks}"
                    )
                yield batch

            cursor += len(batch)
            if cursor >= len(idxs):
                finished_chunks += 1
                active.pop(slot)
                _push_next_chunk()
            else:
                active[slot][2] = cursor

    def __len__(self):
        chunk_groups = self.dataset.get_chunk_groups()
        total_batches = 0
        for indices in chunk_groups.values():
            n = len(indices)
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += (n + self.batch_size - 1) // self.batch_size
        return total_batches

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
    def __init__(self, train_json,  attention = False , preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/pt/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        self.shard_files = []
        self.use_attention = attention
        self.get_files(train_json)
        # for pattern in shard_pattern:
        #     lists_ = glob.glob(pattern)
        #     random.shuffle(lists_)
        #     import pdb;pdb.set_trace()
        #     self.shard_files.extend(lists_[:40])
        assert len(self.shard_files) > 0, f"No shards found at {train_json}"

        self.preload = preload
        self.shards = []   # 存 torch.load 的结果（如果 preload=True）
        self.shard_sizes = []  # 每个 shard 的样本数
        self.index_map = []    # 全局 index → (shard_id, local_index)

        # 扫描每个 shard
        for shard_id, shard_file in enumerate(self.shard_files):
            print(f"scan shard id {shard_id} ...")
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
        print(f"Total samples: {self.total_size}")

    def get_files(self , train_json):
        import json
        with open(train_json , 'r') as f:
            train_json_data = json.load(f) 
        path = train_json_data['path']
        split = train_json_data['split']
        for beta in path.keys():
            for distance in path[beta].keys():
                print(path[beta][distance])
                lists_1 = glob.glob(path[beta][distance])
                random.shuffle(lists_1)
                self.shard_files.extend(lists_1[:split[beta][distance]])
                # self.shard_files.extend(lists_1)

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
