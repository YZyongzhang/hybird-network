# dataloader
from torch.utils.data import DataLoader , Dataset
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
from network.foundation_model import Network
import glob
class LoadLmdb:

    def __init__(self, path):

        self.paths = self._deal_path(path)
        
        
    @classmethod
    def get_files(cls , path):
        import os
        # 获取到object的路径
        object_name_files = []
        parents = [os.path.join(path , i) for i in os.listdir(path = path)]
        for path in parents:
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

        buffer_visuals, buffer_audios, buffer_actions , buffer_angles = [], [], [] , []

        for file in tqdm(files):
            with open(file, 'rb') as f:
                data = pickle.load(f)
         
            obs = data['obs']
            

            action_id = data['action_id']
            action_id = np.array(action_id).reshape(-1).tolist()
            for v, a  , info in zip(obs[1:-1], action_id[1:] , data['info']):
                # if info['distance_to_goal'] > 8.0:
                #     tqdm.write(f"distence is {info['distance_to_goal']} , drop")
                #     continue
                visual = torch.from_numpy(v['rgb']).float() / 255.0
                audio = torch.from_numpy(v['spectrogram'][0]).float()
                
                action = torch.tensor(a, dtype=torch.long)
                angel = np.degrees(v['angle'][1])
                buffer_visuals.append(visual)
                buffer_audios.append(audio)
                buffer_actions.append(action)
                buffer_angles.append(torch.tensor(angel))
                # 写一个 shard
                if len(buffer_visuals) >= shard_size:
                    torch.save({
                        'visuals': torch.stack(buffer_visuals),
                        'audios': torch.stack(buffer_audios),
                        'actions': torch.stack(buffer_actions),
                        'angles':torch.stack(buffer_angles)
                    }, f"{config.LMDB.TO_PATH}/foundation_model_shard_{shard_id}.pt")

                    print(f"保存 shard {shard_id}, size={len(buffer_visuals)}")

                    buffer_visuals, buffer_audios, buffer_actions , buffer_angles = [], [], [] , []
                    shard_id += 1

        # 保存最后一个不满 shard 的数据
        if buffer_visuals:
            torch.save({
                'visuals': torch.stack(buffer_visuals),
                'audios': torch.stack(buffer_audios),
                'actions': torch.stack(buffer_actions),
                'angles':torch.stack(buffer_angles)
            }, f"{config.LMDB.TO_PATH}/foundation_model_shard_{shard_id}.pt")
            print(f"保存 shard {shard_id}, size={len(buffer_visuals)}")
    
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

            # 遍历每一对 (state, next_state)
            for i in range(len(obs) - 1):
                v_now = obs[i]
                v_next = obs[i + 1]
                a = action_id[i]
                r = rewards[i]
                d = dones[i]

                # 提取特征
                visual_now = torch.from_numpy(v_now['rgb']).float() / 255.0
                audio_now = torch.from_numpy(v_now['spectrogram'][0]).float()
                visual_next = torch.from_numpy(v_next['rgb']).float() / 255.0
                audio_next = torch.from_numpy(v_next['spectrogram'][0]).float()

                # 编码成状态向量
                with torch.no_grad():
                    state = model.embedding_forward(audio_now.to(model.device), visual_now.to(model.device))
                    next_state = model.embedding_forward(audio_next.to(model.device), visual_next.to(model.device))

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


class ShardedPTDataset(Dataset):
    def __init__(self, shard_pattern, preload=True):
        super().__init__()
        self.shard_files = sorted(glob.glob(shard_pattern))
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

        visual = data["visuals"][local_idx]
        audio = data["audios"][local_idx]
        action = data["actions"][local_idx]
        angle = data['angles'][local_idx]
        std_audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        return  std_audio, visual , angle , action
      
class ShardedPTDatasetOffline(Dataset):
    def __init__(self, shard_pattern="./dataset/pt/offline/offline_model_shard_*.pt", preload=True):
        """
        shard_pattern: shard 文件路径模式，比如 ./dataset/pt/foundation_model_shard_*.pt
        preload: 是否把所有 shard 一次性加载到内存（大数据集建议 False）
        """
        super().__init__()
        self.shard_files = sorted(glob.glob(shard_pattern))
        assert len(self.shard_files) > 0, f"No shards found at {shard_pattern}"

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

        # 取出一个 transition
        state       = data["states"][local_idx]
        next_state  = data["next_states"][local_idx]
        action      = data["actions"][local_idx]
        reward      = data["rewards"][local_idx]
        done        = data["dones"][local_idx]
        state = state.squeeze(0)
        next_state = next_state.squeeze(0)
        return state, next_state , action, reward, done