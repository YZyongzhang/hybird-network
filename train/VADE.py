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

class LoadLmdb:

    def __init__(self, path):

        self.paths = self._deal_path(path)


    @classmethod
    def load_lmdb(cls,paths):
        
        # 这里看需不需要使用这个dealpath ，也可以在外部处理，如果文件层级简单的话推荐使用内部的
        # paths = cls._deal_path(paths)
        dataset_name = input("please input the dataset name:")
        env = lmdb.open(f"./experiment/dataset/{dataset_name}", map_size=1024*1014*1024*30)
        sum_index = 0
        txn = env.begin(write=True) 
        eps = 0.9 # 90%的概率丢弃lable为0的数据
        for path_i in tqdm(paths):
            with open(path_i ,'rb') as f:
                
                data = pickle.load(f)

                audio , visual , action  = cls.get_values_tuple(data)
                for index in range(len(audio)):
                    
                    if action[index] == 0 and random.random() < eps:
                        tqdm.write((f"丢去0 label"))
                        continue
                    key = f'index_{sum_index}'.encode('utf-8')
                    values = (audio[index],visual[index] , action[index])
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
        print(f"dataset location the ./experiment/dataset/{dataset_name}")

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
    def get_values_tuple(cls, data):
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
        visual , audio  , action = sample

        visual , audio= self._deal_datas(visual,audio)

        return visual ,audio  ,action

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
        
    # def audio_mask(self , audio):
    #     """
    #     输入:
    #         audio: np.ndarray of shape (2, length)
    #     返回:
    #         masked: shape (2, length)，只保留被 mask 的通道，其他为 0
    #         keeped: shape (2, length)，只保留未被 mask 的通道，其他为 0
    #     """
    #     assert audio.shape[0] == 2, "输入必须是双通道音频"
    #     length = audio.shape[1]

    #     c = random.randint(0, 1)  # 0 或 1，表示被 mask 的通道

    #     masked = np.zeros_like(audio)
    #     keeped = np.zeros_like(audio)

    #     masked[c] = audio[c]          # 被 mask 的通道值复制
    #     keeped[1 - c] = audio[1 - c]  # 未被 mask 的通道值复制

    #     return masked, keeped



######
#test#
######


def get_lmdb(path):

    # get 到所有的pickle 文件
    files = get_files(path)
    # 写入数据库数据
    LoadLmdb.load_lmdb(files)
    
def get_files(path):
    import os
    # 获取到object的路径
    object_name_files = [os.path.join(path , i ) for i in os.listdir(path=path) if i != "a.md"]

    return object_name_files

if __name__ == "__main__":
    data_path = './experiment/data/soundspaces_val'
    get_lmdb(path=data_path)