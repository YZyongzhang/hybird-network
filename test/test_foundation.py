import pickle
files_path = './experiment/data/soundspaces'
import os
files = [os.path.join(files_path , i) for i in os.listdir(files_path) if i != "a.md"]
import pickle
with open(files[0] , 'rb') as f:
    data = pickle.load(f)
import librosa
import numpy as np
import torch
def mel_audio(audio):
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
visual = np.array(data[0][0]['camera'])
audio = np.array(data[0][0]['audio'])
audio = audio.astype(float) / 32768.0
visual = visual[:,:,:,:-1]
image_tensor = torch.tensor(np.array(visual))
audio_list = list()
for i in range(len(audio)):
    audio_tensor_i = mel_audio(audio[i])
    audio_list.append(audio_tensor_i)

audio_tensor = torch.stack(audio_list , dim=0)
device = torch.device('cuda')
model_path = './experiment/train/ckpt/model_epoch_800.pth'
import sys
sys.path.append("/home/getuanhui/project/finnal_exp")
from network.foundation_model import Network
model = Network().to(device)
model.load_state_dict(torch.load(model_path))
output , avencoder , x1 , x2 = model(audio_tensor.to(device),image_tensor.to(device))
import pdb;pdb.set_trace()
torch.argmax(output,dim=1)
data[0][0]['rl_pred']