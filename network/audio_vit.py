import torch.nn as nn
from network.Encoder import Encoder
import torch
class PositionalEcoder(nn.Module):
    def __init__(self, max_len = 10000, d_models = 128 , device = None):
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.d_models = d_models
        # 优先构造出整个位置编码矩阵
        self.position_matrix = torch.zeros(max_len , self.d_models , device=self.device)

        self.position_matrix.requires_grad = False # 1. 后续相加的时候生成的向量还会有梯度？后续相加后有梯度，但是位置编码不参与反向传播。

        # 位置编码矩阵参数
        self.pos = torch.arange(0 , self.max_len , step= 1).unsqueeze(1)
        self._2k = torch.arange(0 , self.d_models , step=2)

        # 填充位置编码矩阵
        self.position_matrix[:,0::2] = torch.sin(self.pos * (1000 ** (self._2k / self.d_models)))
        self.position_matrix[:,1::2] = torch.cos(self.pos * (1000 ** (self._2k / self.d_models)))
        
    def forward(self, x :torch.tensor) -> torch.tensor:
        batch,  len ,encode_dim ,= x.shape
        return self.position_matrix[:len,:]

class AudioEncoder(nn.Module):
    def __init__(self,   input_dim = 2, output_dim = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv2d_1 = nn.Conv2d(in_channels=self.input_dim , out_channels= self.output_dim, kernel_size=16 , stride=16 , padding=1)
        
        self.audio_transformer_encoder = Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)
        self.positional = PositionalEcoder()
    def forward(self, mel_audio):
        # mel_audio = self._deal_audio(audio)
        
        # print(mel_audio.shape)
        
        audio_cnn = self.conv2d_1(mel_audio)
        a_batch , a_dim , a_h , a_w = audio_cnn.shape
        audio_cnn = audio_cnn.reshape(a_batch ,   a_h * a_w  , a_dim)
        
        a_position = self.positional(audio_cnn)
        audio_encoder = audio_cnn + a_position
        # print(x_1.shape)# (batch , 16 , 32 , 16)
        return audio_encoder
    
class AngleProdict(nn.Module):
    def __init__(self, hidden_dim = 128):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*128 , hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim , 8)
        )

    def forward(self, x:torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        audio_encoder = self.audio_encoder(x)
        x2 = self.fc1(audio_encoder)
        return x2
