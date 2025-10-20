import sys
 
import torch
import torch.nn as nn
from network.ViT import ViTEncoder
from network.audio import AudioCRNN
from network.ViT import VisualEncoder , PositionalEcoder
from network.Encoder import Encoder
class Attention(nn.Module):
    def __init__(self , input_dim , visual_dim , audio_dim ,hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.fc1 = nn.Linear(input_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


        self.attention_weight = nn.Parameter(torch.randn(self.output_dim) * 0.01)
        
        
    def forward(self, encode):
        x1 = torch.relu(self.fc1(encode))
        x2 = torch.relu(self.fc2(x1))
        self.attention_weight.unsqueeze(1)
        # import pdb;pdb.set_trace()
        # y = torch.matmul(x2.unsqueeze(-1) , self.attention_weight)
        y = x2 * self.attention_weight
        return x2
    
class Vote(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        
        self.voter = nn.Parameter(torch.randn(self.input_dim , 32) * 0.01)

    def forward(self,embeding):
        output = torch.matmul(embeding , self.voter)
        return output
    
class Decision(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(Decision, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_matrix):
        x = input_matrix.squeeze(1)
        # import pdb;pdb.set_trace()
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Finnal_model(nn.Module):
    def __init__(self, input_dim , hidden_dim , output_dim):
        super().__init__()
        self.input_dim = 17*128
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim , self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 4)
        )

    def forward(self,encoder):
        x1 = self.fc1(encoder)
        x2 = self.dropout(x1)
        x3 = self.fc2(x2)
        return x3

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # self.vitpartnet = ViTEncoder()
        self.visual_encoder = VisualEncoder()
        self.add_position = PositionalEcoder()
        self.visual_transformer_encoder =  Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)
        self.audio_encoder = AudioCRNN()
        path = './HybirdNetworkCkpt/audio_part/ckpt/audio/model_epoch_best.pth'
        self.audio_encoder.load_state_dict(torch.load(path) ,  strict=True)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            
            param.requires_grad = False

        self.vaencoder =  Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)
        self.final = Finnal_model(input_dim = 512 , hidden_dim= 256 , output_dim=4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def forward(self,audio , visual):
        if len(audio.shape) == 3:
            audio = audio.unsqueeze(0)
        if len(visual.shape) == 3:
            visual = visual.unsqueeze(0)
            
        audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        visual = visual.permute(0,3, 1, 2)
        
        with torch.no_grad():
            audio_encoder = self.audio_encoder.encoder_forward(audio)
        visual_cnn = self.visual_encoder(visual)
        v_batch , v_dim , v_h , v_w = visual_cnn.shape
        visual_cnn = visual_cnn.reshape(v_batch , v_h * v_w  , v_dim)
        v_position = self.add_position(visual_cnn)
        visual_cnn = visual_cnn + v_position
        visual_encoder = self.visual_transformer_encoder(visual_cnn)
        concat_encoder = torch.cat((audio_encoder , visual_encoder ) , dim=1)
        share_visual_audio_encoder = self.vaencoder(concat_encoder)
        p_share_encoder = self.add_position(share_visual_audio_encoder)
        share_encoder = share_visual_audio_encoder + p_share_encoder

        finnal_output = self.final(share_encoder)
        return finnal_output
        # return attentioned
    
    def embedding_forward(self, audio , visual):
        if len(audio.shape) == 3:
            audio = audio.unsqueeze(0)
        if len(visual.shape) == 3:
            visual = visual.unsqueeze(0)
        
        audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        visual = visual.permute(0,3, 1, 2)
        
        audio_encoder = self.audio_encoder.encoder_forward(audio)
        visual_cnn = self.visual_encoder(visual)
        v_batch , v_dim , v_h , v_w = visual_cnn.shape
        visual_cnn = visual_cnn.reshape(v_batch , v_h * v_w  , v_dim)
        v_position = self.add_position(visual_cnn)
        visual_cnn = visual_cnn + v_position
        visual_encoder = self.visual_transformer_encoder(visual_cnn)
        concat_encoder = torch.cat((audio_encoder , visual_encoder ) , dim=1)
        share_visual_audio_encoder = self.vaencoder(concat_encoder)
        p_share_encoder = self.add_position(share_visual_audio_encoder)
        share_encoder = share_visual_audio_encoder + p_share_encoder
        embedding = self.final.fc1(share_encoder)
        embedding.squeeze(0)
        return embedding
        # return attentioned