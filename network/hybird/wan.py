import sys
 
import torch
import torch.nn as nn
from network.hybird.ViT import ViTEncoder
from network.hybird.audio import AudioCRNN
from network.hybird.ViT import VisualEncoder , PositionalEcoder
from network.hybird.Encoder import Encoder
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
    
class Final_model(nn.Module):
    def __init__(self, input_dim , hidden_dim , output_dim, sequence_length=5, action_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim , self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.action_sequence = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.sequence_length * self.action_dim)
        )

    def forward(self,encoder):
        x1 = self.fc1(encoder)
        x2 = self.dropout(x1)
        action_sequence = self.action_sequence(x2)
        return action_sequence.view(-1, self.sequence_length, self.action_dim)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTEncoder()
        self.final = Final_model(input_dim = 512 , hidden_dim= 256 , output_dim=4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def forward(self,audio , rgb , depth):
        visual = torch.cat([rgb , depth], dim=-1)
        visual = visual.permute(0, 3, 1, 2)
        audio = audio.permute(0 , 3 , 1, 2)
        embedding = self.vit(audio , visual)
        action_sequence = self.final(embedding)
        return action_sequence
