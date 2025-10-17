import sys
 
from module import *

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
    def __init__(self,input_dim , hidden_dim , output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim , self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self,embeding):
        x1 = self.fc1(embeding)
        x2 = self.dropout(x1)
        x3 = self.fc2(x2)
        return x1 ,x2 , x3
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.vitpartnet = ViTEncoder()
        # self.mermory = Mermory(input_dim=128,hidden_dim=128)
        self.finnal = Finnal_model(input_dim = 512 , hidden_dim=32 , output_dim=4)


    def forward(self,audio , visual):
        audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        visual = visual.permute(0,3, 1, 2)
        avencoder= self.vitpartnet(audio , visual)

        x1 , x2 ,finnal_output = self.finnal(avencoder)
        return finnal_output
        # return attentioned
