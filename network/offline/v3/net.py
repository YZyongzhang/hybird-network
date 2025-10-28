import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from network.hybird.foundation_model import Network

class hybrid_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size,stack_frames,lstm_out,lstm_layer):
        super(hybrid_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.stack_frames = stack_frames
        self.hybrid = Network()
        self.hybrid_dim = 256
        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.hybrid_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """
        audio , rgb , depth = input
        rgb = rgb.permute(0,1 , 4, 2, 3)
        depth = depth.permute(0,1 , 4, 2, 3)

        audio = (audio - audio.mean()) / (audio.std() + 1e-6)

        # if input.shape[1]>1:
        batch_size = rgb.shape[0]
        seq_len = rgb.shape[1]
        rgb = rgb.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        depth = depth.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        audio = audio.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])

        x=self.hybrid(audio , rgb ,depth)
        x=x.reshape(x.shape[0],-1)
        x=x.reshape(batch_size,seq_len,-1)
        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        audio , rgb , depth = input
        rgb = rgb.permute(0,1 , 4, 2, 3)
        depth = depth.permute(0,1 , 4, 2, 3)
        audio = (audio - audio.mean()) / (audio.std() + 1e-6)

        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        x=self.hybrid(audio , rgb ,depth)
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        # log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(2, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

