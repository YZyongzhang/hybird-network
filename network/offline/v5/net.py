import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from network.hybird.foundation_model import Network
from torch.distributions import Categorical 
import math
class Attention(nn.Module):
    def __init__(self  , input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.k = nn.Linear(self.input_dim , self.input_dim)
        self.q = nn.Linear(self.input_dim , self.input_dim)
        self.v = nn.Linear(self.input_dim , self.input_dim)
        self.scale = 1.0 / math.sqrt(input_dim)

    def forward(self , audio_token , visual_audio_token):

        assert len(audio_token.shape) == 3 # batch , seq, embedding
        assert len(visual_audio_token.shape)  == 3 
        b , t , d = audio_token.shape
        pair = torch.stack([audio_token, visual_audio_token], dim=2)  # [B, T, 2, D]

        pair_bt = pair.view(b * t, 2, d)

        Q = self.q(pair_bt)
        K = self.k(pair_bt)
        V = self.v(pair_bt)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)
        out = out.view(b, t, 2 * d)
        return out
    
class hybrid_LSTM(nn.Module):
    def __init__(self, state_size, action_size,lstm_out,lstm_layer):
        super(hybrid_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.hybrid_dim = 128
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

        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(input,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        

        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(input)

        else:
            x, (ht, ct) = self.lstm(input,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
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
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size, seed=1):
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
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

