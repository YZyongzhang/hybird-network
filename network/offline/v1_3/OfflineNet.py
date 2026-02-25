import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm   
from network import HybirdNetwork
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.down = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17 * 128 , 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
    def forward(self, x):
        if len(x.shape) < 2: # batch , dim
            x = x.unsqueeze(0)
        # x = self.down(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.down = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17 * 128 , 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
    def forward(self, x):
        # x = self.down(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
class SAC_Hybird_model(torch.nn.Module):
    ''' 处理离散动作的SAC算法 '''
    def __init__(self,  state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, beta ,device):
        super(SAC_Hybird_model, self).__init__()
        self.hybird = HybirdNetwork().to(device)
        self.hybird.load_state_dict(torch.load("heard_unheard/heard/hybird_ckpt/model_epoch_100.pth"))
        self.hybird.train()
        self.hybird_optimizer = torch.optim.Adam(self.hybird.parameters(),
                            lr= actor_lr / 10)
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        # self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float) # 能跑出spl0.3的版本
        self.log_alpha = torch.tensor(np.log(1), dtype=torch.float) # 改成这个，可以跑出spl0.33
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.beta = beta  # CQL的超参数
        self.clip_grad_param = 1

    def get_action(self, state):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        audio , rgb , depth = state
        state = self.hybird.embedding_forward(audio.to(self.device) , rgb.to(self.device) , depth.to(self.device)).float()
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)

        # 因为这里要追求最大化熵
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value.squeeze(1) * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, states, actions, rewards, next_states, dones):
        audio , rgb , depth = states
        audio_next , rgb_next , depth_next = next_states
        states = self.hybird.embedding_forward(audio.to(self.device) , rgb.to(self.device) , depth.to(self.device)).float()
        next_states = self.hybird.embedding_forward(audio_next.to(self.device) , rgb_next.to(self.device) , depth_next.to(self.device)).float()

        # states = states.float().to(self.device)
        # next_states = next_states.float().to(self.device)
        rewards = rewards.float().to(self.device)
        # import pdb ; pdb.set_trace()
        dones = dones.float().to(self.device)
        actions = actions.long().to(self.device)
        actions = actions.unsqueeze(1)  # 确保动作是二维的

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        # actor_loss = (probs * (self.log_alpha.exp().to(self.device) * log_probs - min_qvalue )).sum(1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        # alpha_loss = - (self.log_alpha.exp() * (log_probs.cpu() + self.target_entropy).detach().cpu()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)


        critic_1_q_values = self.critic_1(states)
        critic_1_q_values_ = critic_1_q_values.gather(1, actions).squeeze(1)
        
        
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values_, td_target.detach()))

        critic_2_q_values = self.critic_2(states)
        critic_2_q_values_ = critic_2_q_values.gather(1, actions).squeeze(1)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values_, td_target.detach()))

        cql1_scaled_loss = torch.logsumexp(critic_1_q_values, dim=1).mean() - critic_1_q_values.mean()
        cql2_scaled_loss = torch.logsumexp(critic_2_q_values, dim=1).mean() - critic_2_q_values.mean()

        # cql1_scaled_loss = torch.logsumexp(critic_1_q_values, dim=1).mean() - critic_1_q_values_.mean()
        # cql2_scaled_loss = torch.logsumexp(critic_2_q_values, dim=1).mean() - critic_2_q_values_.mean()
        
        cql_1_loss = critic_1_loss + self.beta * cql1_scaled_loss
        cql_2_loss = critic_2_loss + self.beta * cql2_scaled_loss
        self.critic_1_optimizer.zero_grad()
        cql_1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_param)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        # self.hybird_optimizer.zero_grad()
        cql_2_loss.backward()
        clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_param)
        # self.hybird_optimizer.step()
        self.critic_2_optimizer.step()



        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        return {
            'critic1_loss': critic_1_loss.item(),
            'critic2_loss': critic_2_loss.item(),
            'cql1_scaled_loss': cql1_scaled_loss.item(),
            'cql2_scaled_loss': cql2_scaled_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'cql_1_loss': cql_1_loss.item(),
            'cql_2_loss': cql_2_loss.item(),
            'entropy': entropy.mean().item(),
            'alpha': self.log_alpha.exp().item()
        }