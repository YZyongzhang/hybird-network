"""
利用offline RL进行训练。
"""
import sys
import torch
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from utils.log import logger
class OfflineTrain:
    def __init__(self, dataloader , sac_model, writer , online_test , batch_size, epoch , save_dir ,  device='cuda'):
        self.device = device
        self.agent = sac_model
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.writer = writer
        self.epoch = epoch
        self.online_test = online_test
        self.save_dir = save_dir
    def train(self):
        
        global_step = 0
        for epoch in tqdm(range(1, self.epoch + 1),desc="epoch nums"):

            for batch in self.dataloader:
                
                global_step += 1
                states , next_states , actions , rewards , dones = batch
                loss_dict = self.agent.update(states, actions , rewards, next_states ,  dones)


                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)

                tqdm.write(f"step:{global_step} ,critic_1_loss:{loss_dict['critic_1_loss']} , critic_2_loss : {loss_dict['critic_2_loss']} , actor_loss : {loss_dict['actor_loss']} , alpha_loss:{loss_dict['alpha_loss']}")

            if epoch % 10 == 0 :
                torch.save(self.agent.state_dict() , f'{self.save_dir}/sac_2level_model_{epoch}.pth')
            if epoch % 10 == 0:
                # train_acc = self.val(epoch)
                self.agent.eval()
                online_reward  , spl = self.online_test.rollout(self.agent.actor ,logger )
                self.agent.train()
                # self.writer.add_scalar("Val/train_Accuracy", train_acc, global_step=epoch)
                self.writer.add_scalar("Val/online_reward", online_reward, global_step=epoch)
                self.writer.add_scalar("Val/spl", spl, global_step=epoch)
                

    def val(self, epoch):
        self.agent.eval() 

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:  
                states, next_states, actions, rewards, dones = batch
                states = states.to('cuda')
                actions = actions.to('cuda')

                actions = actions.long()  
                a_predicted_logits = self.agent.actor(states)  

                pred_classes = torch.argmax(a_predicted_logits, dim=1)   # 预测类别
                correct = (pred_classes == actions).sum().item()         # 预测正确的数量
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.agent.train()
        return acc