import sys
import torch
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from utils.log import logger
class OfflineTrain:
    def __init__(self, dataset , dataloader , sac_model, writer , online_test ,online_test_epoch ,  batch_size, epoch , save_dir ,  config ,device='cuda'):
        self.device = device
        self.agent = sac_model
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.writer = writer
        self.epoch = epoch
        self.online_test = online_test
        self.online_test_epoch = online_test_epoch
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
                tqdm.write(f"actor_loss: {loss_dict['actor_loss']} , critic1_loss:{loss_dict['critic1_loss']} , critic2_loss:{loss_dict['critic2_loss']}")
            if epoch % 1 == 0 :
                torch.save(self.agent.state_dict() , f'{self.save_dir}/sac_2level_model_{epoch}.pth')
            if epoch % self.online_test_epoch== 0: # 可以设置一个非常大的数进行调整曲线不进行在线测试，或者设置成使用acc进行简单的判断
                # train_acc = self.val(epoch)
                self.agent.eval()
                online_reward  , spl = self.online_test.rollout(epoch , self.agent ,logger )
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
                a_predicted_logits = self.agent.get_action(states)  

                pred_classes = torch.argmax(a_predicted_logits, dim=1)   # 预测类别
                correct = (pred_classes == actions).sum().item()         # 预测正确的数量
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.agent.train()
        return acc
    
class OfflineTrainBuffer:
    def __init__(self, dataset , dataloader , sac_model, writer , online_test ,online_test_epoch ,  batch_size, epoch , save_dir , config, device='cuda'):
        self.device = device
        self.agent = sac_model
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataloader = dataloader
        self.writer = writer
        self.epoch = epoch
        self.online_test = online_test
        self.online_test_epoch = online_test_epoch
        self.save_dir = save_dir
        self.config = config
    def train(self):
        
        global_step = 0
        for epoch in tqdm(range(1, self.epoch + 1),desc="epoch nums"):
            if epoch % self.config.replay_epoch == 0 and epoch != 0 and epoch < 10:
                self.dataset.replay(logger)
                self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True ,  pin_memory=True)
            for batch in self.dataloader:
                
                global_step += 1
                states , next_states , actions , rewards , dones = batch

                loss_dict = self.agent.update(states, actions , rewards, next_states ,  dones)

                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)
                tqdm.write(f"actor_loss: {loss_dict['actor_loss']} , critic1_loss:{loss_dict['critic1_loss']} , critic2_loss:{loss_dict['critic2_loss']}")
            if epoch % 1 == 0 :
                torch.save(self.agent.state_dict() , f'{self.save_dir}/sac_2level_model_{epoch}.pth')
            if epoch % self.online_test_epoch== 0:
                # train_acc = self.val(epoch)
                self.agent.eval()
                online_reward  , spl = self.online_test.rollout(epoch , self.agent ,logger )
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
                a_predicted_logits = self.agent.get_action(states)  

                pred_classes = torch.argmax(a_predicted_logits, dim=1)  
                correct = (pred_classes == actions).sum().item()        
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.agent.train()
        return acc
    
class OfflineAndHybird:
    def __init__(self, dataset , dataloader , sac_model, writer , online_test ,online_test_epoch ,  batch_size, epoch , save_dir , config, device='cuda'):
        self.device = device
        self.agent = sac_model
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataloader = dataloader
        self.writer = writer
        self.epoch = epoch
        self.online_test = online_test
        self.online_test_epoch = online_test_epoch
        self.save_dir = save_dir
        self.config = config
    def train(self):
        
        global_step = 0
        for epoch in tqdm(range(1, self.epoch + 1),desc="epoch nums"):
            if epoch % self.config.replay_epoch == 0 and epoch != 0 and epoch < 50:
                self.dataset.replay(logger)
                self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True ,  pin_memory=True)
            for batch in self.dataloader:
                
                global_step += 1

                # batch_audio , batch_rgb ,batch_depth, batch_angle , batch_action = batch
                states , next_states , actions , rewards , dones = batch

                loss_dict = self.agent.update(states, actions , rewards, next_states ,  dones)

                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)
                tqdm.write(f"actor_loss: {loss_dict['actor_loss']} , critic1_loss:{loss_dict['critic1_loss']} , critic2_loss:{loss_dict['critic2_loss']}")
            if epoch % 10 == 0 :
                torch.save(self.agent.state_dict() , f'{self.save_dir}/sac_2level_model_{epoch}.pth')
            if epoch % self.online_test_epoch== 0:
                # train_acc = self.val(epoch)
                self.agent.eval()
                online_reward  , spl = self.online_test.rollout(epoch , self.agent ,logger )
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
                a_predicted_logits = self.agent.get_action(states)  

                pred_classes = torch.argmax(a_predicted_logits, dim=1)  
                correct = (pred_classes == actions).sum().item()         
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.agent.train()
        return acc