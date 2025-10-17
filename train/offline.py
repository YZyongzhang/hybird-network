"""
利用offline RL进行训练。
"""
import sys
 
from network.OfflineNet import SAC_model
from train.VADE import VADE_Offline
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from env.env import MultiAudioEnv as Env
class SAC_train:
    def __init__(self, dataset, sac_model, writer , batch_size=64, device='cuda'):
        self.device = device
        self.agent = sac_model
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True ,drop_last=True , num_workers = 16,pin_memory=True)
        self.writer = writer
        self.env = Env()

    def train(self, H = "offline" ,num_epochs=1000, log_interval=10):
        
        global_step = 0
        for epoch in tqdm(range(1, num_epochs + 1),desc="epoch nums"):

            

            for batch in self.dataloader:
                
                global_step += 1
                states , next_states , actions , rewards , dones = batch
                loss_dict = self.agent.update(states, actions , rewards, next_states ,  dones)


                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)

                tqdm.write(f"step:{global_step} ,critic_1_loss:{loss_dict['critic_1_loss']} , critic_2_loss : {loss_dict['critic_2_loss']} , actor_loss : {loss_dict['actor_loss']} , alpha_loss:{loss_dict['alpha_loss']}")

            if epoch % 100 == 0 :
                path = f'./experiment/train/offline/ckpt/{H}/'
                times = time.localtime()
                time_str = f"{times.tm_year}-{times.tm_mon}-{times.tm_mday}-{times.tm_hour}_{times.tm_min}_{times.tm_sec}"
                os.makedirs(path,exist_ok=True)
                torch.save(self.agent.state_dict() , f'{path}/Foundation_model_{time_str}_{epoch}.pth')
            if epoch % 10 == 0:
                self.val(epoch)

    def val(self, epoch):
        self.agent.eval() 

        total_a_loss = 0
        total_correct = 0
        total_steps = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:  
                states, next_states, actions, rewards, dones = batch
                states = states.to(self.device)
                actions = actions.to(self.device)

                actions = actions.long()  # 分类任务用 long
                a_predicted_logits = self.agent.actor(states)  # 输出还没 softmax 的 logits

                # === 计算准确率 ===
                pred_classes = torch.argmax(a_predicted_logits, dim=1)   # 预测类别
                correct = (pred_classes == actions).sum().item()         # 预测正确的数量
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0

        self.writer.add_scalar("Val/a_Accuracy", acc, global_step=epoch)

        self.agent.train()


def train():


    D = input("输入要训练的数据库名称，路径为默认配置：")
    dataset_path = f"./experiment/dataset/{D}"
    dataset = VADE_Offline(path=dataset_path)


    state_dim = 128
    action_dim = 4
    hidden_dim = 64
    lr = 1e-4
    target_entropy = -action_dim  
    tau = 0.005
    gamma = 0.99
    batch_size = 64
    num_epochs = 4000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sac_model = SAC_model(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        actor_lr=lr,
        critic_lr=lr,
        alpha_lr=lr,
        target_entropy=target_entropy,
        tau=tau,
        gamma=gamma,
        beta=1.0,
        device=device
    )

    H = input("输入该次训练的名称：")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"./experiment/train/offline/loss/{H}")

    E = input("输入该次训练的备注：")
    with open(f"./experiment/train/offline/explain.txt", 'a' ,  encoding='utf-8') as f:
        f.write(f"{H}:{E}\n")

    trainer = SAC_train(
        dataset=dataset,
        sac_model=sac_model,
        batch_size=batch_size,
        device=device,
        writer=writer
    )


    trainer.train(num_epochs=num_epochs , H=H)


    writer.close()

if __name__ == "__main__":
    train()