from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from utils.log import logger
import numpy as np
NUM_SECTORS = 8
SECTOR_ANGLE = 2 * 180 / NUM_SECTORS  # 每个扇区角度

class DirectionLoss(nn.Module):
    def __init__(self, num_sectors=NUM_SECTORS):
        super().__init__()
        self.num_sectors = num_sectors
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self,pred, true_angle, N=8):
        true_sector = torch.remainder(true_angle + 22.5, 2*180) // SECTOR_ANGLE
        true_sector = true_sector.long()
        batch_size = pred.shape[0]
        device = pred.device
        
        # 先转成 softmax 概率
        prob = F.softmax(pred, dim=1)
        
        total_loss = 0.0
        for i in range(batch_size):
            t = true_sector[i]
            # 对每个可能的预测 sector 计算环形距离
            idx = torch.arange(N, device=device)
            dist = torch.abs(idx - t)
            dist = torch.minimum(dist, N - dist)  # 环形距离
            
            # 计算指数惩罚 α
            alpha = torch.where(dist == 0, torch.zeros_like(dist), 2 ** dist)
            
            loss_i = torch.sum(prob[i] * alpha)
            total_loss += loss_i
    
        return total_loss / batch_size , true_sector
    
class Train:
    def __init__(self, model, Adam, train_loader,val_loader, epoch, writer, save_dir ,train_size , val_size, device='cuda'):
        self.train_model = model.to(device)
        self.optimizer = Adam
        self.epoch = epoch
        self.writer = writer
        self.device = device
        self.save_dir = save_dir
        self.train_size = train_size
        self.val_size = val_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.losser = DirectionLoss()



    def train(self):
        global_step = 0

        for ep in range(self.epoch):
            local_step = 0
            epoch_angle_loss = 0.0
            
            for batch in self.train_loader:

                batch_audio , batch_visual , batch_angle , batch_action = batch
                batch_audio  = batch_audio.to(self.device)
                batch_angle = batch_angle.float().to(self.device)

                angle_predict = self.train_model(batch_audio)

                loss_angle , label  = self.losser(angle_predict , batch_angle)

                self.optimizer.zero_grad()
                loss_angle.backward()
                self.optimizer.step()
                preds = angle_predict.argmax(dim=1)
                
                diff = torch.abs(preds - label)

                dist = torch.minimum(diff, 8 - diff)

                correct = (dist <= 1).sum().item() / batch_angle.shape[0]
                
                print(f"Epoch {ep}, Step {global_step} , angle loss {loss_angle:.6f}")
                epoch_angle_loss += loss_angle.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step_angle", loss_angle.item(), global_step)
                    self.writer.add_scalar("Loss/step_acc" , correct , global_step)
                global_step += 1
                local_step += 1

                # if global_step % 500 == 0:
                #     self.validate(global_step)
                
            self.validate(ep)


            avg_angle_loss = epoch_angle_loss / local_step
            if self.writer:
                self.writer.add_scalar("Loss/epoch_angle", avg_angle_loss, ep)
            print(f"Epoch {ep+1} finished, average angle loss: {avg_angle_loss:.4f}")
            if (ep + 1) % 50 == 0:
                save_path = f"{self.save_dir}/model_epoch_{ep+1}.pth"
                torch.save(self.train_model.state_dict(), save_path)
                tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        val_action_loss = 0.0
        total_acc = 0.0
        correct = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch_audio , batch_visual , batch_angle , batch_action = batch
                batch_audio  = batch_audio.to(self.device)
                batch_angle = batch_angle.float().to(self.device)

                angle_predict = self.train_model(batch_audio)
                # angle_loss = F.mse_loss(angle_predict.squeeze(1) , batch_angle)
                # angle_loss = self.bounded_mse_loss(angle_predict , batch_angle)
                angle_loss ,label = self.losser(angle_predict , batch_angle)
                
                val_action_loss += angle_loss.item()
                preds = angle_predict.argmax(dim=1)  # [batch]
                diff = torch.abs(preds - label)

                dist = torch.minimum(diff, 8 - diff)

                correct += (dist <= 1).sum().item()

        avg_action_loss = val_action_loss / len(self.val_loader)
        total_acc = correct / self.val_size

        if self.writer:
            self.writer.add_scalar("Val/angle_loss", avg_action_loss, step)
            self.writer.add_scalar("Val/acc", total_acc, step)
        self.train_model.train()

