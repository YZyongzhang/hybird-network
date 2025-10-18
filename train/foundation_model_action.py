from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from utils.log import logger
import numpy as np
import math
import pickle
NUM_SECTORS = 8
SECTOR_ANGLE = 2 * 180 / NUM_SECTORS  # 每个扇区角度
class DirectionLoss(nn.Module):
    def __init__(self, num_sectors=NUM_SECTORS):
        super().__init__()
        self.num_sectors = num_sectors
        self.ce_loss = nn.CrossEntropyLoss()

    # def forward(self, sector_logits, true_angle):
    #     """
    #     sector_logits: [B, num_sectors] 分类 logits
    #     delta_pred: [B] 微调预测, 单位 rad
    #     true_angle: [B] 真值角度, 单位 rad
    #     """
    #     # -------------------
    #     # 1. 分类标签
    #     # -------------------
    #     # 将真实角度映射到扇区索引
        
    #     sector_label = torch.remainder(true_angle + 22.5, 2*180) // SECTOR_ANGLE
    #     sector_label = sector_label.long()

    #     # -------------------
    #     # 2. 分类 loss
    #     # -------------------
    #     loss_ce = self.ce_loss(sector_logits, sector_label)

    #     return loss_ce , sector_label
        
    def forward(self,pred, true_angle, N=8):
        """
        pred: [batch, N] softmax 概率或 logits
        true_sector: [batch] 真实扇区编号
        """
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

    def deal_batch_angle(self , batch_angle):
        # batch_angle: (batch , 1)
        return torch.stack([torch.tensor((i.item()-30 , i.item()+30) , dtype=float) for i in batch_angle])
    
    # def angle_error(self, angle_predict , batch_angle):
    #     return torch.nn.ReLU(angle_predict - batch_angle)
    # def compute_loss_by_guss(self , angle_predict , batch_angle):
    #     # angle_pre[0] for i in angle_predict
    #     # batch_angle[0]
    #     self.guss(angle , lable)
    
    # def guss(self , angle , lable):
    #     error = lable - angle
    #     loss = 
    # def bounded_mse_loss(self , pred, target, tol=30):  # 容忍 ±30°
    #     diff = torch.remainder(pred - target + 180.0, 360.0) - 180.0
    #     # 超出容忍区间的才计算惩罚
    #     penalty = torch.clamp(torch.abs(diff) - tol, min=0.0)
    #     return torch.mean(penalty ** 2)

    def train(self):
        global_step = 0

        for ep in range(self.epoch):
            local_step = 0
            epoch_angle_loss = 0.0
            
            for batch in self.train_loader:
                # unpack 数据
                    # batch_visual, batch_audio, batch_action , batch_angle = batc
                batch_audio , batch_visual , batch_angle , batch_action = batch
                batch_audio  , batch_visual ,batch_action = batch_audio.to(self.device) , batch_visual.to(self.device) , batch_action.to(self.device)
                batch_angle = batch_angle.float().to(self.device)
                batch_action = batch_action.long()
                # batch_angle = self.deal_batch_angle(batch_angle).float().to(self.device)


                # import pdb;pdb.set_trace()
                action_predict = self.train_model(batch_audio , batch_visual)
                # error = self.angle_error(angle_predict.sque , seze(1) , batch_angle)
                # loss_angle = F.mse_loss(angle_predict.squeeze(1) , batch_angle)
                # loss_angle = self.bounded_mse_loss(angle_predict , batch_angle)
                # loss_angle , label  = self.losser(angle_predict , batch_angle)
                # loss_angle = self.compute_loss_by_guss(angle_predict , batch_angle)
                loss_action = F.cross_entropy(action_predict , batch_action)
                # loss = loss_angle
                self.optimizer.zero_grad()
                loss_action.backward()
                self.optimizer.step()
                preds = action_predict.argmax(dim=1)
                
                # diff = torch.abs(preds - label)

                # 因为是环形的，所以 0 和 7 也是相邻的（取 min(dist, 8 - dist)）
                # dist = torch.minimum(diff, 8 - diff)

                # 判断是否在相邻范围内
                # correct = (dist <= 1).sum().item() / batch_angle.shape[0]
                
                correct = (preds == batch_action).sum() / batch_angle.shape[0]
                print(f"Epoch {ep}, Step {global_step} , angle loss {loss_action:.6f}")
                epoch_angle_loss += loss_action.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step_angle", loss_action.item(), global_step)
                    self.writer.add_scalar("Loss/step_acc" , correct , global_step)
                global_step += 1
                local_step += 1

                # if global_step % 500 == 0:
                #     self.validate(global_step)
            self.validate(ep)


            avg_angle_loss = epoch_angle_loss / local_step
            if self.writer:
                self.writer.add_scalar("Loss/epoch_action", avg_angle_loss, ep)
            print(f"Epoch {ep+1} finished, average action loss: {avg_angle_loss:.4f}")
            if (ep + 1) % 50 == 0:
                save_path = f"{self.save_dir}/model_epoch_{ep+1}.pth"
                torch.save(self.train_model.state_dict(), save_path)
                tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        total = 0
        val_action_loss = 0.0
        total_acc = 0.0
        correct = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch_audio , batch_visual , batch_angle , batch_action = batch
                batch_audio  , batch_visual , batch_action = batch_audio.to(self.device) , batch_visual.to(self.device) , batch_action.to(self.device)
                batch_angle = batch_angle.float().to(self.device)
                batch_action = batch_action.long()

                # batch_angle = self.deal_batch_angle(batch_angle).float().to(self.device)
                action_predict = self.train_model(batch_audio , batch_visual)
                # angle_loss = F.mse_loss(angle_predict.squeeze(1) , batch_angle)
                # angle_loss = self.bounded_mse_loss(angle_predict , batch_angle)
                # angle_loss ,label = self.losser(angle_predict , batch_angle)
                action_loss = F.cross_entropy(action_predict , batch_action)
                
                val_action_loss += action_loss.item()
                preds = action_predict.argmax(dim=1)  # [batch]
                # diff = torch.abs(preds - )

                # 因为是环形的，所以 0 和 7 也是相邻的（取 min(dist, 8 - dist)）
                # dist = torch.minimum(diff, 8 - diff)

                # 判断是否在相邻范围内
                # correct += (dist <= 1).sum().item()
                correct += (preds == batch_action).sum().item()


        avg_action_loss = val_action_loss / len(self.val_loader)
        total_acc = correct / self.val_size

        if self.writer:
            self.writer.add_scalar("Val/action_loss", avg_action_loss, step)
            self.writer.add_scalar("Val/acc", total_acc, step)
        self.train_model.train()

