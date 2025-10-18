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


    def train(self):
        global_step = 0

        for ep in range(self.epoch):
            local_step = 0
            epoch_angle_loss = 0.0
            
            for batch in self.train_loader:

                batch_audio , batch_visual , batch_angle , batch_action = batch
                batch_audio  , batch_visual ,batch_action = batch_audio.to(self.device) , batch_visual.to(self.device) , batch_action.to(self.device)
                batch_action = batch_action.long()


                action_predict = self.train_model(batch_audio , batch_visual)

                loss_action = F.cross_entropy(action_predict , batch_action)
                self.optimizer.zero_grad()
                loss_action.backward()
                self.optimizer.step()
                preds = action_predict.argmax(dim=1)

                
                correct = (preds == batch_action).sum() / batch_angle.shape[0]
                print(f"Epoch {ep}, Step {global_step} , action loss {loss_action:.6f}")
                epoch_angle_loss += loss_action.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step_action", loss_action.item(), global_step)
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
                batch_action = batch_action.long()

                action_predict = self.train_model(batch_audio , batch_visual)

                action_loss = F.cross_entropy(action_predict , batch_action)
                
                val_action_loss += action_loss.item()
                preds = action_predict.argmax(dim=1)  # [batch]

                correct += (preds == batch_action).sum().item()


        avg_action_loss = val_action_loss / len(self.val_loader)
        total_acc = correct / self.val_size

        if self.writer:
            self.writer.add_scalar("Val/action_loss", avg_action_loss, step)
            self.writer.add_scalar("Val/acc", total_acc, step)
        self.train_model.train()

