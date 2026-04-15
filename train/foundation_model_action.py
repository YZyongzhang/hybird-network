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
                batch_audio , batch_rgb ,batch_depth, batch_angle , batch_action = batch

                batch_audio  , batch_rgb ,batch_depth , batch_action = batch_audio.to(self.device) , batch_rgb.to(self.device) ,batch_depth.to(self.device), batch_action.to(self.device)
                batch_action = batch_action.float()

                action_predict = self.train_model(batch_audio , batch_rgb , batch_depth)
                loss_action = F.mse_loss(action_predict, batch_action)
                radius_mae = torch.mean(torch.abs(action_predict[:, 0] - batch_action[:, 0]))
                theta_mae = torch.mean(torch.abs(action_predict[:, 1] - batch_action[:, 1]))
                self.optimizer.zero_grad()
                loss_action.backward()
                self.optimizer.step()

                print(f"Epoch {ep}, Step {global_step} , polar loss {loss_action:.6f}")
                epoch_angle_loss += loss_action.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step_polar", loss_action.item(), global_step)
                    self.writer.add_scalar("Loss/step_radius_mae" , radius_mae.item() , global_step)
                    self.writer.add_scalar("Loss/step_theta_mae" , theta_mae.item() , global_step)
                global_step += 1
                local_step += 1

                # if global_step % 500 == 0:
                #     self.validate(global_step)
            self.validate(ep)


            avg_angle_loss = epoch_angle_loss / local_step
            if self.writer:
                self.writer.add_scalar("Loss/epoch_polar", avg_angle_loss, ep)
            print(f"Epoch {ep+1} finished, average polar loss: {avg_angle_loss:.4f}")
            if (ep + 1) % 1 == 0:
                save_path = f"{self.save_dir}/model_epoch_{ep+1}.pth"
                torch.save(self.train_model.state_dict(), save_path)
                tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        val_action_loss = 0.0
        val_radius_mae = 0.0
        val_theta_mae = 0.0
        with torch.no_grad():

            for batch in self.val_loader:
                batch_audio , batch_rgb ,batch_depth , batch_angle , batch_action = batch
                batch_audio  , batch_rgb , batch_depth , batch_action = batch_audio.to(self.device) , batch_rgb.to(self.device) ,batch_depth.to(self.device), batch_action.to(self.device)
                batch_action = batch_action.float()

                action_predict = self.train_model(batch_audio , batch_rgb , batch_depth)
                action_loss = F.mse_loss(action_predict, batch_action)
                radius_mae = torch.mean(torch.abs(action_predict[:, 0] - batch_action[:, 0]))
                theta_mae = torch.mean(torch.abs(action_predict[:, 1] - batch_action[:, 1]))
                
                val_action_loss += action_loss.item()
                val_radius_mae += radius_mae.item()
                val_theta_mae += theta_mae.item()

        avg_action_loss = val_action_loss / len(self.val_loader)
        avg_radius_mae = val_radius_mae / len(self.val_loader)
        avg_theta_mae = val_theta_mae / len(self.val_loader)
        if self.writer:
            self.writer.add_scalar("Val/polar_loss", avg_action_loss, step)
            self.writer.add_scalar("Val/radius_mae", avg_radius_mae, step)
            self.writer.add_scalar("Val/theta_mae", avg_theta_mae, step)
        self.train_model.train()

