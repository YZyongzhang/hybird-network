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
            consistency_weight = 0.2
            polar_weight = 0.5
            for batch in self.train_loader:
                batch_audio , batch_rgb ,batch_depth, batch_angle , batch_action, batch_action_id, batch_consistency = batch

                batch_audio  , batch_rgb ,batch_depth = batch_audio.to(self.device) , batch_rgb.to(self.device) ,batch_depth.to(self.device)
                batch_action = batch_action.to(self.device).float()
                batch_action_id = batch_action_id.to(self.device).long()
                batch_angle = batch_angle.float().to(self.device)
                batch_consistency = batch_consistency.to(self.device).float()

                polar_predict, angle_logits, action_logits = self.train_model(batch_audio , batch_rgb , batch_depth)
                loss_polar = F.mse_loss(polar_predict, batch_action)
                loss_action = F.cross_entropy(action_logits, batch_action_id)
                angle_labels = torch.clamp(torch.floor(torch.remainder(batch_angle + 180.0, 360.0) / 45.0).long(), min=0, max=7)
                loss_angle = F.cross_entropy(angle_logits , angle_labels)
                loss_total = loss_action + loss_angle + loss_polar
                radius_mae = torch.mean(torch.abs(polar_predict[:, 0] - batch_action[:, 0]))
                theta_mae = torch.mean(torch.abs(polar_predict[:, 1] - batch_action[:, 1]))
                action_acc = (action_logits.argmax(dim=1) == batch_action_id).float().mean()
                angle_acc = (angle_logits.argmax(dim=1) == angle_labels).float().mean()
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                print(f"Epoch {ep}, Step {global_step} , total loss {loss_total:.6f}, action loss {loss_action:.6f}, polar loss {loss_polar:.6f}, angle loss {loss_angle:.6f}")
                epoch_angle_loss += loss_total.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step_total", loss_total.item(), global_step)
                    self.writer.add_scalar("Loss/step_action", loss_action.item(), global_step)
                    self.writer.add_scalar("Loss/step_polar", loss_polar.item(), global_step)
                    self.writer.add_scalar("Loss/step_angle", loss_angle.item(), global_step)
                    self.writer.add_scalar("Acc/step_action", action_acc.item(), global_step)
                    self.writer.add_scalar("Acc/step_angle", angle_acc.item(), global_step)
                    self.writer.add_scalar("Loss/step_radius_mae" , radius_mae.item() , global_step)
                    self.writer.add_scalar("Loss/step_theta_mae" , theta_mae.item() , global_step)
                global_step += 1
                local_step += 1

                # if global_step % 500 == 0:
                #     self.validate(global_step)
            self.validate(ep)


            avg_angle_loss = epoch_angle_loss / local_step
            if self.writer:
                self.writer.add_scalar("Loss/epoch_total", avg_angle_loss, ep)
            print(f"Epoch {ep+1} finished, average total loss: {avg_angle_loss:.4f}")
            if (ep + 1) % 1 == 0:
                save_path = f"{self.save_dir}/model_epoch_{ep+1}.pth"
                torch.save(self.train_model.state_dict(), save_path)
                tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        val_total_loss = 0.0
        val_action_loss = 0.0
        val_polar_loss = 0.0
        val_angle_loss = 0.0
        val_radius_mae = 0.0
        val_theta_mae = 0.0
        val_action_acc = 0.0
        val_angle_acc = 0.0
        with torch.no_grad():

            for batch in self.val_loader:
                batch_audio , batch_rgb ,batch_depth , batch_angle , batch_action, batch_action_id, batch_consistency = batch
                batch_audio  , batch_rgb , batch_depth = batch_audio.to(self.device) , batch_rgb.to(self.device) ,batch_depth.to(self.device)
                batch_action = batch_action.to(self.device).float()
                batch_action_id = batch_action_id.to(self.device).long()
                batch_angle = batch_angle.float().to(self.device)
                batch_consistency = batch_consistency.to(self.device).float()

                polar_predict, angle_logits, action_logits = self.train_model(batch_audio , batch_rgb , batch_depth)
                polar_loss = F.mse_loss(polar_predict, batch_action)
                action_loss = F.cross_entropy(action_logits, batch_action_id)
                angle_labels = torch.clamp(torch.floor(torch.remainder(batch_angle + 180.0, 360.0) / 45.0).long(), min=0, max=7)
                angle_loss = F.cross_entropy(angle_logits , angle_labels)
                total_loss = action_loss + angle_loss + polar_loss
                radius_mae = torch.mean(torch.abs(polar_predict[:, 0] - batch_action[:, 0]))
                theta_mae = torch.mean(torch.abs(polar_predict[:, 1] - batch_action[:, 1]))
                action_acc = (action_logits.argmax(dim=1) == batch_action_id).float().mean()
                angle_acc = (angle_logits.argmax(dim=1) == angle_labels).float().mean()
                
                val_total_loss += total_loss.item()
                val_action_loss += action_loss.item()
                val_polar_loss += polar_loss.item()
                val_angle_loss += angle_loss.item()
                val_radius_mae += radius_mae.item()
                val_theta_mae += theta_mae.item()
                val_action_acc += action_acc.item()
                val_angle_acc += angle_acc.item()

        avg_total_loss = val_total_loss / len(self.val_loader)
        avg_action_loss = val_action_loss / len(self.val_loader)
        avg_polar_loss = val_polar_loss / len(self.val_loader)
        avg_angle_loss = val_angle_loss / len(self.val_loader)
        avg_radius_mae = val_radius_mae / len(self.val_loader)
        avg_theta_mae = val_theta_mae / len(self.val_loader)
        avg_action_acc = val_action_acc / len(self.val_loader)
        avg_angle_acc = val_angle_acc / len(self.val_loader)
        if self.writer:
            self.writer.add_scalar("Val/total_loss", avg_total_loss, step)
            self.writer.add_scalar("Val/action_loss", avg_action_loss, step)
            self.writer.add_scalar("Val/polar_loss", avg_polar_loss, step)
            self.writer.add_scalar("Val/angle_loss", avg_angle_loss, step)
            self.writer.add_scalar("Val/action_acc", avg_action_acc, step)
            self.writer.add_scalar("Val/angle_acc", avg_angle_acc, step)
            self.writer.add_scalar("Val/radius_mae", avg_radius_mae, step)
            self.writer.add_scalar("Val/theta_mae", avg_theta_mae, step)
        self.train_model.train()

