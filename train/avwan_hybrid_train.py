import torch
import torch.nn.functional as F
from tqdm import tqdm


class Train:
    def __init__(self, model, Adam, train_loader, val_loader, epoch, writer, save_dir, train_size, val_size, device='cuda'):
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

    def _move_batch(self, batch):
        (
            batch_audio,
            batch_rgb,
            batch_depth,
            batch_angle,
            batch_action,
            batch_action_id,
            batch_pose,
            batch_ego_map,
            batch_collision,
            batch_consistency,
        ) = batch
        return (
            batch_audio.to(self.device),
            batch_rgb.to(self.device),
            batch_depth.to(self.device),
            batch_angle.float().to(self.device),
            batch_action.float().to(self.device),
            batch_action_id.long().to(self.device),
            batch_pose.float().to(self.device),
            batch_ego_map.float().to(self.device),
            batch_collision.float().to(self.device),
            batch_consistency.float().to(self.device),
        )

    def _step(self, batch):
        (
            batch_audio,
            batch_rgb,
            batch_depth,
            batch_angle,
            batch_action,
            batch_action_id,
            batch_pose,
            batch_ego_map,
            batch_collision,
            batch_consistency,
        ) = self._move_batch(batch)

        polar_predict, angle_logits, action_logits, collision_logits = self.train_model(
            batch_audio,
            batch_rgb,
            batch_depth,
            batch_pose,
            batch_ego_map,
        )
        loss_polar = F.mse_loss(polar_predict, batch_action)
        loss_action = F.cross_entropy(action_logits, batch_action_id)
        angle_labels = torch.clamp(
            torch.floor(torch.remainder(batch_angle + 180.0, 360.0) / 45.0).long(),
            min=0,
            max=7,
        )
        loss_angle = F.cross_entropy(angle_logits, angle_labels)
        loss_collision = F.binary_cross_entropy_with_logits(collision_logits, batch_collision)
        loss_total = loss_action + loss_angle + loss_polar + loss_collision
        radius_mae = torch.mean(torch.abs(polar_predict[:, 0] - batch_action[:, 0]))
        theta_mae = torch.mean(torch.abs(polar_predict[:, 1] - batch_action[:, 1]))
        action_acc = (action_logits.argmax(dim=1) == batch_action_id).float().mean()
        angle_acc = (angle_logits.argmax(dim=1) == angle_labels).float().mean()
        collision_acc = ((torch.sigmoid(collision_logits) >= 0.5).float() == batch_collision).float().mean()
        return loss_total, loss_action, loss_polar, loss_angle, loss_collision, radius_mae, theta_mae, action_acc, angle_acc, collision_acc

    def train(self):
        global_step = 0
        for ep in range(self.epoch):
            local_step = 0
            epoch_loss = 0.0
            for batch in self.train_loader:
                loss_total, loss_action, loss_polar, loss_angle, loss_collision, radius_mae, theta_mae, action_acc, angle_acc, collision_acc = self._step(batch)
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                print(
                    f"Epoch {ep}, Step {global_step} , total loss {loss_total:.6f}, "
                    f"action loss {loss_action:.6f}, polar loss {loss_polar:.6f}, angle loss {loss_angle:.6f}, collision loss {loss_collision:.6f}"
                )
                epoch_loss += loss_total.item()
                if self.writer:
                    self.writer.add_scalar("Loss/step_total", loss_total.item(), global_step)
                    self.writer.add_scalar("Loss/step_action", loss_action.item(), global_step)
                    self.writer.add_scalar("Loss/step_polar", loss_polar.item(), global_step)
                    self.writer.add_scalar("Loss/step_angle", loss_angle.item(), global_step)
                    self.writer.add_scalar("Loss/step_collision", loss_collision.item(), global_step)
                    self.writer.add_scalar("Acc/step_action", action_acc.item(), global_step)
                    self.writer.add_scalar("Acc/step_angle", angle_acc.item(), global_step)
                    self.writer.add_scalar("Acc/step_collision", collision_acc.item(), global_step)
                    self.writer.add_scalar("Loss/step_radius_mae", radius_mae.item(), global_step)
                    self.writer.add_scalar("Loss/step_theta_mae", theta_mae.item(), global_step)
                global_step += 1
                local_step += 1

            self.validate(ep)
            avg_loss = epoch_loss / max(1, local_step)
            if self.writer:
                self.writer.add_scalar("Loss/epoch_total", avg_loss, ep)
            print(f"Epoch {ep + 1} finished, average total loss: {avg_loss:.4f}")
            save_path = f"{self.save_dir}/model_epoch_{ep + 1}.pth"
            torch.save(self.train_model.state_dict(), save_path)
            tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        val_total_loss = 0.0
        val_action_loss = 0.0
        val_polar_loss = 0.0
        val_angle_loss = 0.0
        val_collision_loss = 0.0
        val_radius_mae = 0.0
        val_theta_mae = 0.0
        val_action_acc = 0.0
        val_angle_acc = 0.0
        val_collision_acc = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                total_loss, action_loss, polar_loss, angle_loss, collision_loss, radius_mae, theta_mae, action_acc, angle_acc, collision_acc = self._step(batch)
                val_total_loss += total_loss.item()
                val_action_loss += action_loss.item()
                val_polar_loss += polar_loss.item()
                val_angle_loss += angle_loss.item()
                val_collision_loss += collision_loss.item()
                val_radius_mae += radius_mae.item()
                val_theta_mae += theta_mae.item()
                val_action_acc += action_acc.item()
                val_angle_acc += angle_acc.item()
                val_collision_acc += collision_acc.item()

        avg_total_loss = val_total_loss / len(self.val_loader)
        avg_action_loss = val_action_loss / len(self.val_loader)
        avg_polar_loss = val_polar_loss / len(self.val_loader)
        avg_angle_loss = val_angle_loss / len(self.val_loader)
        avg_collision_loss = val_collision_loss / len(self.val_loader)
        avg_radius_mae = val_radius_mae / len(self.val_loader)
        avg_theta_mae = val_theta_mae / len(self.val_loader)
        avg_action_acc = val_action_acc / len(self.val_loader)
        avg_angle_acc = val_angle_acc / len(self.val_loader)
        avg_collision_acc = val_collision_acc / len(self.val_loader)
        if self.writer:
            self.writer.add_scalar("Val/total_loss", avg_total_loss, step)
            self.writer.add_scalar("Val/action_loss", avg_action_loss, step)
            self.writer.add_scalar("Val/polar_loss", avg_polar_loss, step)
            self.writer.add_scalar("Val/angle_loss", avg_angle_loss, step)
            self.writer.add_scalar("Val/collision_loss", avg_collision_loss, step)
            self.writer.add_scalar("Val/action_acc", avg_action_acc, step)
            self.writer.add_scalar("Val/angle_acc", avg_angle_acc, step)
            self.writer.add_scalar("Val/collision_acc", avg_collision_acc, step)
            self.writer.add_scalar("Val/radius_mae", avg_radius_mae, step)
            self.writer.add_scalar("Val/theta_mae", avg_theta_mae, step)
        self.train_model.train()
