import sys
 
from module import *
from network.foundation_model import Network
from train.VADE import VADE
class Train:
    def __init__(self, model, Adam, train_loader ,val_loader, epoch, writer, device , save_dir):
        self.train_model = model.to(device)
        self.optimizer = Adam
        self.epoch = epoch
        self.writer = writer
        self.device = device
        self.save_dir = save_dir

        self.train_loader = train_loader
        self.val_loader = val_loader
    def train(self):
        global_step = 0

        for ep in range(self.epoch):
            local_step = 0
            epoch_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {ep+1}/{self.epoch}")

            for batch in pbar:
                # unpack 数据
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    batch_visual, batch_audio, batch_action = batch
                    batch_visual, batch_audio = batch_visual.to(self.device), batch_audio.to(self.device)
                else:
                    raise ValueError("训练数据格式不正确，应为 (visual, audio, label) 三元组")
                
                batch_action = torch.tensor([int(a) for a in batch_action], dtype=torch.long).to(self.device)

                


                outputs = self.train_model(batch_audio, batch_visual)
                loss = F.cross_entropy(outputs, batch_action)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                tqdm.write(f"Epoch {ep}, Step {global_step}, Loss: {loss:.6f}")
                epoch_loss += loss.item()

                if self.writer:
                    self.writer.add_scalar("Loss/step", loss.item(), global_step)
                    
                if global_step % 100 == 0:
                    self.validate(global_step)
                if global_step % 1000 == 0:
                    save_path = f"{self.save_dir}/model_epoch_{global_step}.pth"
                    torch.save(self.train_model.state_dict(), save_path)
                    tqdm.write(f"Saved model checkpoint to {save_path}")
                global_step += 1
                local_step += 1
                pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / local_step
            if self.writer:
                self.writer.add_scalar("Loss/epoch", avg_loss, ep)

            tqdm.write(f"Epoch {ep+1} finished, average loss: {avg_loss:.4f}")


    def validate(self, epoch):
        self.train_model.eval()
        correct = 0
        total = 0
        val_action_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    batch_visual, batch_audio, batch_action = batch
                    batch_visual, batch_audio = batch_visual.to(self.device).squeeze(0), batch_audio.to(self.device).squeeze(0)
                    # batch_angle = batch_angle.float().to(self.device)
                else:
                    raise ValueError("验证数据格式不正确，应为 (visual, audio, label) 三元组")

                batch_action = torch.tensor([int(a) for a in batch_action], dtype=torch.long).to(self.device)

                action_predict  = self.train_model(batch_audio, batch_visual)
                action_loss = F.cross_entropy(action_predict, batch_action)
                # angle_loss = F.mse_loss(angle_predict.squeeze(1) , batch_angle)

                val_action_loss += action_loss.item()
                preds = action_predict.argmax(dim=1)  # [batch]
                correct += (preds == batch_action).sum().item()
                total += batch_action.size(0)

        acc = correct / total if total > 0 else 0
        avg_action_loss = val_action_loss / len(self.val_loader)
        print(f"[Val] Epoch {epoch+1}: Loss={avg_action_loss:.4f}, Acc={acc:.4f}")

        if self.writer:
            self.writer.add_scalar("Val/action_loss", avg_action_loss, epoch)
            self.writer.add_scalar("Val/Acc", acc, epoch)

        self.train_model.train()






