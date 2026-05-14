import os

import torch
import torch.nn.functional as F
from tqdm import tqdm


class Train:
    def __init__(
        self,
        model,
        Adam,
        train_loader,
        val_loader,
        epoch,
        writer,
        save_dir,
        train_size,
        val_size,
        device="cuda",
    ):
        self.train_model = model.to(device)
        self.optimizer = Adam
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = int(epoch)
        self.writer = writer
        self.save_dir = save_dir
        self.train_size = int(train_size)
        self.val_size = int(val_size)
        self.device = device

    def _move_to_device(self, batch):
        batch_audio, batch_rgb, batch_depth, batch_action = batch[:4]
        batch_audio = batch_audio.to(self.device)
        batch_rgb = batch_rgb.to(self.device)
        batch_depth = batch_depth.to(self.device)
        batch_action = batch_action.long().to(self.device)
        return batch_audio, batch_rgb, batch_depth, batch_action

    def _forward(self, batch_audio, batch_rgb, batch_depth):
        # batch_size, seq_len = batch_audio.shape[:2]
        # audio = batch_audio.reshape(batch_size * seq_len, *batch_audio.shape[2:])
        # rgb = batch_rgb.reshape(batch_size * seq_len, *batch_rgb.shape[2:])
        # depth = batch_depth.reshape(batch_size * seq_len, *batch_depth.shape[2:])
        action_logits = self.train_model(batch_audio, batch_rgb, batch_depth)
        # if action_logits.dim() != 3:
        #     raise ValueError(f"Expected action logits to have 3 dims, got {tuple(action_logits.shape)}")
        # if action_logits.shape[0] == batch_size * seq_len:
        #     action_logits = action_logits.view(batch_size, seq_len, action_logits.shape[1], action_logits.shape[2])
        return action_logits

    def _compute_loss_and_acc(self, action_logits, batch_action):
        if action_logits.dim() == 4:
            if action_logits.shape[2] != batch_action.shape[1]:
                raise ValueError(
                    f"Model sequence length {action_logits.shape[2]} does not match target length {batch_action.shape[1]}"
                )
            expanded_target = batch_action.unsqueeze(1).expand(-1, action_logits.shape[1], -1)
            loss = F.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]),
                expanded_target.reshape(-1),
            )
            pred = action_logits.argmax(dim=-1)
            acc = (pred == expanded_target).float().mean()
            sequence_acc = (pred == expanded_target).all(dim=-1).float().mean()
            return loss, acc, sequence_acc

        if action_logits.shape[1] != batch_action.shape[1]:
            raise ValueError(
                f"Model sequence length {action_logits.shape[1]} does not match target length {batch_action.shape[1]}"
            )
        loss = F.cross_entropy(
            action_logits.reshape(-1, action_logits.shape[-1]),
            batch_action.reshape(-1),
        )
        pred = action_logits.argmax(dim=-1)
        acc = (pred == batch_action).float().mean()
        sequence_acc = (pred == batch_action).all(dim=-1).float().mean()
        return loss, acc, sequence_acc

    def _run_epoch(self, loader, step_prefix, global_step=None):
        total_loss = 0.0
        total_acc = 0.0
        total_sequence_acc = 0.0
        steps = 0

        for batch in loader:
            batch_audio, batch_rgb, batch_depth, batch_action = self._move_to_device(batch)
            action_logits = self._forward(batch_audio, batch_rgb, batch_depth)
            loss, acc, sequence_acc = self._compute_loss_and_acc(action_logits, batch_action)

            if self.train_model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            total_sequence_acc += sequence_acc.item()

            if self.writer is not None and global_step is not None:
                self.writer.add_scalar(f"Loss/{step_prefix}_action", loss.item(), global_step)
                self.writer.add_scalar(f"Acc/{step_prefix}_token", acc.item(), global_step)
                self.writer.add_scalar(f"Acc/{step_prefix}_sequence", sequence_acc.item(), global_step)
                global_step += 1

            steps += 1

        avg_loss = total_loss / max(1, steps)
        avg_acc = total_acc / max(1, steps)
        avg_sequence_acc = total_sequence_acc / max(1, steps)
        return avg_loss, avg_acc, avg_sequence_acc, global_step

    def train(self):
        os.makedirs(self.save_dir, exist_ok=True)
        global_step = 0

        for ep in range(self.epoch):
            self.train_model.train()
            train_loss, train_acc, train_sequence_acc, global_step = self._run_epoch(
                self.train_loader,
                step_prefix="step",
                global_step=global_step,
            )

            val_loss, val_acc, val_sequence_acc = self.validate(ep)

            if self.writer is not None:
                self.writer.add_scalar("Loss/epoch_action", train_loss, ep)
                self.writer.add_scalar("Acc/epoch_token", train_acc, ep)
                self.writer.add_scalar("Acc/epoch_sequence", train_sequence_acc, ep)
                self.writer.add_scalar("Val/action_loss", val_loss, ep)
                self.writer.add_scalar("Val/action_token_acc", val_acc, ep)
                self.writer.add_scalar("Val/action_sequence_acc", val_sequence_acc, ep)

            print(
                f"Epoch {ep + 1} finished, "
                f"train_loss={train_loss:.4f}, train_token_acc={train_acc:.4f}, train_sequence_acc={train_sequence_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_token_acc={val_acc:.4f}, val_sequence_acc={val_sequence_acc:.4f}"
            )

            save_path = f"{self.save_dir}/model_epoch_{ep + 1}.pth"
            torch.save(self.train_model.state_dict(), save_path)
            tqdm.write(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_sequence_acc, _ = self._run_epoch(
                self.val_loader,
                step_prefix="val_step",
                global_step=None,
            )
        self.train_model.train()
        return val_loss, val_acc, val_sequence_acc
