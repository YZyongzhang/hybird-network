import torch
import torch.nn.functional as F

from network.hybird.semantic_audio import NUM_SECTORS, SemanticAudioLoss, angle_to_sector_label


class Train:
    """
    Standalone semantic-audio trainer.
    Following HybirdNetworkAudio data flow, but optimize two audio losses:
    1) semantic label loss
    2) direction loss (AudioCNN direction head)
    """

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
        self.epoch = int(epoch)
        self.writer = writer
        self.device = device
        self.save_dir = save_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_size = int(train_size)
        self.val_size = int(val_size)

        self.lambda_label = 1.0
        self.lambda_direction = 1.0
        self.semantic_loss = SemanticAudioLoss(
            num_sectors=NUM_SECTORS,
            lambda_classifier=1.0,
            lambda_classifier_entropy=0.0,
            lambda_regressor=0.0,
        )

    def train(self):
        global_step = 0
        for ep in range(self.epoch):
            epoch_total = 0.0
            epoch_label = 0.0
            epoch_dir = 0.0
            seen = 0
            correct_label = 0
            correct_dir = 0
            local_step = 0

            for batch in self.train_loader:
                batch_audio, batch_rgb, batch_depth, batch_angle, batch_action = batch
                del batch_rgb, batch_depth, batch_action

                batch_audio = batch_audio.to(self.device)
                batch_angle = batch_angle.float().to(self.device)

                label_logits, direction_logits = self.train_model(batch_audio)

                sem_out = self.semantic_loss(label_logits, batch_angle)
                label_loss = sem_out["classifier_loss"]
                labels = sem_out["labels"]
                label_preds = sem_out["preds"]

                direction_loss = F.cross_entropy(direction_logits, labels)
                direction_preds = direction_logits.argmax(dim=1)

                total_loss = self.lambda_label * label_loss + self.lambda_direction * direction_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                bs = int(labels.shape[0])
                seen += bs
                correct_label += (label_preds == labels).sum().item()
                correct_dir += (direction_preds == labels).sum().item()

                epoch_total += float(total_loss.item())
                epoch_label += float(label_loss.item())
                epoch_dir += float(direction_loss.item())
                local_step += 1

                if self.writer:
                    self.writer.add_scalar("Loss/step_total", float(total_loss.item()), global_step)
                    self.writer.add_scalar("Loss/step_label", float(label_loss.item()), global_step)
                    self.writer.add_scalar("Loss/step_direction", float(direction_loss.item()), global_step)
                global_step += 1

            self.validate(ep)

            avg_total = epoch_total / max(1, local_step)
            avg_label = epoch_label / max(1, local_step)
            avg_dir = epoch_dir / max(1, local_step)
            label_acc = correct_label / max(1, seen)
            dir_acc = correct_dir / max(1, seen)
            print(
                f"Epoch {ep+1} finished, total={avg_total:.4f}, "
                f"label={avg_label:.4f}, direction={avg_dir:.4f}, "
                f"label_acc={label_acc:.4f}, direction_acc={dir_acc:.4f}"
            )
            if self.writer:
                self.writer.add_scalar("Loss/epoch_total", avg_total, ep)
                self.writer.add_scalar("Loss/epoch_label", avg_label, ep)
                self.writer.add_scalar("Loss/epoch_direction", avg_dir, ep)
                self.writer.add_scalar("Acc/epoch_label", label_acc, ep)
                self.writer.add_scalar("Acc/epoch_direction", dir_acc, ep)

            save_path = f"{self.save_dir}/model_epoch_{ep + 1}.pth"
            torch.save(self.train_model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    def validate(self, step):
        self.train_model.eval()
        val_total = 0.0
        val_label = 0.0
        val_dir = 0.0
        seen = 0
        correct_label = 0
        correct_dir = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch_audio, batch_rgb, batch_depth, batch_angle, batch_action = batch
                del batch_rgb, batch_depth, batch_action

                batch_audio = batch_audio.to(self.device)
                batch_angle = batch_angle.float().to(self.device)

                label_logits, direction_logits = self.train_model(batch_audio)

                sem_out = self.semantic_loss(label_logits, batch_angle)
                label_loss = sem_out["classifier_loss"]
                labels = sem_out["labels"]
                label_preds = sem_out["preds"]

                direction_loss = F.cross_entropy(direction_logits, labels)
                direction_preds = direction_logits.argmax(dim=1)

                total_loss = self.lambda_label * label_loss + self.lambda_direction * direction_loss

                bs = int(labels.shape[0])
                seen += bs
                correct_label += (label_preds == labels).sum().item()
                correct_dir += (direction_preds == labels).sum().item()

                val_total += float(total_loss.item())
                val_label += float(label_loss.item())
                val_dir += float(direction_loss.item())

        avg_total = val_total / max(1, len(self.val_loader))
        avg_label = val_label / max(1, len(self.val_loader))
        avg_dir = val_dir / max(1, len(self.val_loader))
        label_acc = correct_label / max(1, seen)
        dir_acc = correct_dir / max(1, seen)

        if self.writer:
            self.writer.add_scalar("Val/total_loss", avg_total, step)
            self.writer.add_scalar("Val/label_loss", avg_label, step)
            self.writer.add_scalar("Val/direction_loss", avg_dir, step)
            self.writer.add_scalar("Val/label_acc", label_acc, step)
            self.writer.add_scalar("Val/direction_acc", dir_acc, step)

        self.train_model.train()

