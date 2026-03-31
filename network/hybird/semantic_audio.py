import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional

from network.hybird.audio import AudioCRNN


NUM_SECTORS = 8


def angle_to_sector_label(angle_deg: torch.Tensor, num_sectors: int = NUM_SECTORS) -> torch.Tensor:
    """Map angle in degrees to circular sector labels [0, num_sectors-1]."""
    angle_deg = angle_deg.float()
    sector_size = 360.0 / float(num_sectors)
    normalized = torch.remainder(angle_deg + 180.0, 360.0)
    labels = torch.floor(normalized / sector_size).long()
    return torch.clamp(labels, min=0, max=num_sectors - 1)


def angle_to_sin_cos(angle_deg: torch.Tensor) -> torch.Tensor:
    """Convert angle in degrees to [cos(theta), sin(theta)] targets."""
    theta = torch.deg2rad(angle_deg.float())
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


class SemanticAudioLoss:
    """
    AudioVisual-master style semantic-audio objective:
    - classifier_loss = CE(logits, labels) - lambda_entropy * entropy
    - optional regressor_loss = MSE(pred, target_sin_cos)
    - total = lambda_cls * classifier_loss + lambda_reg * regressor_loss
    """

    def __init__(
        self,
        num_sectors: int = NUM_SECTORS,
        lambda_classifier: float = 1.0,
        lambda_classifier_entropy: float = 0.0,
        lambda_regressor: float = 0.0,
    ) -> None:
        self.num_sectors = int(num_sectors)
        self.lambda_classifier = float(lambda_classifier)
        self.lambda_classifier_entropy = float(lambda_classifier_entropy)
        self.lambda_regressor = float(lambda_regressor)

    def __call__(
        self,
        angle_logits: torch.Tensor,
        angle_deg: torch.Tensor,
        predicted_dir: Optional[torch.Tensor] = None,
    ) -> dict:
        labels = angle_to_sector_label(angle_deg, self.num_sectors).to(angle_logits.device)

        classifier_loss = F.cross_entropy(angle_logits, labels)
        classifier_entropy = Categorical(logits=angle_logits).entropy().mean()
        classifier_loss = classifier_loss - self.lambda_classifier_entropy * classifier_entropy

        total_loss = self.lambda_classifier * classifier_loss
        regressor_loss = torch.tensor(0.0, device=angle_logits.device)

        if predicted_dir is not None and self.lambda_regressor > 0.0:
            rot_label = angle_to_sin_cos(angle_deg).to(predicted_dir.device)
            regressor_loss = F.mse_loss(predicted_dir, rot_label)
            total_loss = total_loss + self.lambda_regressor * regressor_loss

        preds = angle_logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        return {
            "total_loss": total_loss,
            "classifier_loss": classifier_loss,
            "classifier_entropy": classifier_entropy,
            "regressor_loss": regressor_loss,
            "labels": labels,
            "preds": preds,
            "acc": acc,
        }


class SemanticAudioNet(nn.Module):
    """
    Standalone semantic-audio model for audio-only training.
    Outputs:
    - label_logits: semantic label branch
    - direction_logits: direction branch from AudioCNN head
    """

    def __init__(self, num_classes: int = NUM_SECTORS) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.audio_cnn = AudioCRNN()
        self.label_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes),
        )

    def _extract_audio_feature(self, audio: torch.Tensor) -> torch.Tensor:
        feat = self.audio_cnn.encoder_forward(audio)
        if feat.dim() == 3:
            feat = feat.squeeze(1)
        elif feat.dim() > 2:
            feat = feat.flatten(1)
        return feat

    def forward(self, audio: torch.Tensor):
        feat = self._extract_audio_feature(audio)
        label_logits = self.label_head(feat)
        direction_logits = self.audio_cnn(audio)
        return label_logits, direction_logits
