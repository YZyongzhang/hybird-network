import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.hybird.foundation_model import Network as FoundationModel


def _ensure_tensor(data, device: torch.device) -> torch.Tensor:
    """Best-effort conversion of numpy arrays or tensors to float tensors on device."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=torch.float32)
    return torch.as_tensor(data, device=device, dtype=torch.float32)


class BaseOnlineRL(nn.Module):
    """Common utilities for Online RL policies backed by the foundation model encoder."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        foundation_ckpt: Optional[str] = None,
        use_pose_encoder: bool = False,
        pose_dim: int = 7,
        pose_hidden_dim: int = 64,
        *,
        load_foundation_weights: bool,
        freeze_backbone: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = FoundationModel().to(self.device)
        self.pending_foundation_ckpt: Optional[str] = None
        if load_foundation_weights:
            if foundation_ckpt is None:
                raise ValueError("foundation_ckpt must be provided when load_foundation_weights=True.")
            self._load_foundation_weights(foundation_ckpt)
        elif foundation_ckpt:
            # Allows manually loading later while keeping clear intent.
            self.pending_foundation_ckpt = foundation_ckpt

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.use_pose_encoder = bool(use_pose_encoder)
        self.pose_dim = int(pose_dim)
        self.pose_hidden_dim = int(pose_hidden_dim)
        self.embedding_dim = self._infer_embedding_dim()
        if self.use_pose_encoder:
            self.pose_encoder = nn.Sequential(
                nn.LayerNorm(self.pose_dim),
                nn.Linear(self.pose_dim, self.pose_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.pose_hidden_dim, self.pose_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.pose_fusion = nn.Sequential(
                nn.Linear(self.embedding_dim + self.pose_hidden_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.pose_encoder = None
            self.pose_fusion = None
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _infer_embedding_dim(self) -> int:
        for module in self.backbone.final.fc1:
            if isinstance(module, nn.Linear):
                return module.out_features
        raise RuntimeError("Unable to infer embedding dimension from foundation model.")

    def _load_foundation_weights(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"foundation checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=self.device)
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Loading foundation weights incomplete. missing_keys={missing}, unexpected_keys={unexpected}"
            )

    def encode(self, audio, rgb, depth, pose=None) -> torch.Tensor:
        audio_t = _ensure_tensor(audio, self.device)
        rgb_t = _ensure_tensor(rgb, self.device)
        depth_t = _ensure_tensor(depth, self.device)
        embedding = self.backbone.embedding_forward(audio_t, rgb_t, depth_t).float()
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        if not self.use_pose_encoder:
            return embedding

        if pose is None:
            raise ValueError("pose input is required when use_pose_encoder=True.")
        pose_t = _ensure_tensor(pose, self.device)
        if pose_t.ndim == 1:
            pose_t = pose_t.unsqueeze(0)
        if pose_t.size(-1) != self.pose_dim:
            raise ValueError(
                f"pose dim mismatch: expected last dim={self.pose_dim}, got {pose_t.size(-1)}"
            )
        pose_feat = self.pose_encoder(pose_t)
        return self.pose_fusion(torch.cat([embedding, pose_feat], dim=-1))

    def forward(
        self, audio, rgb, depth, pose=None, *, return_embedding: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        embedding = self.encode(audio, rgb, depth, pose=pose)
        logits = self.policy_head(embedding)
        state_value = self.value_head(embedding).squeeze(-1)
        if return_embedding:
            return logits, state_value, embedding
        return logits, state_value

    def act(self, audio, rgb, depth, pose=None, *, greedy: bool = False) -> torch.Tensor:
        logits, _ = self.forward(audio, rgb, depth, pose=pose, return_embedding=False)
        probs = F.softmax(logits, dim=-1)
        if greedy:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action

    def value(self, audio, rgb, depth, pose=None) -> torch.Tensor:
        _, state_value = self.forward(audio, rgb, depth, pose=pose, return_embedding=False)
        return state_value

    def backbone_parameters(self):
        return self.backbone.parameters()

    def policy_parameters(self):
        return self.policy_head.parameters()

    def value_parameters(self):
        return self.value_head.parameters()

    def pose_parameters(self):
        if not self.use_pose_encoder:
            return []
        return list(self.pose_encoder.parameters()) + list(self.pose_fusion.parameters())
