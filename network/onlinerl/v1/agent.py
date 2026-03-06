from typing import Optional

import torch

from network.onlinerl.base import BaseOnlineRL


class OnlineRLV1(BaseOnlineRL):
    """Online RL agent bootstrapped from foundation model weights."""

    def __init__(
        self,
        action_dim: int,
        foundation_ckpt: str,
        hidden_dim: int = 256,
        use_pose_encoder: bool = False,
        pose_dim: int = 7,
        pose_hidden_dim: int = 64,
        *,
        freeze_backbone: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            foundation_ckpt=foundation_ckpt,
            use_pose_encoder=use_pose_encoder,
            pose_dim=pose_dim,
            pose_hidden_dim=pose_hidden_dim,
            load_foundation_weights=True,
            freeze_backbone=freeze_backbone,
            device=device,
        )

    @classmethod
    def from_config(cls, config, *, device: Optional[torch.device] = None):
        return cls(
            action_dim=int(getattr(config, "action_dim")),
            foundation_ckpt=str(getattr(config, "FOUNDATION_CKPT")),
            hidden_dim=int(getattr(config, "hidden_dim", 256)),
            use_pose_encoder=bool(getattr(config, "use_pose_encoder", False)),
            pose_dim=int(getattr(config, "pose_dim", 7)),
            pose_hidden_dim=int(getattr(config, "pose_hidden_dim", 64)),
            freeze_backbone=bool(getattr(config, "freeze_backbone", False)),
            device=device,
        )
