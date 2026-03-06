from typing import Optional

import torch

from network.onlinerl.base import BaseOnlineRL


class OnlineRLV2(BaseOnlineRL):
    """Online RL agent with randomly initialised foundation backbone."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        foundation_ckpt: Optional[str] = None,
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
            load_foundation_weights=False,
            freeze_backbone=freeze_backbone,
            device=device,
        )

    def load_pending_foundation(self) -> None:
        """Optionally load a pending foundation checkpoint after initialisation."""
        if not self.pending_foundation_ckpt:
            raise ValueError("No pending foundation checkpoint recorded.")
        self._load_foundation_weights(self.pending_foundation_ckpt)
        self.pending_foundation_ckpt = None

    @classmethod
    def from_config(cls, config, *, device: Optional[torch.device] = None):
        return cls(
            action_dim=int(getattr(config, "action_dim")),
            hidden_dim=int(getattr(config, "hidden_dim", 256)),
            foundation_ckpt=getattr(config, "FOUNDATION_CKPT", None),
            use_pose_encoder=bool(getattr(config, "use_pose_encoder", False)),
            pose_dim=int(getattr(config, "pose_dim", 7)),
            pose_hidden_dim=int(getattr(config, "pose_hidden_dim", 64)),
            freeze_backbone=bool(getattr(config, "freeze_backbone", False)),
            device=device,
        )
