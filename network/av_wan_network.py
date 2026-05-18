import torch
import torch.nn as nn

from network.hybird.Encoder import Encoder
from network.hybird.ViT import PositionalEcoder, VisualEncoder
from network.hybird.audio import AudioCRNN


class Network(nn.Module):
    def __init__(
        self,
        action_dim=4,
        pose_dim=4,
        sound_num_embeddings=32,
        modality_dim=128,
        hidden_dim=256,
    ):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.add_position = PositionalEcoder()
        self.visual_transformer_encoder = Encoder(
            d_model=128,
            ffn_hidden=64,
            n_head=4,
            n_layers=3,
            drop_prob=0.2,
        )
        self.audio_encoder = AudioCRNN()
        self.vaencoder = Encoder(
            d_model=128,
            ffn_hidden=64,
            n_head=4,
            n_layers=3,
            drop_prob=0.2,
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, modality_dim),
            nn.ReLU(),
            nn.LayerNorm(modality_dim),
        )
        self.ego_map_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, modality_dim),
            nn.ReLU(),
            nn.LayerNorm(modality_dim),
        )
        self.collision_head = nn.Linear(modality_dim, 1)

        self.fusion = nn.Sequential(
            nn.Linear(modality_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dropout = nn.Dropout(0.2)
        self.polar_head = nn.Linear(hidden_dim, 2)
        self.angle_head = nn.Linear(hidden_dim, 8)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_dim),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_inputs(self, audio, rgb, depth):
        if len(audio.shape) == 3:
            audio = audio.unsqueeze(0)
        if len(rgb.shape) == 3:
            rgb = rgb.unsqueeze(0)
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(0)
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)
        rgbd = torch.cat([rgb, depth], dim=1)
        return audio, rgbd

    def _encode_audio_visual(self, audio, rgb, depth):
        audio, rgbd = self._prepare_inputs(audio, rgb, depth)
        audio_encoder = self.audio_encoder.encoder_forward(audio)
        visual_cnn = self.visual_encoder(rgbd)
        v_batch, v_dim, v_h, v_w = visual_cnn.shape
        visual_cnn = visual_cnn.reshape(v_batch, v_h * v_w, v_dim)
        v_position = self.add_position(visual_cnn)
        visual_encoder = self.visual_transformer_encoder(visual_cnn + v_position)
        concat_encoder = torch.cat((audio_encoder, visual_encoder), dim=1)
        p_share_encoder = self.add_position(concat_encoder)
        share_visual_audio_encoder = self.vaencoder(concat_encoder + p_share_encoder)
        return share_visual_audio_encoder.mean(dim=1)

    def _encode_pose(self, pose):
        return self.pose_encoder(pose.float())

    def _encode_ego_map(self, ego_map):
        if ego_map.dim() == 3:
            ego_map = ego_map.unsqueeze(1)
        ego_map = ego_map.permute(0, 3, 2, 1)
        return self.ego_map_encoder(ego_map.float())

    def embedding_forward(self, audio, rgb, depth , pose , ego_map):
        if len(pose.shape) == 1:
            pose = pose.unsqueeze(0)
        if len(ego_map.shape) == 3:
            ego_map = ego_map.unsqueeze(0)
        av_feature = self._encode_audio_visual(audio, rgb, depth)
        pose_feature = self._encode_pose(pose)
        ego_map_feature = self._encode_ego_map(ego_map)
        fused = torch.cat([av_feature, pose_feature, ego_map_feature], dim=1).squeeze(0)
        # import pdb;pdb.set_trace()
        embedding = self.fusion(fused)
        return embedding

    def encode_state(self, audio, rgb, depth, pose, ego_map):
        av_feature = self._encode_audio_visual(audio, rgb, depth)
        pose_feature = self._encode_pose(pose)
        ego_map_feature = self._encode_ego_map(ego_map)
        fused = torch.cat([av_feature, pose_feature, ego_map_feature], dim=1)
        hidden = self.dropout(self.fusion(fused))
        collision_logits = self.collision_head(av_feature).squeeze(-1)
        return hidden, collision_logits

    def forward(self, audio, rgb, depth, pose, ego_map):
        hidden, collision_logits = self.encode_state(audio, rgb, depth, pose, ego_map)
        polar_predict = self.polar_head(hidden)
        angle_logits = self.angle_head(hidden)
        action_logits = self.action_head(hidden)
        return polar_predict, angle_logits, action_logits, collision_logits
