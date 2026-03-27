"""Depthwise-separable CNN feature extractor for the Doom hybrid observation."""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DepthwiseSepConv2d(nn.Module):
    """Depthwise-separable convolution: depthwise 3x3 + pointwise 1x1."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class DoomFeatureExtractor(BaseFeaturesExtractor):
    """Hybrid CNN + state vector feature extractor for SB3.

    Visual path: 3 depthwise-separable conv blocks (stride 2 each)
        (4, 45, 60) → (16, 23, 30) → (32, 12, 15) → (32, 6, 8)
        Flatten → 1536

    State path: 20-float vector passed through directly.

    Combined: concat(1536 + 20) → Dense(256, ReLU) → 256-dim features.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # Must call super with the final features_dim.
        super().__init__(observation_space, features_dim)

        self.visual_net = nn.Sequential(
            DepthwiseSepConv2d(4, 16, stride=2),
            DepthwiseSepConv2d(16, 32, stride=2),
            DepthwiseSepConv2d(32, 32, stride=2),
            nn.Flatten(),
        )

        # Compute CNN output dim dynamically.
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 45, 60)
            cnn_out_dim = self.visual_net(dummy).shape[1]

        state_dim = observation_space["state"].shape[0]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + state_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        vis = self.visual_net(observations["visual"])
        state = observations["state"]
        return self.fc(torch.cat([vis, state], dim=1))
