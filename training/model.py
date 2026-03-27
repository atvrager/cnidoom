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


class DoomFeatureExtractorV2(BaseFeaturesExtractor):
    """V2 feature extractor: deeper CNN + Global Average Pooling.

    Visual path: 6 depthwise-separable conv blocks
        (4, 60, 80) → s2 → (32, 30, 40) → s2 → (64, 15, 20)
        → s1 → (64, 15, 20) → s2 → (128, 8, 10) → s1 → (128, 8, 10)
        → s2 → (192, 4, 5)
        GAP → 192

    State path: 20-float vector passed through directly.

    Combined: concat(192 + 20) = 212 → Dense(256, ReLU) → Dense(128, ReLU)
              → 128-dim features.
    """

    # Channel config per block: (out_channels, stride).
    V2_BLOCKS = [(32, 2), (64, 2), (64, 1), (128, 2), (128, 1), (192, 2)]

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        in_ch = observation_space["visual"].shape[0]  # frame stack depth (4)
        layers: list[nn.Module] = []
        for out_ch, stride in self.V2_BLOCKS:
            layers.append(DepthwiseSepConv2d(in_ch, out_ch, stride=stride))
            in_ch = out_ch

        self.visual_net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        state_dim = observation_space["state"].shape[0]
        gap_dim = in_ch  # last block's output channels

        self.fc = nn.Sequential(
            nn.Linear(gap_dim + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        vis = self.visual_net(observations["visual"])
        vis = self.gap(vis).flatten(1)  # [B, C, 1, 1] → [B, C]
        state = observations["state"]
        return self.fc(torch.cat([vis, state], dim=1))
