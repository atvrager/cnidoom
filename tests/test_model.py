"""Tests for the DoomFeatureExtractor — pure PyTorch, no VizDoom needed."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from training.model import DoomFeatureExtractor, DepthwiseSepConv2d


# ---------------------------------------------------------------------------
# DepthwiseSepConv2d
# ---------------------------------------------------------------------------


class TestDepthwiseSepConv2d:
    def test_output_shape(self):
        layer = DepthwiseSepConv2d(4, 16, stride=2)
        x = torch.randn(2, 4, 45, 60)
        y = layer(x)
        assert y.shape == (2, 16, 23, 30)

    def test_stride_1(self):
        layer = DepthwiseSepConv2d(8, 16, stride=1)
        x = torch.randn(1, 8, 10, 10)
        y = layer(x)
        assert y.shape == (1, 16, 10, 10)

    def test_output_not_all_zeros(self):
        """ReLU may zero some, but a random input shouldn't be all zeros."""
        layer = DepthwiseSepConv2d(4, 16, stride=2)
        x = torch.randn(1, 4, 45, 60)
        y = layer(x)
        assert y.abs().sum() > 0


# ---------------------------------------------------------------------------
# DoomFeatureExtractor
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_space():
    return gym.spaces.Dict(
        {
            "visual": gym.spaces.Box(0, 1, (4, 45, 60), dtype=np.float32),
            "state": gym.spaces.Box(-1, 1, (20,), dtype=np.float32),
        }
    )


class TestDoomFeatureExtractor:
    def test_output_shape(self, obs_space):
        extractor = DoomFeatureExtractor(obs_space, features_dim=256)
        obs = {
            "visual": torch.randn(4, 4, 45, 60),
            "state": torch.randn(4, 20),
        }
        features = extractor(obs)
        assert features.shape == (4, 256)

    def test_single_sample(self, obs_space):
        extractor = DoomFeatureExtractor(obs_space, features_dim=256)
        obs = {
            "visual": torch.randn(1, 4, 45, 60),
            "state": torch.randn(1, 20),
        }
        features = extractor(obs)
        assert features.shape == (1, 256)

    def test_custom_features_dim(self, obs_space):
        extractor = DoomFeatureExtractor(obs_space, features_dim=128)
        obs = {
            "visual": torch.randn(1, 4, 45, 60),
            "state": torch.randn(1, 20),
        }
        assert extractor(obs).shape == (1, 128)

    def test_features_dim_attribute(self, obs_space):
        extractor = DoomFeatureExtractor(obs_space, features_dim=256)
        assert extractor.features_dim == 256

    def test_param_count_reasonable(self, obs_space):
        """Total params should be in the ~400K range per the spec."""
        extractor = DoomFeatureExtractor(obs_space, features_dim=256)
        total = sum(p.numel() for p in extractor.parameters())
        assert 300_000 < total < 600_000

    def test_gradient_flows(self, obs_space):
        """Verify gradients propagate through the full network."""
        extractor = DoomFeatureExtractor(obs_space, features_dim=256)
        obs = {
            "visual": torch.randn(2, 4, 45, 60),
            "state": torch.randn(2, 20),
        }
        out = extractor(obs)
        loss = out.sum()
        loss.backward()
        # Check that conv layer weights got gradients.
        first_conv = extractor.visual_net[0].depthwise
        assert first_conv.weight.grad is not None
        assert first_conv.weight.grad.abs().sum() > 0

    def test_deterministic_eval_mode(self, obs_space):
        """In eval mode, BatchNorm uses running stats → deterministic output."""
        extractor = DoomFeatureExtractor(obs_space, features_dim=256)
        extractor.eval()
        obs = {
            "visual": torch.randn(1, 4, 45, 60),
            "state": torch.randn(1, 20),
        }
        with torch.no_grad():
            out1 = extractor(obs)
            out2 = extractor(obs)
        torch.testing.assert_close(out1, out2)
