"""Tests for the export pipeline — focuses on InferencePolicy and ONNX export.

Heavy deps (tensorflow, onnx2tf) are mocked or skipped. The InferencePolicy
and ONNX export are tested with real PyTorch.
"""

from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from training.model import DoomFeatureExtractor

# ---------------------------------------------------------------------------
# Helpers: build a fake SB3-like model structure
# ---------------------------------------------------------------------------


def _make_fake_sb3_model():
    """Create a mock that mimics SB3 PPO's .policy structure."""
    obs_space = gym.spaces.Dict(
        {
            "visual": gym.spaces.Box(0, 1, (4, 45, 60), dtype=np.float32),
            "state": gym.spaces.Box(-1, 1, (20,), dtype=np.float32),
        }
    )

    features_extractor = DoomFeatureExtractor(obs_space, features_dim=256)
    policy_net = nn.Sequential(nn.Linear(256, 64), nn.ReLU())
    action_net = nn.Linear(64, 6)

    policy = MagicMock()
    policy.features_extractor = features_extractor
    policy.mlp_extractor.policy_net = policy_net
    policy.action_net = action_net

    model = MagicMock()
    model.policy = policy
    return model


# ---------------------------------------------------------------------------
# InferencePolicy
# ---------------------------------------------------------------------------


class TestInferencePolicy:
    def test_output_shape(self):
        from training.export import InferencePolicy

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        vis = torch.randn(2, 4, 45, 60)
        state = torch.randn(2, 20)
        with torch.no_grad():
            out = inf(vis, state)
        assert out.shape == (2, 6)

    def test_output_is_probability(self):
        """Sigmoid output should be in [0, 1]."""
        from training.export import InferencePolicy

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        vis = torch.randn(10, 4, 45, 60)
        state = torch.randn(10, 20)
        with torch.no_grad():
            out = inf(vis, state)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_deterministic_in_eval(self):
        from training.export import InferencePolicy

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        vis = torch.randn(1, 4, 45, 60)
        state = torch.randn(1, 20)
        with torch.no_grad():
            out1 = inf(vis, state)
            out2 = inf(vis, state)
        torch.testing.assert_close(out1, out2)

    def test_no_value_head(self):
        """InferencePolicy should not have a value head attribute."""
        from training.export import InferencePolicy

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        assert not hasattr(inf, "value_net")
        assert not hasattr(inf, "vf_net")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


onnxscript = pytest.importorskip("onnxscript")


class TestOnnxExport:
    def test_export_creates_file(self, tmp_path):
        from training.export import InferencePolicy, export_onnx

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        onnx_path = tmp_path / "test.onnx"
        export_onnx(inf, onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_onnx_loadable(self, tmp_path):
        """Exported ONNX should be parseable."""
        onnx = pytest.importorskip("onnx")
        from training.export import InferencePolicy, export_onnx

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        onnx_path = tmp_path / "test.onnx"
        export_onnx(inf, onnx_path)

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_onnx_input_names(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        from training.export import InferencePolicy, export_onnx

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        onnx_path = tmp_path / "test.onnx"
        export_onnx(inf, onnx_path)

        model = onnx.load(str(onnx_path))
        input_names = [inp.name for inp in model.graph.input]
        assert "visual" in input_names
        assert "state" in input_names

    def test_onnx_output_name(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        from training.export import InferencePolicy, export_onnx

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        onnx_path = tmp_path / "test.onnx"
        export_onnx(inf, onnx_path)

        model = onnx.load(str(onnx_path))
        output_names = [out.name for out in model.graph.output]
        assert "action_probs" in output_names


# ---------------------------------------------------------------------------
# extract_inference_model (mocked SB3 load)
# ---------------------------------------------------------------------------


class TestExtractInferenceModel:
    def test_returns_inference_policy(self):
        from training.export import InferencePolicy, extract_inference_model

        fake_model = _make_fake_sb3_model()
        with patch("stable_baselines3.PPO") as mock_ppo:
            mock_ppo.load.return_value = fake_model
            inf = extract_inference_model("fake_checkpoint.zip")
            assert isinstance(inf, InferencePolicy)
            mock_ppo.load.assert_called_once_with("fake_checkpoint.zip", device="cpu")
