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

onnx = pytest.importorskip("onnx")

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
        from training.export import InferencePolicy, export_onnx

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()

        onnx_path = tmp_path / "test.onnx"
        export_onnx(inf, onnx_path)

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_onnx_input_names(self, tmp_path):
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


# ---------------------------------------------------------------------------
# prepare_onnx_for_tflite
# ---------------------------------------------------------------------------


class TestPrepareOnnxForTflite:
    def _export_onnx(self, tmp_path):
        """Helper: export a test ONNX model and return its path."""
        from training.export import InferencePolicy, export_onnx

        fake_model = _make_fake_sb3_model()
        inf = InferencePolicy(fake_model)
        inf.eval()
        onnx_path = tmp_path / "test.onnx"
        export_onnx(inf, onnx_path)
        return onnx_path

    def test_creates_output_file(self, tmp_path):
        from training.export import prepare_onnx_for_tflite

        onnx_path = self._export_onnx(tmp_path)
        result = prepare_onnx_for_tflite(onnx_path)
        assert result.exists()
        assert result.name == "test_same_pad.onnx"

    def test_removes_explicit_pads(self, tmp_path):
        from training.export import prepare_onnx_for_tflite

        onnx_path = self._export_onnx(tmp_path)
        result = prepare_onnx_for_tflite(onnx_path)

        model = onnx.load(str(result))
        for node in model.graph.node:
            if node.op_type in ("Conv", "ConvTranspose"):
                pads = [a for a in node.attribute if a.name == "pads"]
                auto_pad = [a for a in node.attribute if a.name == "auto_pad"]
                # Either no pads attribute, or it was a pointwise conv (no padding).
                has_nonzero_pads = any(any(p > 0 for p in a.ints) for a in pads)
                if has_nonzero_pads:
                    pytest.fail(f"Conv node still has explicit nonzero pads: {pads}")
                # If auto_pad was set (not the default NOTSET), it
                # should be SAME_UPPER.
                for a in auto_pad:
                    if a.s not in (b"", b"NOTSET"):
                        assert a.s == b"SAME_UPPER"

    def test_output_is_valid_onnx(self, tmp_path):
        from training.export import prepare_onnx_for_tflite

        onnx_path = self._export_onnx(tmp_path)
        result = prepare_onnx_for_tflite(onnx_path)
        model = onnx.load(str(result))
        onnx.checker.check_model(model)

    def test_no_change_for_pointwise_convs(self, tmp_path):
        """Pointwise (1×1) convs have no padding — should be left alone."""
        from training.export import prepare_onnx_for_tflite

        onnx_path = self._export_onnx(tmp_path)

        result = prepare_onnx_for_tflite(onnx_path)
        model = onnx.load(str(result))

        # Pointwise convs (kernel_size=1) should not get auto_pad=SAME_UPPER.
        for node in model.graph.node:
            if node.op_type != "Conv":
                continue
            kernel = next((a for a in node.attribute if a.name == "kernel_shape"), None)
            if kernel and all(k == 1 for k in kernel.ints):
                auto_pad = next(
                    (a for a in node.attribute if a.name == "auto_pad"), None
                )
                assert auto_pad is None or auto_pad.s in (b"", b"NOTSET")


# ---------------------------------------------------------------------------
# collect_representative_dataset (mocked VizDoom)
# ---------------------------------------------------------------------------


class TestCollectRepresentativeDataset:
    def test_returns_correct_count(self):
        from training.export import collect_representative_dataset

        with patch("training.env.DoomHybridEnv") as mock_env_cls:
            env = MagicMock()
            mock_env_cls.return_value = env

            obs = {
                "visual": np.random.rand(4, 45, 60).astype(np.float32),
                "state": np.random.rand(20).astype(np.float32),
            }
            env.reset.return_value = (obs, {})
            env.step.return_value = (obs, 1.0, False, False, {})
            env.action_space.sample.return_value = np.zeros(6)

            samples = collect_representative_dataset(n_samples=10)
            assert len(samples) == 10
            env.close.assert_called_once()

    def test_sample_shapes(self):
        from training.export import collect_representative_dataset

        with patch("training.env.DoomHybridEnv") as mock_env_cls:
            env = MagicMock()
            mock_env_cls.return_value = env

            obs = {
                "visual": np.random.rand(4, 45, 60).astype(np.float32),
                "state": np.random.rand(20).astype(np.float32),
            }
            env.reset.return_value = (obs, {})
            env.step.return_value = (obs, 1.0, False, False, {})
            env.action_space.sample.return_value = np.zeros(6)

            samples = collect_representative_dataset(n_samples=5)
            # Visual should be NHWC with batch dim.
            assert samples[0]["visual"].shape == (1, 45, 60, 4)
            assert samples[0]["state"].shape == (1, 20)
            assert samples[0]["visual"].dtype == np.float32

    def test_handles_episode_reset(self):
        from training.export import collect_representative_dataset

        with patch("training.env.DoomHybridEnv") as mock_env_cls:
            env = MagicMock()
            mock_env_cls.return_value = env

            obs = {
                "visual": np.random.rand(4, 45, 60).astype(np.float32),
                "state": np.random.rand(20).astype(np.float32),
            }
            env.reset.return_value = (obs, {})
            # Episode ends on 3rd step.
            env.step.side_effect = [
                (obs, 1.0, False, False, {}),
                (obs, 1.0, False, False, {}),
                (obs, 0.0, True, False, {}),
                (obs, 1.0, False, False, {}),
                (obs, 1.0, False, False, {}),
            ]
            env.action_space.sample.return_value = np.zeros(6)

            samples = collect_representative_dataset(n_samples=5)
            assert len(samples) == 5
            # Should have reset twice: once at start, once after done.
            assert env.reset.call_count == 2
