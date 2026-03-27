"""Export trained SB3 PPO model to INT8 TFLite via ONNX → TF → TFLite.

Pipeline:
    1. Extract inference-only policy (no value head) from SB3 checkpoint
    2. Export to ONNX
    3. Preprocess ONNX graph (pads → auto_pad to avoid explicit Pad ops)
    4. Convert ONNX → TF SavedModel (handles NCHW → NHWC)
    5. Full-integer INT8 quantization with representative dataset
    6. Verify BatchNorm folding and INT8 vs FP32 agreement

Usage:
    uv run python -m training.export --checkpoint doom_agent_ppo.zip
    uv run python -m training.export --checkpoint doom_agent_ppo.zip \\
        --cfg scenarios/my.cfg
"""

import argparse
import os
from pathlib import Path

# Prevent protobuf MessageFactory errors from tensorflow/tensorboard.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# Suppress TF CPU instruction set warning (pip TF binary doesn't use AVX/FMA).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
import torch.nn as nn


class InferencePolicy(nn.Module):
    """Inference-only policy: features → policy_net → action_net → sigmoid.

    Strips the value head and normalizes outputs to [0, 1] probabilities.
    """

    def __init__(self, trained_model):
        super().__init__()
        policy = trained_model.policy
        self.features_extractor = policy.features_extractor
        self.pi_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, visual: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor({"visual": visual, "state": state})
        latent = self.pi_net(features)
        return torch.sigmoid(self.action_net(latent))


def extract_inference_model(checkpoint_path: str) -> InferencePolicy:
    """Load SB3 checkpoint and extract the inference-only policy."""
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint_path, device="cpu")
    inference_model = InferencePolicy(model)
    inference_model.eval()
    return inference_model


def _infer_visual_shape(model: InferencePolicy) -> tuple[int, int, int, int]:
    """Infer the expected visual input shape from the model's observation space.

    Returns (batch, channels, height, width) in NCHW format.
    """
    extractor = model.features_extractor
    if hasattr(extractor, "observation_space"):
        vis_shape = extractor.observation_space["visual"].shape  # (C, H, W)
        return (1, vis_shape[0], vis_shape[1], vis_shape[2])
    # Fallback for mock/test models: use baseline default.
    return (1, 4, 45, 60)


def export_onnx(model: InferencePolicy, onnx_path: Path) -> None:
    """Export the inference policy to ONNX format."""
    vis_shape = _infer_visual_shape(model)
    dummy_vis = torch.randn(*vis_shape)
    dummy_state = torch.randn(1, 20)

    torch.onnx.export(
        model,
        (dummy_vis, dummy_state),
        str(onnx_path),
        input_names=["visual", "state"],
        output_names=["action_probs"],
        opset_version=18,
    )
    print(f"ONNX model saved to {onnx_path} (visual shape: {list(vis_shape)})")


def prepare_onnx_for_tflite(onnx_path: Path) -> Path:
    """Rewrite Conv pads to auto_pad='SAME_UPPER' to avoid explicit Pad ops.

    During ONNX→TF conversion, onnx2tf conservatively creates explicit
    tf.pad() ops for Conv nodes with explicit pads attributes. TFLite's
    full-integer INT8 calibrator then fails on PAD dimension metadata.

    This rewrite is safe when padding is symmetric and nonzero (our case:
    padding=1 with 3×3 kernels), which is equivalent to TF's SAME padding.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    rewritten = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        pads_attr = next((a for a in node.attribute if a.name == "pads"), None)
        if pads_attr is None or len(pads_attr.ints) == 0:
            continue
        # Only rewrite symmetric nonzero padding.
        if not (
            all(p == pads_attr.ints[0] for p in pads_attr.ints)
            and pads_attr.ints[0] > 0
        ):
            continue
        node.attribute.remove(pads_attr)
        auto_pad_attr = next((a for a in node.attribute if a.name == "auto_pad"), None)
        if auto_pad_attr:
            auto_pad_attr.s = b"SAME_UPPER"
        else:
            node.attribute.append(onnx.helper.make_attribute("auto_pad", "SAME_UPPER"))
        rewritten += 1

    out_path = onnx_path.with_stem(onnx_path.stem + "_same_pad")
    onnx.save(model, str(out_path))
    print(f"Rewrote {rewritten} Conv node(s) to auto_pad=SAME_UPPER → {out_path}")
    return out_path


def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    """Convert ONNX model to TF SavedModel (handles NCHW → NHWC).

    Also generates FP32 and FP16 .tflite files in the output directory.
    """
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
        batch_size=1,
    )
    print(f"TF SavedModel saved to {saved_model_dir}")


def collect_representative_dataset(
    cfg_path: str | None = None,
    n_samples: int = 200,
    obs_h: int = 45,
    obs_w: int = 60,
) -> list[dict[str, np.ndarray]]:
    """Collect calibration frames from VizDoom for full-integer quantization.

    Returns a list of dicts with 'visual' (NHWC float32) and 'state' (float32)
    arrays, each with batch dimension 1.

    Requires vizdoom — must run in the training venv, not the export venv.
    Use save/load_calibration_data() to bridge the two venvs.
    """
    from training.env import DoomHybridEnv

    env = DoomHybridEnv(cfg_path=cfg_path, obs_h=obs_h, obs_w=obs_w)
    samples = []

    obs, _ = env.reset()
    for _ in range(n_samples):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        # Training uses NCHW; TFLite expects NHWC.
        vis_nchw = obs["visual"]  # (C, H, W)
        vis_nhwc = np.transpose(vis_nchw, (1, 2, 0))[np.newaxis]  # (1, H, W, C)
        state = obs["state"][np.newaxis]  # (1, 20)
        samples.append(
            {"visual": vis_nhwc.astype(np.float32), "state": state.astype(np.float32)}
        )
        if done:
            obs, _ = env.reset()

    env.close()
    print(f"Collected {len(samples)} calibration samples from VizDoom")
    return samples


def save_calibration_data(
    samples: list[dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Save calibration samples to a .npz file for cross-venv use."""
    visuals = np.concatenate([s["visual"] for s in samples], axis=0)
    states = np.concatenate([s["state"] for s in samples], axis=0)
    np.savez(str(path), visual=visuals, state=states)
    print(f"Saved {len(samples)} calibration samples to {path}")


def load_calibration_data(path: Path) -> list[dict[str, np.ndarray]]:
    """Load calibration samples from a .npz file."""
    data = np.load(str(path))
    n = data["visual"].shape[0]
    samples = []
    for i in range(n):
        samples.append(
            {
                "visual": data["visual"][i : i + 1],
                "state": data["state"][i : i + 1],
            }
        )
    print(f"Loaded {n} calibration samples from {path}")
    return samples


def quantize_tflite(
    saved_model_dir: Path,
    tflite_path: Path,
    representative_data: list[dict[str, np.ndarray]] | None = None,
) -> bytes:
    """Quantize TF SavedModel to full-integer INT8 TFLite.

    When representative_data is provided, uses full-integer quantization:
    INT8 weights, activations, inputs, and outputs. This is required for
    the RISC-V embedded target (no float compute).

    Without representative_data, falls back to dynamic-range quantization
    (INT8 weights, float activations/I/O).
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_data is not None:
        # Full-integer quantization with calibration data.
        # The SavedModel signature orders inputs alphabetically:
        # state (1, 20) then visual (1, H, W, C) in NHWC.
        def representative_dataset():
            for sample in representative_data:
                yield [sample["state"], sample["visual"]]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print(
            "Using full-integer INT8 quantization with "
            f"{len(representative_data)} calibration samples"
        )
    else:
        print("WARNING: No calibration data — using dynamic-range quantization")

    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"INT8 TFLite model saved to {tflite_path} ({size_kb:.1f} KB)")
    return tflite_model


def quantize_tflite_fp16(
    saved_model_dir: Path,
    tflite_path: Path,
) -> bytes:
    """Convert TF SavedModel to FP16 TFLite (float16 weights, float32 compute)."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"FP16 TFLite model saved to {tflite_path} ({size_kb:.1f} KB)")
    return tflite_model


def verify_no_batchnorm(tflite_path: Path) -> bool:
    """Verify that BatchNorm ops were folded into convolutions."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    ops = {detail["op_name"] for detail in interpreter._get_ops_details()}
    print(f"TFLite ops: {sorted(ops)}")

    # PAD is expected (from Conv2d padding decomposition) — not a BN artifact.
    bad_ops = ops & {"BATCH_NORM"}
    if bad_ops:
        print(f"WARNING: unfused BatchNorm ops found: {bad_ops}")
        return False
    print("BatchNorm folding verified — no unfused ops found.")
    return True


def verify_int8_vs_fp32(
    checkpoint_path: str,
    tflite_path: Path,
    n_samples: int = 50,
    tolerance: float = 0.15,
) -> bool:
    """Compare INT8 TFLite output against FP32 PyTorch output.

    Returns True if mean absolute difference is within tolerance.
    The TFLite model uses NHWC layout (onnx2tf transposes automatically),
    while the PyTorch model uses NCHW.
    """
    import tensorflow as tf

    # FP32 reference.
    model = extract_inference_model(checkpoint_path)
    model.eval()

    # INT8 TFLite.
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Infer visual shape from the model.
    vis_shape = _infer_visual_shape(model)

    diffs = []
    for _ in range(n_samples):
        vis_nchw = np.random.rand(*vis_shape).astype(np.float32)
        state = np.random.rand(1, 20).astype(np.float32)

        # FP32 PyTorch inference (NCHW).
        with torch.no_grad():
            fp32_out = model(torch.from_numpy(vis_nchw), torch.from_numpy(state))
            fp32_probs = fp32_out.numpy()

        # TFLite inference — set inputs based on shape/name.
        for detail in input_details:
            name = detail["name"]
            dtype = detail["dtype"]
            if "visual" in name or (
                len(detail["shape"]) == 4 and detail["shape"][-1] == 4
            ):
                # NCHW → NHWC for visual input.
                data = np.transpose(vis_nchw, (0, 2, 3, 1))
            else:
                data = state

            if dtype == np.int8:
                scale, zp = detail["quantization"]
                data = np.clip(np.round(data / scale) + zp, -128, 127).astype(np.int8)
            else:
                data = data.astype(dtype)
            interpreter.set_tensor(detail["index"], data)

        interpreter.invoke()

        out_detail = output_details[0]
        raw_out = interpreter.get_tensor(out_detail["index"])
        if out_detail["dtype"] == np.int8:
            out_scale, out_zp = out_detail["quantization"]
            tflite_probs = (raw_out.astype(np.float32) - out_zp) * out_scale
        else:
            tflite_probs = raw_out.astype(np.float32)

        diffs.append(np.abs(fp32_probs - tflite_probs).mean())

    mean_diff = np.mean(diffs)
    print(f"INT8 vs FP32 mean abs diff: {mean_diff:.4f} (tolerance: {tolerance})")
    ok = mean_diff < tolerance
    if ok:
        print("Quantization accuracy check PASSED.")
    else:
        print("WARNING: Quantization accuracy check FAILED.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Export Doom agent to INT8 TFLite")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SB3 PPO checkpoint (.zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration .npz file for full-integer INT8 quantization "
        "(generate with: uv run python -m training.calibrate)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip INT8 vs FP32 verification",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    onnx_path = out / "doom_agent.onnx"
    saved_model_dir = out / "doom_agent_tf"
    tflite_path = out / "doom_agent_int8.tflite"

    # 1. Extract and export to ONNX.
    print("=" * 60)
    print("Step 1: Extracting inference policy and exporting to ONNX")
    print("=" * 60)
    model = extract_inference_model(args.checkpoint)
    export_onnx(model, onnx_path)

    # 2. Preprocess ONNX graph (pads → auto_pad).
    print("\n" + "=" * 60)
    print("Step 2: Preprocessing ONNX graph for TFLite compatibility")
    print("=" * 60)
    prepared_onnx = prepare_onnx_for_tflite(onnx_path)

    # 3. ONNX → TF SavedModel + FP32/FP16 TFLite.
    print("\n" + "=" * 60)
    print("Step 3: Converting ONNX → TF SavedModel")
    print("=" * 60)
    onnx_to_saved_model(prepared_onnx, saved_model_dir)

    # 4. Load calibration data for full-integer quantization.
    representative_data = None
    if args.calibration_data:
        print("\n" + "=" * 60)
        print("Step 4: Loading calibration data")
        print("=" * 60)
        representative_data = load_calibration_data(
            Path(args.calibration_data),
        )

    # 5. Export clean FP32 and FP16 TFLite from SavedModel.
    fp32_tflite = out / "doom_agent_fp32.tflite"
    fp16_tflite = out / "doom_agent_fp16.tflite"
    print("\n" + "=" * 60)
    print("Step 5: Exporting FP32 and FP16 TFLite")
    print("=" * 60)
    quantize_tflite(saved_model_dir, fp32_tflite, None)
    quantize_tflite_fp16(saved_model_dir, fp16_tflite)

    # 6. Quantize to INT8 TFLite.
    print("\n" + "=" * 60)
    print("Step 6: Quantizing to INT8 TFLite")
    print("=" * 60)
    quantize_tflite(saved_model_dir, tflite_path, representative_data)

    # 7. Verify.
    print("\n" + "=" * 60)
    print("Step 7: Verification")
    print("=" * 60)
    verify_no_batchnorm(tflite_path)

    if not args.skip_verify:
        verify_int8_vs_fp32(args.checkpoint, tflite_path)

    # Summary.
    fp32_size = fp32_tflite.stat().st_size / 1024 if fp32_tflite.exists() else 0
    fp16_size = fp16_tflite.stat().st_size / 1024 if fp16_tflite.exists() else 0
    int8_size = tflite_path.stat().st_size / 1024

    quant_mode = "full-integer" if representative_data else "dynamic-range"
    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  ONNX:        {onnx_path}")
    print(f"  FP32 TFLite: {fp32_tflite} ({fp32_size:.1f} KB)")
    print(f"  FP16 TFLite: {fp16_tflite} ({fp16_size:.1f} KB)")
    print(f"  INT8 TFLite: {tflite_path} ({int8_size:.1f} KB) [{quant_mode}]")
    print(f"  Compression: {fp32_size / int8_size:.1f}x" if int8_size else "")
    print("=" * 60)


if __name__ == "__main__":
    main()
