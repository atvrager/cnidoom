"""Export trained SB3 PPO model to INT8 TFLite via ONNX → TF → TFLite.

Pipeline:
    1. Extract inference-only policy (no value head) from SB3 checkpoint
    2. Export to ONNX (opset 13)
    3. Convert ONNX → TF SavedModel (handles NCHW → NHWC)
    4. Quantize to INT8 TFLite with representative dataset
    5. Verify BatchNorm folding and INT8 vs FP32 agreement

Usage:
    uv run python -m training.export --checkpoint doom_agent_ppo.zip
"""

import argparse
from pathlib import Path

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


def export_onnx(model: InferencePolicy, onnx_path: Path) -> None:
    """Export the inference policy to ONNX format."""
    dummy_vis = torch.randn(1, 4, 45, 60)
    dummy_state = torch.randn(1, 20)

    torch.onnx.export(
        model,
        (dummy_vis, dummy_state),
        str(onnx_path),
        input_names=["visual", "state"],
        output_names=["action_probs"],
        opset_version=13,
        dynamic_axes={
            "visual": {0: "batch"},
            "state": {0: "batch"},
            "action_probs": {0: "batch"},
        },
    )
    print(f"ONNX model saved to {onnx_path}")


def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    """Convert ONNX model to TF SavedModel (handles NCHW → NHWC)."""
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        non_verbose=True,
    )
    print(f"TF SavedModel saved to {saved_model_dir}")


def collect_representative_dataset(
    n_samples: int = 200,
    cfg_path: str | None = None,
):
    """Collect calibration data from random gameplay for INT8 quantization."""
    from training.env import DoomHybridEnv

    env = DoomHybridEnv(cfg_path=cfg_path)
    obs, _ = env.reset()
    samples = []

    for _ in range(n_samples):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        samples.append(
            (
                obs["visual"][np.newaxis].astype(np.float32),
                obs["state"][np.newaxis].astype(np.float32),
            )
        )
        if done:
            obs, _ = env.reset()

    env.close()
    return samples


def quantize_tflite(
    saved_model_dir: Path,
    tflite_path: Path,
    representative_data: list[tuple[np.ndarray, np.ndarray]],
) -> bytes:
    """Convert TF SavedModel to INT8 TFLite with full-integer quantization."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for visual, state in representative_data:
            yield [visual, state]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"INT8 TFLite model saved to {tflite_path} ({size_kb:.1f} KB)")
    return tflite_model


def verify_no_batchnorm(tflite_path: Path) -> bool:
    """Verify that BatchNorm ops were folded into convolutions."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    ops = {detail["op_name"] for detail in interpreter._get_ops_details()}
    print(f"TFLite ops: {sorted(ops)}")

    bad_ops = ops & {"BATCH_NORM", "MUL", "ADD"}
    if bad_ops:
        print(f"WARNING: possible unfused BatchNorm ops: {bad_ops}")
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

    diffs = []
    for _ in range(n_samples):
        vis = np.random.rand(1, 4, 45, 60).astype(np.float32)
        state = np.random.rand(1, 20).astype(np.float32)

        # FP32 inference.
        with torch.no_grad():
            fp32_out = model(torch.from_numpy(vis), torch.from_numpy(state))
            fp32_probs = fp32_out.numpy()

        # INT8 inference — quantize inputs, dequantize output.
        for detail in input_details:
            scale, zp = detail["quantization"]
            name = detail["name"]
            data = vis if "visual" in name else state
            # ONNX→TF may have transposed visual to NHWC.
            if detail["shape"][-1] == 4 and len(detail["shape"]) == 4:
                data = np.transpose(data, (0, 2, 3, 1))  # NCHW → NHWC
            q_data = np.clip(np.round(data / scale) + zp, -128, 127).astype(np.int8)
            interpreter.set_tensor(detail["index"], q_data)

        interpreter.invoke()

        out_detail = output_details[0]
        out_scale, out_zp = out_detail["quantization"]
        int8_out = interpreter.get_tensor(out_detail["index"])
        int8_probs = (int8_out.astype(np.float32) - out_zp) * out_scale

        diffs.append(np.abs(fp32_probs - int8_probs).mean())

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
        "--cfg",
        type=str,
        default=None,
        help="VizDoom config for calibration data collection",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=200,
        help="Number of calibration samples for INT8 quantization",
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

    # 2. ONNX → TF SavedModel.
    print("\n" + "=" * 60)
    print("Step 2: Converting ONNX → TF SavedModel")
    print("=" * 60)
    onnx_to_saved_model(onnx_path, saved_model_dir)

    # 3. Collect calibration data.
    print("\n" + "=" * 60)
    print(f"Step 3: Collecting {args.calibration_samples} calibration samples")
    print("=" * 60)
    rep_data = collect_representative_dataset(
        n_samples=args.calibration_samples,
        cfg_path=args.cfg,
    )

    # 4. Quantize to INT8 TFLite.
    print("\n" + "=" * 60)
    print("Step 4: Quantizing to INT8 TFLite")
    print("=" * 60)
    quantize_tflite(saved_model_dir, tflite_path, rep_data)

    # 5. Verify.
    print("\n" + "=" * 60)
    print("Step 5: Verification")
    print("=" * 60)
    verify_no_batchnorm(tflite_path)

    if not args.skip_verify:
        verify_int8_vs_fp32(args.checkpoint, tflite_path)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  ONNX:     {onnx_path}")
    print(f"  TFLite:   {tflite_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
