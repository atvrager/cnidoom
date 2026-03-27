#!/usr/bin/env bash
# Export trained model to INT8 TFLite.
#
# Usage: scripts/export.sh <checkpoint> [extra args...]
#   e.g. scripts/export.sh doom_agent_ppo.zip
#        scripts/export.sh doom_agent_ppo.zip --skip-verify
#        scripts/export.sh doom_agent_ppo.zip --skip-calibration
#
# The script uses TWO venvs:
#   1. Training venv (.venv, Python 3.14) — collects calibration data via VizDoom
#   2. Export venv (.venv-export, Python 3.11) — runs ONNX→TF→TFLite conversion
#
# Pass --skip-calibration to skip VizDoom calibration (dynamic-range quant only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXPORT_VENV="$REPO_ROOT/.venv-export"
CALIBRATION_NPZ="$REPO_ROOT/models/calibration_data.npz"

CHECKPOINT="${1:?Usage: scripts/export.sh <checkpoint> [extra args...]}"
shift

# Parse --skip-calibration from extra args.
SKIP_CALIBRATION=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip-calibration" ]]; then
        SKIP_CALIBRATION=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# --------------------------------------------------------------------------
# Step 0: Collect calibration data in the training venv (has vizdoom).
# --------------------------------------------------------------------------
if [[ "$SKIP_CALIBRATION" -eq 0 ]]; then
    echo "Collecting calibration data with training venv..."
    uv run python -m training.calibrate -o "$CALIBRATION_NPZ"
    EXTRA_ARGS+=("--calibration-data" "$CALIBRATION_NPZ")
fi

# --------------------------------------------------------------------------
# Step 1: Find a Python 3.11-3.13 for the export venv (TF needs it).
# --------------------------------------------------------------------------
EXPORT_PYTHON=""
for v in python3.13 python3.12 python3.11; do
    if command -v "$v" &>/dev/null; then
        EXPORT_PYTHON="$v"
        break
    fi
done
# Fall back to uv-managed Python.
if [[ -z "$EXPORT_PYTHON" ]]; then
    EXPORT_PYTHON="$(uv python find '>=3.11,<3.14' 2>/dev/null || true)"
fi
if [[ -z "$EXPORT_PYTHON" ]]; then
    echo "ERROR: Need Python 3.11-3.13 for TensorFlow (3.14 not supported)."
    echo "Install one with: uv python install 3.13"
    exit 1
fi
echo "Using $EXPORT_PYTHON for export ($($EXPORT_PYTHON --version))"

# --------------------------------------------------------------------------
# Step 2: Create/reuse the export venv.
# --------------------------------------------------------------------------
# Use a marker file so a partially-created venv gets rebuilt.
if [[ ! -f "$EXPORT_VENV/.ready" ]]; then
    echo "Creating export venv at $EXPORT_VENV..."
    rm -rf "$EXPORT_VENV"
    uv venv --python "$EXPORT_PYTHON" "$EXPORT_VENV"
    # Install from /tmp to avoid project-level uv overrides in pyproject.toml.
    (cd /tmp && uv pip install --python "$EXPORT_VENV/bin/python" \
        "torch>=2.2,<2.11" \
        "stable-baselines3>=2.3" \
        "gymnasium>=1.0" \
        "numpy>=1.26" \
        "onnxscript>=0.1" \
        "onnx>=1.16,<1.17" \
        "onnx2tf>=1.25" \
        "sng4onnx" "sne4onnx" "sio4onnx" "ssi4onnx" \
        "snc4onnx" "soc4onnx" "svs4onnx" \
        "onnx_graphsurgeon==0.5.2" \
        "onnxsim>=0.4" \
        "tensorflow>=2.16,<2.20" \
        "tf_keras>=2.16,<2.20" \
        "ai_edge_litert" \
        "protobuf>=5.26,<6" \
        "psutil")
    touch "$EXPORT_VENV/.ready"
fi

# --------------------------------------------------------------------------
# Step 3: Run the export pipeline in the export venv.
# --------------------------------------------------------------------------
cd "$REPO_ROOT"
exec "$EXPORT_VENV/bin/python" -m training.export --checkpoint "$CHECKPOINT" "${EXTRA_ARGS[@]}"
