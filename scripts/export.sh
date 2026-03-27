#!/usr/bin/env bash
# Export trained model to INT8 TFLite.
#
# Usage: scripts/export.sh <checkpoint> [extra args...]
#   e.g. scripts/export.sh doom_agent_ppo.zip
#        scripts/export.sh doom_agent_ppo.zip --skip-verify
#
# Uses a separate Python 3.11 venv because TensorFlow doesn't support 3.14.
# The training venv (3.14 + ROCm torch) is unaffected.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXPORT_VENV="$REPO_ROOT/.venv-export"

CHECKPOINT="${1:?Usage: scripts/export.sh <checkpoint>}"
shift

# Find Python 3.11-3.13.
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

# Create/reuse the export venv.
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
        "tensorflow>=2.16,<2.20" \
        "tf_keras>=2.16,<2.20" \
        "ai_edge_litert" \
        "protobuf>=5.26,<6" \
        "psutil")
    touch "$EXPORT_VENV/.ready"
fi

cd "$REPO_ROOT"
exec "$EXPORT_VENV/bin/python" -m training.export --checkpoint "$CHECKPOINT" "$@"
