#!/usr/bin/env bash
# Compare FP32, FP16, and INT8 models by running each through the host engine
# and collecting golden logs.
#
# Usage: scripts/compare_models.sh [--frames N]
#
# Requires: build/doom_agent_host (built with -DDOOM_AGENT_HOST=ON)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOST_BIN="$REPO_ROOT/build/doom_agent_host"
MODELS_DIR="$REPO_ROOT/models"
LOGS_DIR="$REPO_ROOT/models/golden_logs"

FRAMES=500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --frames) FRAMES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$HOST_BIN" ]]; then
    echo "ERROR: $HOST_BIN not found. Build with:"
    echo "  cmake -B build -S inference -DDOOM_AGENT_HOST=ON"
    echo "  cmake --build build"
    exit 1
fi

mkdir -p "$LOGS_DIR"

# Model variants to compare.
declare -A MODELS
MODELS[fp32]="$MODELS_DIR/doom_agent_tf/doom_agent_float32.tflite"
MODELS[fp16]="$MODELS_DIR/doom_agent_tf/doom_agent_float16.tflite"
MODELS[int8]="$MODELS_DIR/doom_agent_int8.tflite"

echo "================================================================"
echo "  Model Comparison ($FRAMES frames each)"
echo "================================================================"
echo ""

for variant in fp32 fp16 int8; do
    model="${MODELS[$variant]}"
    if [[ ! -f "$model" ]]; then
        echo "  $variant: SKIPPED (${model} not found)"
        continue
    fi

    log="$LOGS_DIR/${variant}.csv"
    size_kb=$(( $(stat -c%s "$model") / 1024 ))

    echo -n "  $variant (${size_kb}KB): running..."

    # Run the host engine headless-ish (it will exit after the game ends
    # or we can kill it after N frames).
    DOOM_GOLDEN_LOG="$log" timeout 60 "$HOST_BIN" \
        --model "$model" -iwad "$REPO_ROOT/wads/DOOM1.WAD" \
        2>/dev/null || true

    if [[ -f "$log" ]]; then
        n_rows=$(( $(wc -l < "$log") - 1 ))  # subtract header
        # Compute average and p99 inference time.
        stats=$(awk -F, 'NR > 1 {
            sum += $NF; n++; times[n] = $NF
        } END {
            if (n == 0) { print "0 0"; exit }
            avg = sum / n
            asort(times)
            p99_idx = int(n * 0.99)
            if (p99_idx < 1) p99_idx = 1
            printf "%.0f %.0f", avg, times[p99_idx]
        }' "$log" 2>/dev/null || echo "0 0")

        avg_us=$(echo "$stats" | awk '{print $1}')
        p99_us=$(echo "$stats" | awk '{print $2}')
        echo " $n_rows inferences, avg=${avg_us}us, p99=${p99_us}us"
    else
        echo " no log generated"
    fi
done

echo ""

# Compare action agreement between INT8 and FP32.
fp32_log="$LOGS_DIR/fp32.csv"
int8_log="$LOGS_DIR/int8.csv"

if [[ -f "$fp32_log" && -f "$int8_log" ]]; then
    agreement=$(paste -d, "$fp32_log" "$int8_log" | awk -F, 'NR > 1 {
        n++
        if ($2 == $10) agree++  # action_bits columns
    } END {
        if (n > 0) printf "%.1f", agree / n * 100
        else print "N/A"
    }')
    echo "  INT8 vs FP32 action agreement: ${agreement}%"
fi

echo ""
echo "Golden logs saved to $LOGS_DIR/"
