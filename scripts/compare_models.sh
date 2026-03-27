#!/usr/bin/env bash
# Compare model variants by running each through the host inference engine
# and analyzing golden logs.
#
# Compares across two dimensions:
#   1. Quantization: FP32 vs FP16 vs INT8 (same architecture)
#   2. Architecture: baseline vs V2 (both INT8)
#
# For each variant, reports:
#   - Model size (KB)
#   - Inference latency (avg, p50, p99)
#   - Action probability distributions
#   - Action agreement vs reference (FP32 baseline)
#
# Usage:
#   scripts/compare_models.sh [--frames N] [--models-dir DIR]
#
# Requires: build/doom_agent_host (built with -DDOOM_AGENT_HOST=ON)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOST_BIN="$REPO_ROOT/build/doom_agent_host"
MODELS_DIR="$REPO_ROOT/models"
LOGS_DIR="$REPO_ROOT/models/golden_logs"
WAD="$REPO_ROOT/wads/DOOM1.WAD"

FRAMES=500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --frames) FRAMES="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$HOST_BIN" ]]; then
    echo "ERROR: $HOST_BIN not found. Build with:"
    echo "  cmake -B build -S inference -DDOOM_AGENT_HOST=ON"
    echo "  cmake --build build"
    exit 1
fi

if [[ ! -f "$WAD" ]]; then
    echo "ERROR: $WAD not found. Run: scripts/download_wad.sh"
    exit 1
fi

mkdir -p "$LOGS_DIR"

# ----------------------------------------------------------------
# Discover available models
# ----------------------------------------------------------------
declare -A MODEL_FILES
declare -A MODEL_LABELS

# Baseline quantization variants.
for f in "$MODELS_DIR/doom_agent_tf/doom_agent_float32.tflite" \
         "$MODELS_DIR/doom_agent_fp32.tflite"; do
    [[ -f "$f" ]] && MODEL_FILES[baseline_fp32]="$f" && break
done
for f in "$MODELS_DIR/doom_agent_tf/doom_agent_float16.tflite" \
         "$MODELS_DIR/doom_agent_fp16.tflite"; do
    [[ -f "$f" ]] && MODEL_FILES[baseline_fp16]="$f" && break
done
[[ -f "$MODELS_DIR/doom_agent_int8.tflite" ]] && \
    MODEL_FILES[baseline_int8]="$MODELS_DIR/doom_agent_int8.tflite"

# V2 variants.
[[ -f "$MODELS_DIR/doom_agent_v2_int8.tflite" ]] && \
    MODEL_FILES[v2_int8]="$MODELS_DIR/doom_agent_v2_int8.tflite"
[[ -f "$MODELS_DIR/doom_agent_v2_fp32.tflite" ]] && \
    MODEL_FILES[v2_fp32]="$MODELS_DIR/doom_agent_v2_fp32.tflite"
for f in "$MODELS_DIR/doom_agent_v2_tf/doom_agent_float32.tflite"; do
    [[ -f "$f" ]] && MODEL_FILES[v2_fp32]="$f" && break
done

MODEL_LABELS[baseline_fp32]="Baseline FP32"
MODEL_LABELS[baseline_fp16]="Baseline FP16"
MODEL_LABELS[baseline_int8]="Baseline INT8"
MODEL_LABELS[v2_fp32]="V2 FP32"
MODEL_LABELS[v2_int8]="V2 INT8"

# Preferred display order.
ALL_VARIANTS=(baseline_fp32 baseline_fp16 baseline_int8 v2_fp32 v2_int8)

found=0
for v in "${ALL_VARIANTS[@]}"; do
    [[ -n "${MODEL_FILES[$v]:-}" ]] && found=$((found + 1))
done

if [[ "$found" -eq 0 ]]; then
    echo "ERROR: No model files found in $MODELS_DIR"
    echo "  Expected: doom_agent_int8.tflite, doom_agent_v2_int8.tflite, etc."
    echo "  Run the export pipeline first:"
    echo "    uv run python -m training.export --checkpoint doom_agent_ppo.zip"
    exit 1
fi

# ----------------------------------------------------------------
# Run each variant
# ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Model Comparison — $FRAMES frames, $found variant(s)"
echo "================================================================"

declare -A LOG_FILES
declare -A SIZES

for variant in "${ALL_VARIANTS[@]}"; do
    model="${MODEL_FILES[$variant]:-}"
    [[ -z "$model" ]] && continue

    label="${MODEL_LABELS[$variant]}"
    log="$LOGS_DIR/${variant}.csv"
    size_kb=$(( $(stat -c%s "$model") / 1024 ))
    SIZES[$variant]=$size_kb

    echo ""
    echo -n "  $label (${size_kb} KB): running $FRAMES frames..."

    DOOM_GOLDEN_LOG="$log" timeout 120 "$HOST_BIN" \
        --model "$model" -iwad "$WAD" \
        2>/dev/null || true

    if [[ -f "$log" ]]; then
        LOG_FILES[$variant]="$log"
        echo " done"
    else
        echo " no log (model may have failed to load)"
    fi
done

# ----------------------------------------------------------------
# Analyze results
# ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Results"
echo "================================================================"
echo ""

# Print table header.
printf "  %-16s %7s %8s %8s %8s %8s %6s\n" \
    "Variant" "Size" "Infer" "Avg(us)" "P50(us)" "P99(us)" "Agree"
printf "  %-16s %7s %8s %8s %8s %8s %6s\n" \
    "----------------" "-------" "--------" "--------" "--------" "--------" "------"

# Reference log for agreement calculation (prefer baseline FP32).
ref_variant=""
for rv in baseline_fp32 baseline_int8 v2_fp32 v2_int8; do
    if [[ -n "${LOG_FILES[$rv]:-}" ]]; then
        ref_variant="$rv"
        break
    fi
done

for variant in "${ALL_VARIANTS[@]}"; do
    log="${LOG_FILES[$variant]:-}"
    [[ -z "$log" ]] && continue

    label="${MODEL_LABELS[$variant]}"
    size_kb="${SIZES[$variant]}"

    # Compute latency stats: count, avg, p50, p99.
    read -r n_infer avg_us p50_us p99_us <<< "$(awk -F, 'NR > 1 {
        sum += $NF; n++; times[n] = $NF
    } END {
        if (n == 0) { print "0 0 0 0"; exit }
        avg = sum / n
        asort(times)
        p50_idx = int(n * 0.50); if (p50_idx < 1) p50_idx = 1
        p99_idx = int(n * 0.99); if (p99_idx < 1) p99_idx = 1
        printf "%d %.0f %.0f %.0f", n, avg, times[p50_idx], times[p99_idx]
    }' "$log" 2>/dev/null || echo "0 0 0 0")"

    # Compute action agreement vs reference.
    agree="ref"
    if [[ "$variant" != "$ref_variant" && -n "${LOG_FILES[$ref_variant]:-}" ]]; then
        ref_log="${LOG_FILES[$ref_variant]}"
        agree=$(paste -d, "$ref_log" "$log" | awk -F, 'NR > 1 {
            n++
            if ($2 == $10) agree++
        } END {
            if (n > 0) printf "%.1f%%", agree / n * 100
            else print "N/A"
        }' 2>/dev/null || echo "N/A")
    fi

    printf "  %-16s %5dKB %8d %8s %8s %8s %6s\n" \
        "$label" "$size_kb" "$n_infer" "$avg_us" "$p50_us" "$p99_us" "$agree"
done

# ----------------------------------------------------------------
# Per-action probability comparison (if multiple variants exist)
# ----------------------------------------------------------------
action_names=("fwd" "bwd" "left" "right" "fire" "use")

# Only show if we have at least 2 logs.
log_count=${#LOG_FILES[@]}
if [[ "$log_count" -ge 2 ]]; then
    echo ""
    echo "  Mean action probabilities:"
    printf "  %-16s" ""
    for a in "${action_names[@]}"; do
        printf " %7s" "$a"
    done
    echo ""
    printf "  %-16s" "----------------"
    for _ in "${action_names[@]}"; do
        printf " %7s" "-------"
    done
    echo ""

    for variant in "${ALL_VARIANTS[@]}"; do
        log="${LOG_FILES[$variant]:-}"
        [[ -z "$log" ]] && continue

        label="${MODEL_LABELS[$variant]}"
        probs=$(awk -F, 'NR > 1 {
            for (i = 3; i <= 8; i++) { sums[i] += $i; }
            n++
        } END {
            for (i = 3; i <= 8; i++) {
                printf "%.3f ", (n > 0 ? sums[i] / n : 0)
            }
        }' "$log" 2>/dev/null || echo "0 0 0 0 0 0")

        printf "  %-16s" "$label"
        for p in $probs; do
            printf " %7s" "$p"
        done
        echo ""
    done
fi

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
echo ""
echo "  Reference for agreement: ${MODEL_LABELS[$ref_variant]:-none}"
echo "  Golden logs: $LOGS_DIR/"
echo ""
