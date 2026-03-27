#!/usr/bin/env bash
# Train the Doom agent through the full curriculum.
#
# Phases (from doom-agent-project.md):
#   1. basic          — single room, 1 enemy     — 500K steps  — aim + shoot
#   2. deadly_corridor — hallway, multiple enemies — 1M steps   — move + shoot + dodge
#   3. defend_the_center — 360° enemies           — 1M steps   — spatial awareness
#   4. E1M1 full level                            — 2.5M steps — navigation + combat
#
# Each phase resumes from the previous phase's checkpoint.
# Final model is saved to doom_agent_ppo.zip.
#
# Usage:
#   scripts/train_curriculum.sh                        # baseline, full curriculum
#   scripts/train_curriculum.sh --model v2             # V2 model
#   scripts/train_curriculum.sh --model v2 --from 3    # V2, resume from phase 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Parse flags.
START_PHASE=1
MODEL_VERSION="baseline"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) START_PHASE="$2"; shift 2 ;;
        --model) MODEL_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCENARIOS_DIR="training/scenarios"
CHECKPOINT="doom_agent_ppo.zip"

echo "Model version: $MODEL_VERSION"

run_phase() {
    local phase="$1" cfg="$2" steps="$3" desc="$4"
    echo ""
    echo "================================================================"
    echo "  Phase $phase: $desc"
    echo "  Config: $cfg | Steps: $steps | Model: $MODEL_VERSION"
    echo "================================================================"
    echo ""

    local resume_flag=""
    if [[ -f "$CHECKPOINT" ]]; then
        resume_flag="--resume $CHECKPOINT"
    fi

    uv run python -m training.train \
        --cfg "$cfg" \
        --timesteps "$steps" \
        --envs 8 \
        --model-version "$MODEL_VERSION" \
        $resume_flag
}

if [[ "$START_PHASE" -le 1 ]]; then
    run_phase 1 "$SCENARIOS_DIR/basic.cfg" 500000 \
        "basic — learn to aim + shoot"
fi

if [[ "$START_PHASE" -le 2 ]]; then
    run_phase 2 "$SCENARIOS_DIR/deadly_corridor.cfg" 1000000 \
        "deadly_corridor — move + shoot + dodge"
fi

if [[ "$START_PHASE" -le 3 ]]; then
    run_phase 3 "$SCENARIOS_DIR/defend_the_center.cfg" 1000000 \
        "defend_the_center — spatial awareness"
fi

if [[ "$START_PHASE" -le 4 ]]; then
    run_phase 4 "$SCENARIOS_DIR/e1m1_agent.cfg" 2500000 \
        "E1M1 full level — navigation + combat"
fi

echo ""
echo "================================================================"
echo "  Curriculum complete! Final model: $CHECKPOINT"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  scripts/export.sh $CHECKPOINT"
