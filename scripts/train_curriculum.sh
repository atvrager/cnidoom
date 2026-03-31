#!/usr/bin/env bash
# Train the Doom agent through the full curriculum.
#
# Phases:
#   1. basic                   — single room, 1 enemy      —  500K  — aim + shoot
#   2. deadly_corridor         — hallway, enemies           —  750K  — move + shoot + dodge
#   3. defend_the_center       — 360° arena                 —  750K  — spatial awareness
#   4. my_way_home             — maze, no enemies           —  500K  — navigation
#   5. health_gathering_supreme — poison floor, health packs —  750K  — resource awareness
#   6. defend_the_line         — waves of enemies           —    1M  — sustained combat
#   7. E1M1 short (1 min)     — starting area               —    2M  — E1M1 warmup
#   8. E1M1 full  (2 min)     — full level                  —    4M  — navigation + combat
#
# Total: ~10.25M steps  (old curriculum: 5M)
#
# Each phase resumes from the previous phase's checkpoint.
# Final model is saved to doom_agent_{version}_ppo.zip.
#
# Usage:
#   scripts/train_curriculum.sh                        # V2, full curriculum
#   scripts/train_curriculum.sh --model baseline       # baseline model
#   scripts/train_curriculum.sh --model v2 --from 5    # V2, resume from phase 5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Parse flags.
START_PHASE=1
MODEL_VERSION="v2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) START_PHASE="$2"; shift 2 ;;
        --model) MODEL_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCENARIOS_DIR="training/scenarios"

# Version-specific checkpoint so baseline and V2 don't overwrite each other.
if [[ "$MODEL_VERSION" == "baseline" ]]; then
    CHECKPOINT="doom_agent_ppo"
else
    CHECKPOINT="doom_agent_${MODEL_VERSION}_ppo"
fi

echo "Model version: $MODEL_VERSION"
echo "Checkpoint:    ${CHECKPOINT}.zip"

run_phase() {
    local phase="$1" cfg="$2" steps="$3" desc="$4"
    echo ""
    echo "================================================================"
    echo "  Phase $phase: $desc"
    echo "  Config: $cfg | Steps: $steps | Model: $MODEL_VERSION"
    echo "================================================================"
    echo ""

    local resume_flag=""
    if [[ -f "${CHECKPOINT}.zip" ]]; then
        resume_flag="--resume ${CHECKPOINT}.zip"
    fi

    uv run python -m training.train \
        --cfg "$cfg" \
        --timesteps "$steps" \
        --envs 8 \
        --model-version "$MODEL_VERSION" \
        --output "$CHECKPOINT" \
        $resume_flag
}

# Phase 1: Learn to aim and shoot (single room, one enemy)
if [[ "$START_PHASE" -le 1 ]]; then
    run_phase 1 "$SCENARIOS_DIR/basic.cfg" 500000 \
        "basic — learn to aim + shoot"
fi

# Phase 2: Move, shoot, dodge in a hallway
if [[ "$START_PHASE" -le 2 ]]; then
    run_phase 2 "$SCENARIOS_DIR/deadly_corridor.cfg" 750000 \
        "deadly_corridor — move + shoot + dodge"
fi

# Phase 3: 360° spatial awareness in an arena
if [[ "$START_PHASE" -le 3 ]]; then
    run_phase 3 "$SCENARIOS_DIR/defend_the_center.cfg" 750000 \
        "defend_the_center — spatial awareness"
fi

# Phase 4: Pure navigation — find the goal in a maze (no enemies)
if [[ "$START_PHASE" -le 4 ]]; then
    run_phase 4 "$SCENARIOS_DIR/my_way_home.cfg" 500000 \
        "my_way_home — maze navigation"
fi

# Phase 5: Survive poison floor by collecting health packs
if [[ "$START_PHASE" -le 5 ]]; then
    run_phase 5 "$SCENARIOS_DIR/health_gathering_supreme.cfg" 750000 \
        "health_gathering_supreme — resource awareness"
fi

# Phase 6: Sustained combat against waves of approaching enemies
if [[ "$START_PHASE" -le 6 ]]; then
    run_phase 6 "$SCENARIOS_DIR/defend_the_line.cfg" 1000000 \
        "defend_the_line — sustained combat"
fi

# Phase 7: E1M1 warmup — short episodes (1 min) to learn the start area
if [[ "$START_PHASE" -le 7 ]]; then
    run_phase 7 "$SCENARIOS_DIR/e1m1_short.cfg" 2000000 \
        "E1M1 short — learn starting area"
fi

# Phase 8: E1M1 full — longer episodes (2 min) for deep exploration
if [[ "$START_PHASE" -le 8 ]]; then
    run_phase 8 "$SCENARIOS_DIR/e1m1_agent.cfg" 4000000 \
        "E1M1 full — navigation + combat"
fi

echo ""
echo "================================================================"
echo "  Curriculum complete! Final model: ${CHECKPOINT}.zip"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  scripts/export.sh ${CHECKPOINT}.zip"
