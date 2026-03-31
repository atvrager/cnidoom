#!/usr/bin/env bash
# Train the Doom agent through the full curriculum.
#
# Designed for large-scale training (128 envs, 256 CPU cores, GPU).
# PPO hyperparameters auto-scale with --envs (see train.py).
#
# Phases:
#   ── Arena fundamentals ──────────────────────────────────────────────
#   1.  basic                    —  500K  — aim + shoot
#   2.  deadly_corridor          —  750K  — move + shoot + dodge
#   3.  defend_the_center        —  750K  — 360° spatial awareness
#   ── Navigation + survival ───────────────────────────────────────────
#   4.  my_way_home              —  500K  — maze navigation (no enemies)
#   5.  health_gathering_supreme —  750K  — resource awareness (poison floor)
#   6.  defend_the_line          —    1M  — sustained combat (approaching waves)
#   ── E1M1 difficulty ramp ────────────────────────────────────────────
#   7.  E1M1 skill 1 (easy)     —    3M  — navigation focus (weak enemies)
#   8.  E1M1 skill 3 (medium)   —    5M  — full combat + navigation
#   9.  E1M1 skill 5 (nightmare)—    5M  — fast enemies, respawns
#   ── Multi-map generalization ────────────────────────────────────────
#  10.  E1M2 (Nuclear Plant)    —    5M  — tight corridors, acid pits
#  11.  E1M3 (Toxin Refinery)   —    5M  — open areas, multiple paths
#  12.  E1M4 (Command Control)  —    5M  — compact, enemy-dense
#   ── Consolidation ───────────────────────────────────────────────────
#  13.  E1M1 final (skill 3)    —    8M  — long polish on target level
#
# Total: ~40.25M steps  (~2.5h wall-clock at 128 envs on H100)
#
# Usage:
#   scripts/train_curriculum.sh                            # V2, 128 envs
#   scripts/train_curriculum.sh --envs 8                   # small-scale
#   scripts/train_curriculum.sh --model v2 --from 7        # resume from phase 7
#   scripts/train_curriculum.sh --envs 128 --from 10       # multi-map only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Parse flags.
START_PHASE=1
MODEL_VERSION="v2"
N_ENVS=128
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) START_PHASE="$2"; shift 2 ;;
        --model) MODEL_VERSION="$2"; shift 2 ;;
        --envs) N_ENVS="$2"; shift 2 ;;
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

echo "================================================================"
echo "  Doom Agent Curriculum Training"
echo "  Model:       $MODEL_VERSION"
echo "  Envs:        $N_ENVS"
echo "  Checkpoint:  ${CHECKPOINT}.zip"
echo "  Start phase: $START_PHASE"
echo "================================================================"

run_phase() {
    local phase="$1" cfg="$2" steps="$3" desc="$4"
    echo ""
    echo "================================================================"
    echo "  Phase $phase: $desc"
    echo "  Config: $cfg | Steps: $steps | Envs: $N_ENVS"
    echo "================================================================"
    echo ""

    local resume_flag=""
    if [[ -f "${CHECKPOINT}.zip" ]]; then
        resume_flag="--resume ${CHECKPOINT}.zip"
    fi

    uv run python -m training.train \
        --cfg "$cfg" \
        --timesteps "$steps" \
        --envs "$N_ENVS" \
        --model-version "$MODEL_VERSION" \
        --output "$CHECKPOINT" \
        $resume_flag
}

# ── Arena fundamentals ────────────────────────────────────────────────

if [[ "$START_PHASE" -le 1 ]]; then
    run_phase 1 "$SCENARIOS_DIR/basic.cfg" 500000 \
        "basic — aim + shoot"
fi

if [[ "$START_PHASE" -le 2 ]]; then
    run_phase 2 "$SCENARIOS_DIR/deadly_corridor.cfg" 750000 \
        "deadly_corridor — move + shoot + dodge"
fi

if [[ "$START_PHASE" -le 3 ]]; then
    run_phase 3 "$SCENARIOS_DIR/defend_the_center.cfg" 750000 \
        "defend_the_center — 360° spatial awareness"
fi

# ── Navigation + survival ─────────────────────────────────────────────

if [[ "$START_PHASE" -le 4 ]]; then
    run_phase 4 "$SCENARIOS_DIR/my_way_home.cfg" 500000 \
        "my_way_home — maze navigation"
fi

if [[ "$START_PHASE" -le 5 ]]; then
    run_phase 5 "$SCENARIOS_DIR/health_gathering_supreme.cfg" 750000 \
        "health_gathering_supreme — resource awareness"
fi

if [[ "$START_PHASE" -le 6 ]]; then
    run_phase 6 "$SCENARIOS_DIR/defend_the_line.cfg" 1000000 \
        "defend_the_line — sustained combat"
fi

# ── E1M1 difficulty ramp ──────────────────────────────────────────────

if [[ "$START_PHASE" -le 7 ]]; then
    run_phase 7 "$SCENARIOS_DIR/e1m1_skill1.cfg" 3000000 \
        "E1M1 easy (skill 1) — navigation focus"
fi

if [[ "$START_PHASE" -le 8 ]]; then
    run_phase 8 "$SCENARIOS_DIR/e1m1_agent.cfg" 5000000 \
        "E1M1 medium (skill 3) — combat + navigation"
fi

if [[ "$START_PHASE" -le 9 ]]; then
    run_phase 9 "$SCENARIOS_DIR/e1m1_skill5.cfg" 5000000 \
        "E1M1 nightmare (skill 5) — fast enemies, respawns"
fi

# ── Multi-map generalization ──────────────────────────────────────────

if [[ "$START_PHASE" -le 10 ]]; then
    run_phase 10 "$SCENARIOS_DIR/e1m2_agent.cfg" 5000000 \
        "E1M2 Nuclear Plant — tight corridors"
fi

if [[ "$START_PHASE" -le 11 ]]; then
    run_phase 11 "$SCENARIOS_DIR/e1m3_agent.cfg" 5000000 \
        "E1M3 Toxin Refinery — open areas"
fi

if [[ "$START_PHASE" -le 12 ]]; then
    run_phase 12 "$SCENARIOS_DIR/e1m4_agent.cfg" 5000000 \
        "E1M4 Command Control — dense combat"
fi

# ── Consolidation ─────────────────────────────────────────────────────

if [[ "$START_PHASE" -le 13 ]]; then
    run_phase 13 "$SCENARIOS_DIR/e1m1_agent.cfg" 8000000 \
        "E1M1 final polish (skill 3)"
fi

echo ""
echo "================================================================"
echo "  Curriculum complete! Final model: ${CHECKPOINT}.zip"
echo "  Total phases: 13 | ~40M steps"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  scripts/export.sh ${CHECKPOINT}.zip"
