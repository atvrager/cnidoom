# cnidoom

Reinforcement-learning agent that learns to play Doom, trained with PPO via Stable-Baselines3 + VizDoom, exported to INT8 TensorFlow Lite, and run at inference time inside a doomgeneric port using LiteRT Micro on a RISC-V RV32IMF_Zve32x_Zve32f target.

## Repo layout

```
training/          # Python – PPO training, evaluation, export
inference/         # C/C++ – doomgeneric integration + TFLite Micro inference
doomgeneric/       # git submodule (do not modify directly)
tflite-micro/      # git submodule (do not modify directly)
patches/           # Patch files for submodule modifications
models/            # Exported models (gitignored)
scripts/           # Helper scripts (WAD download, etc.)
.githooks/         # Git hooks (pre-commit, commit-msg)
```

## Setup

```bash
# 1. Activate hooks
git config core.hooksPath .githooks

# 2. Install Python deps
uv sync

# 3. Init submodules
git submodule update --init --recursive

# 4. Download shareware WAD (DOOM1.WAD)
scripts/download_wad.sh
```

## Build

**Python (training/export):**
```bash
uv run python -m training.train
uv run python -m training.export
```

**C/C++ (inference):**
```bash
cmake -B build -S inference
cmake --build build
```

## Code style

- **Python**: ruff defaults (lint + format). Run via `uvx ruff check` and `uvx ruff format --check`.
- **C/C++**: Google style via clang-format (see `.clang-format`).
- **Commits**: strict conventional commits — `type(scope): description`. Types: feat, fix, chore, docs, refactor, test, ci, build, perf, style.

## Architecture decisions

- **Frame format**: NCHW during training (PyTorch), NHWC at export (TFLite).
- **Quantization**: INT8 full-integer quantization for TFLite export.
- **CNN backbone**: depthwise-separable convolutions (efficient on constrained HW).
- **Observation**: 4-frame stack at 60×45 resolution + 20-float state vector.
- **Actions**: 6 multi-binary actions with contradiction masking (e.g., no left+right).
- **Action repeat**: 4 tics per agent decision.
- **WAD**: shareware DOOM1.WAD by default (configurable via env var).
- **Target**: RISC-V RV32IMF_Zve32x_Zve32f.

## Submodule policy

Never modify submodule contents directly. If changes are needed, create a patch in `patches/` and apply it during the build.

## Package manager

uv — use `uv run`, `uv sync`, `uv add`.

## Testing

- **Python**: pytest (`uv run pytest`).
- **C/C++**: CTest (`cd build && ctest`).
- **Export verification**: compare INT8 vs FP32 model outputs; agreement within quantization tolerance.
