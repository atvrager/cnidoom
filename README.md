# cnidoom

RL agent that learns to play Doom, trained with PPO, exported to INT8
TFLite, and run at inference time inside a bare-metal doomgeneric port
on RISC-V.

The full pipeline: VizDoom environment &rarr; Stable-Baselines3 PPO
training &rarr; ONNX &rarr; TensorFlow Lite INT8 &rarr; static C codegen
&rarr; bare-metal RISC-V ELF (or x86 host with SDL2).

```
Training (Python)          Export                Codegen             Bare-metal
 ┌──────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
 │ VizDoom  │    │ PyTorch → ONNX  │    │ TFLite → C   │    │ doomgeneric      │
 │ + SB3    │───▶│ → TF → TFLite  │───▶│ weights +    │───▶│ + agent inference│
 │ PPO      │    │ INT8 quant     │    │ graph code   │    │ on RV32 / x86   │
 └──────────┘    └─────────────────┘    └──────────────┘    └──────────────────┘
```

## Quick start

```bash
# 1. Setup
git config core.hooksPath .githooks
uv sync
git submodule update --init --recursive
scripts/download_wad.sh

# 2. Train (curriculum: basic → corridor → arena → E1M1)
scripts/train_curriculum.sh --model v2

# 3. Build static binary (x86 with SDL2 display)
scripts/build_static.sh doom_agent_v2_ppo.zip x86

# 4. Run
./build-x86/doom_agent_static_host --model ignored -iwad wads/DOOM1.WAD
```

Or for bare-metal RISC-V under QEMU:

```bash
scripts/build_static.sh doom_agent_v2_ppo.zip rv32

qemu-system-riscv32 -machine virt \
  -cpu rv32,v=true,vlen=128,zve32f=true -m 128M \
  -nographic -bios none \
  -semihosting-config enable=on,target=native \
  -kernel build-rv32/doom_agent_rv32.elf
```

## Model architecture

Two model variants, both using depthwise-separable convolutions for
compute efficiency on constrained hardware:

**Baseline** &mdash; 3 conv blocks, flatten, single dense layer:

```
Visual (4, 45, 60) NCHW
  → DWSepConv(4→16, s2) → DWSepConv(16→32, s2) → DWSepConv(32→32, s2)
  → Flatten (1536) → concat with state (20) → Dense(256, ReLU)
  → 256-dim features → policy_net → sigmoid → 6 actions
```

**V2** &mdash; 6 conv blocks, Global Average Pooling, two dense layers:

```
Visual (4, 60, 80) NCHW
  → DWSepConv(4→32, s2) → DWSepConv(32→64, s2) → DWSepConv(64→64, s1)
  → DWSepConv(64→128, s2) → DWSepConv(128→128, s1) → DWSepConv(128→192, s2)
  → GAP → (192,) → concat with state (20) = 212
  → Dense(256, ReLU) → Dense(128, ReLU)
  → 128-dim features → policy_net → sigmoid → 6 actions
```

Each `DWSepConv` block: depthwise 3&times;3 &rarr; pointwise 1&times;1
&rarr; BatchNorm &rarr; ReLU. GAP replaces the flatten layer, reducing
the feature vector from 1536 to 192 and eliminating the TRANSPOSE &rarr;
RESHAPE sequence in the exported graph.

### Observation space

| Input | Shape | Description |
|-------|-------|-------------|
| `visual` | `(4, H, W)` float | Grayscale frame stack, channels-first |
| `state` | `(20,)` float | health, armor, ammo[4], weapon_onehot[9], velocity_xy[2], reserved[3] |

### Action space

6 multi-binary actions with contradiction masking (forward+backward and
left+right cannot be simultaneously active):

| Bit | Action |
|-----|--------|
| 0 | Forward |
| 1 | Backward |
| 2 | Turn left |
| 3 | Turn right |
| 4 | Fire |
| 5 | Use |

Action repeat: 4 tics per agent decision (~8.6 decisions/sec at
35 tic/sec).

## Training

Training uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
PPO with a VizDoom environment wrapper (`training/env.py`).

```bash
# Single scenario
uv run python -m training.train --scenario basic --total-timesteps 500000

# Full curriculum (recommended)
scripts/train_curriculum.sh --model v2
```

### Curriculum

The curriculum progressively increases difficulty across 4 phases
(2.5M total steps):

| Phase | Scenario | Steps | Skill |
|-------|----------|-------|-------|
| 1 | `basic.cfg` | 500K | Single room, 1 enemy &mdash; learn to aim and shoot |
| 2 | `deadly_corridor.cfg` | 1M | Hallway with enemies &mdash; move, shoot, dodge |
| 3 | `defend_the_center.cfg` | 1M | 360&deg; arena &mdash; spatial awareness |
| 4 | `e1m1_agent.cfg` | 2.5M | Full E1M1 &mdash; navigation + combat |

Each phase resumes from the previous checkpoint. Monitor with
TensorBoard: `tensorboard --logdir tb_doom/`.

### PPO hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3&times;10<sup>-4</sup> |
| n_steps | 2048 |
| Batch size | 64 |
| Epochs | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.01 |
| Parallel envs | 8 |

### Reward shaping

Dense reward signals layered on top of the base game reward:

| Signal | Weight | Source |
|--------|--------|--------|
| Kill | &times;50 | `KILLCOUNT` delta |
| Health | &times;0.5 | `HEALTH` delta |
| Ammo | &times;0.2 | Total ammo delta |
| Movement | &times;0.01 | XY position delta |
| Time penalty | &minus;0.001 | Per step |

## Export pipeline

Converts a trained SB3 checkpoint to a fully-quantized INT8 TFLite model:

```bash
scripts/export.sh doom_agent_v2_ppo.zip --output-dir models/v2
```

Pipeline steps:

1. **Extract inference policy** &mdash; Strip value head, wrap feature extractor + policy net + sigmoid
2. **PyTorch &rarr; ONNX** &mdash; `torch.onnx.export` (opset 18)
3. **Preprocess ONNX** &mdash; Rewrite Conv pads to `auto_pad='SAME_UPPER'` for TFLite compatibility
4. **ONNX &rarr; TensorFlow** &mdash; `onnx2tf` with NCHW&rarr;NHWC auto-transpose
5. **Collect calibration data** &mdash; 200 representative samples from VizDoom
6. **INT8 quantization** &mdash; TFLite converter with full-integer quantization
7. **Verify** &mdash; Compare INT8 vs FP32 outputs across 50 random inputs

Outputs: `doom_agent.onnx`, `doom_agent_fp32.tflite`,
`doom_agent_fp16.tflite`, `doom_agent_int8.tflite`.

> **Note:** Export requires Python 3.11&ndash;3.13 (TensorFlow doesn't
> support 3.14). The export script auto-creates a separate venv if
> needed.

## Codegen

The static code generator (`tools/codegen_graph.py`) compiles a TFLite
INT8 model into plain C:

```bash
uv run python tools/codegen_graph.py \
  --model models/v2/doom_agent_int8.tflite \
  --output-dir inference/generated
```

This produces:

| File | Contents |
|------|----------|
| `doom_agent_weights.c/h` | Const weight arrays with `.cnidoom.weights` section attribute |
| `doom_agent_graph.c/h` | `run_graph()` &mdash; sequential kernel calls with liveness-optimized scratch buffer |

Key optimizations:
- **TRANSPOSE elimination** &mdash; Folds TRANSPOSE+RESHAPE into FC weight permutation
- **Liveness-based scratch allocation** &mdash; Greedy packing minimizes peak scratch (~53 KB vs 64 KB TFLM arena)
- **Section attributes** &mdash; `.cnidoom.weights` and `.cnidoom.scratch` for linker-script-based memory placement
- **LUT generation** &mdash; Pre-computed 256-entry tanh/logistic lookup tables

## Build system

The inference code builds with CMake. Key options:

```bash
cmake -B build -S inference \
  -DDOOM_AGENT_STATIC=ON \              # Enable codegen backend
  -DDOOM_AGENT_KERNEL_TARGET=riscv \    # Kernel target: generic, x86, riscv
  -DDOOM_AGENT_RV32=ON \               # Bare-metal RISC-V target
  -DDOOM_AGENT_EMBEDDED_LIB=ON \       # Build libcnidoom.a
  -DDOOM_AGENT_HOST=ON                  # SDL2 host binary (x86 only)
```

### Targets

| Target | Binary | Description |
|--------|--------|-------------|
| `doom_agent_host` | x86 | SDL2 display, TFLM from file |
| `doom_agent_static_host` | x86 | SDL2 display, static codegen (no TFLM) |
| `doom_agent_rv32.elf` | rv32 | Bare-metal RISC-V, QEMU virt + ramfb |
| `libcnidoom.a` | rv32 | Embedded library for firmware integration |

### Kernel targets

Platform-optimized implementations override generic C reference kernels:

| Target | ISA | Kernels |
|--------|-----|---------|
| `generic` | Pure C | Reference implementations for all ops |
| `x86` | AVX2 | conv2d, depthwise_conv2d, fully_connected, mean |
| `riscv` | RVV (Zve32x/f) | conv2d, depthwise_conv2d, fully_connected, mean, logistic_lut, tanh_lut |

### One-step build

`scripts/build_static.sh` runs the entire pipeline from checkpoint to
binary:

```bash
# x86 (AVX2 host, runs natively with SDL2)
scripts/build_static.sh doom_agent_v2_ppo.zip x86

# rv32 (bare-metal ELF, runs under QEMU)
scripts/build_static.sh doom_agent_v2_ppo.zip rv32
```

Steps: export &rarr; codegen &rarr; cmake configure &rarr; build &rarr;
bit-accuracy test &rarr; SDK export (rv32).

## Embedded library (`libcnidoom.a`)

For integrating Doom + agent inference into your own firmware. The
library provides weak-symbol fallbacks for all platform-specific
functions; override what you need with strong symbols.

### Public API

```c
#include "cnidoom.h"

// Run with default QEMU virt config:
int main(void) { cnidoom_run(NULL); }

// Or configure:
cnidoom_config_t cfg = cnidoom_default_config();
cfg.wad_path = "DOOM1.WAD";
cfg.clint_mtime_base = 0x200BFF8;
cnidoom_run(&cfg);
```

### Platform callbacks

Override any of these with strong symbols in your platform `.c` file:

| Callback | Default | Purpose |
|----------|---------|---------|
| `cnidoom_platform_init()` | no-op | One-time hardware init |
| `cnidoom_putc(char c)` | semihosting `SYS_WRITEC` | Console output (for printf) |
| `cnidoom_draw(fb, w, h)` | no-op | Display XRGB8888 framebuffer |
| `cnidoom_get_ticks_ms()` | CLINT mtime or semihosting | Monotonic millisecond clock |
| `cnidoom_sleep_ms(ms)` | busy-wait on `get_ticks_ms` | Sleep/delay |

### Linker sections

The library annotates performance-critical data with named sections for
memory placement on real hardware:

| Section | Contents | Typical placement | Size (V2) |
|---------|----------|-------------------|-----------|
| `.cnidoom.weights` | Model weights (const) | SRAM / flash | ~200 KB |
| `.cnidoom.scratch` | Activation scratch (r/w) | fast SRAM / DTCM | ~53 KB |
| `.cnidoom.wad` | Embedded WAD (const) | DDR / ext flash | ~4 MB |

If your linker script doesn't mention these sections, they fall into
default parents (`.rodata`, `.bss`) and still work.

### SDK export

The rv32 build automatically produces a self-contained SDK in
`build-rv32/cnidoom-sdk/`:

```
cnidoom-sdk/
  include/cnidoom.h       Public API (the only header you need)
  lib/libcnidoom.a        Merged static library (2 MB)
  lib/embedded_wad.o      DOOM1.WAD as linkable object (optional)
  cnidoom.mk              Makefile fragment
  BUILD_FLAGS             Compiler flags reference
  examples/               Working QEMU virt platform
    Makefile              Builds firmware.elf from the SDK
    qemu_main.c           Entry point
    qemu_platform.c       UART + ramfb overrides
    startup.S, linker.ld  Boot code + memory map
    uart.c, ramfb.c/h     QEMU virt drivers
```

Build from the SDK with a plain Makefile:

```makefile
CNIDOOM_SDK := path/to/cnidoom-sdk
include $(CNIDOOM_SDK)/cnidoom.mk

firmware.elf: startup.o main.o my_platform.o
	$(CC) -nostartfiles -Wl,--gc-sections -Tlinker.ld $^ $(CNIDOOM_LDFLAGS) -o $@
```

## Testing

```bash
# Python tests
uv run pytest

# C tests (host)
cmake -B build -S inference -DDOOM_AGENT_BUILD_TESTS=ON
cmake --build build && cd build && ctest

# AVX2 vs generic bit-accuracy (x86)
cmake -B build-x86 -S inference \
  -DDOOM_AGENT_STATIC=ON -DDOOM_AGENT_KERNEL_TARGET=x86
cmake --build build-x86 && ./build-x86/test_x86_bitexact

# RVV vs generic bit-accuracy (QEMU)
cmake -B build-rv32 -S inference \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv32-elf-clang.cmake \
  -DDOOM_AGENT_RV32=ON -DDOOM_AGENT_STATIC=ON \
  -DDOOM_AGENT_KERNEL_TARGET=riscv
cmake --build build-rv32
qemu-system-riscv32 -machine virt \
  -cpu rv32,v=true,vlen=128,zve32f=true -m 128M \
  -nographic -bios none \
  -semihosting-config enable=on,target=native \
  -kernel build-rv32/test_rvv_bitexact.elf

# Model comparison (golden logs)
scripts/compare_models.sh
```

## Repository layout

```
training/                 Python — PPO training, evaluation, export
  train.py                Main training script
  model.py                DoomFeatureExtractor (baseline + V2)
  env.py                  DoomHybridEnv (VizDoom wrapper)
  export.py               ONNX → TF → TFLite INT8 pipeline
  calibrate.py            Collect quantization calibration data
  scenarios/              VizDoom config files

inference/                C/C++ — doomgeneric + agent inference
  CMakeLists.txt          Build system
  doom_agent.h/c          Core agent API (init, infer, destroy)
  doom_agent_preprocess.c Frame downsampling + quantization
  doom_agent_static.c     Static codegen backend
  doom_agent_tflm.cc      TFLM interpreter backend
  doom_agent_host.cc      Host (file-based) backend
  cnidoom.h/c             Embedded library public API
  cnidoom_platform_default.c  Weak platform fallbacks
  cnidoom_syscalls.c      Portable libc stubs (semihosting)
  w_file_cnidoom.c        Unified WAD backend (embedded + stdc)
  kernels/
    generic/              Reference C kernels
    x86/                  AVX2-optimized kernels
    riscv/                RVV-optimized kernels
  platform/rv32/          QEMU virt bare-metal platform
    startup.S             Reset handler + vector unit init
    linker.ld             Memory map (ITCM/DTCM/SRAM/DDR)
    qemu_main.c           Entry point for library build
    qemu_platform.c       UART + ramfb strong overrides
    uart.c, ramfb.c       NS16550a UART, QEMU ramfb driver
    syscalls.c            Original monolithic syscalls
  generated/              Auto-generated by codegen (gitignored)

tools/
  codegen_graph.py        TFLite INT8 → static C inference code

scripts/
  download_wad.sh         Fetch shareware DOOM1.WAD
  train_curriculum.sh     Phased curriculum training
  export.sh               Checkpoint → INT8 TFLite
  build_static.sh         Full pipeline + SDK export
  compare_models.sh       Golden log comparison across models

doomgeneric/              Git submodule (do not modify)
tflite-micro/             Git submodule (do not modify)
patches/                  Submodule patches (applied at build time)
```

## Requirements

- **Python** &ge; 3.11 (training + export)
- **uv** package manager
- **VizDoom** (installed via `uv sync`)
- **CMake** &ge; 3.16 (inference build)
- **RISC-V toolchain** &mdash; auto-fetched by
  `cmake/riscv32-elf-clang.cmake` (GCC 15, newlib, RV32IMF_Zve32x_Zve32f)
- **QEMU** &mdash; `qemu-system-riscv32` for bare-metal testing
- **SDL2** &mdash; for x86 host display (optional)

## Architecture decisions

| Decision | Rationale |
|----------|-----------|
| Depthwise-separable convolutions | 8&times; fewer multiply-accumulates than standard conv |
| INT8 full-integer quantization | No float ops at inference &mdash; runs on integer-only HW |
| NCHW training, NHWC export | PyTorch-native training, TFLite-native inference |
| Static codegen over TFLM interpreter | 53 KB scratch vs 64 KB arena; no C++ runtime |
| Weak-symbol platform API | One library binary, any board &mdash; just override what you need |
| Curriculum training | Gradual difficulty prevents catastrophic forgetting |
| Embedded WAD via objcopy | Zero-copy memory-mapped access; optional semihosting fallback |
| Target: RV32IMF_Zve32x_Zve32f | Integer + single-float + vector &mdash; Google Kelvin-class RISC-V |

## License

doomgeneric is licensed under GPLv2. Shareware DOOM1.WAD is &copy; id
Software.
