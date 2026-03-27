#!/usr/bin/env bash
# Build a static inference binary from a trained checkpoint.
#
# Full pipeline: checkpoint → ONNX → TFLite INT8 → codegen → compile.
#
# Usage:
#   scripts/build_static.sh <checkpoint> <target>
#
# Targets:
#   x86    — AVX2-optimized host binary (runs natively with SDL2)
#   rv32   — Bare-metal RISC-V 32-bit ELF (runs under QEMU)
#
# Examples:
#   scripts/build_static.sh doom_agent_v2_ppo.zip x86
#   scripts/build_static.sh doom_agent_ppo.zip rv32
#   scripts/build_static.sh doom_agent_v2_ppo.zip x86 --skip-calibration
#
# The script assumes onnx2tf / tensorflow are available (either in the
# export venv created by scripts/export.sh, or in the current env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------
# Parse arguments
# --------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: scripts/build_static.sh <checkpoint> <target> [export args...]"
    echo ""
    echo "Targets: x86, rv32"
    echo ""
    echo "Extra args are passed to scripts/export.sh:"
    echo "  --skip-calibration   Skip VizDoom calibration (dynamic-range quant)"
    echo "  --skip-verify        Skip INT8 vs FP32 verification"
    exit 1
fi

CHECKPOINT="$1"
TARGET="$2"
shift 2

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

case "$TARGET" in
    x86)
        BUILD_DIR="build-x86"
        KERNEL_TARGET="x86"
        CMAKE_EXTRA_ARGS=""
        ;;
    rv32)
        BUILD_DIR="build-rv32"
        KERNEL_TARGET="riscv"
        CMAKE_EXTRA_ARGS="-DCMAKE_TOOLCHAIN_FILE=$REPO_ROOT/cmake/riscv32-elf-clang.cmake -DDOOM_AGENT_RV32=ON -DDOOM_AGENT_EMBEDDED_LIB=ON"
        ;;
    *)
        echo "ERROR: Unknown target '$TARGET'. Use 'x86' or 'rv32'."
        exit 1
        ;;
esac

# Derive output dir from checkpoint name (e.g. doom_agent_v2_ppo.zip → models/v2).
BASENAME="$(basename "$CHECKPOINT" .zip)"
if [[ "$BASENAME" == *"v2"* ]]; then
    MODELS_DIR="models/v2"
else
    MODELS_DIR="models"
fi
TFLITE="$MODELS_DIR/doom_agent_int8.tflite"

echo "================================================================"
echo "  Static Build Pipeline"
echo "  Checkpoint: $CHECKPOINT"
echo "  Target:     $TARGET ($KERNEL_TARGET kernels)"
echo "  Build dir:  $BUILD_DIR"
echo "  Models dir: $MODELS_DIR"
echo "================================================================"

# --------------------------------------------------------------------------
# Step 1: Export checkpoint → INT8 TFLite
# --------------------------------------------------------------------------
if [[ -f "$TFLITE" ]]; then
    echo ""
    echo "[Step 1] INT8 TFLite already exists: $TFLITE (skipping export)"
    echo "         Delete it to force re-export."
else
    echo ""
    echo "[Step 1] Exporting checkpoint → INT8 TFLite"
    echo ""
    scripts/export.sh "$CHECKPOINT" --output-dir "$MODELS_DIR" "$@"
fi

if [[ ! -f "$TFLITE" ]]; then
    echo "ERROR: Export did not produce $TFLITE"
    exit 1
fi

echo ""
echo "[Step 1] INT8 TFLite: $TFLITE ($(( $(stat -c%s "$TFLITE") / 1024 )) KB)"

# --------------------------------------------------------------------------
# Step 2: Codegen — TFLite → static C inference code
# --------------------------------------------------------------------------
GENERATED_DIR="inference/generated"
echo ""
echo "[Step 2] Running codegen: $TFLITE → $GENERATED_DIR/"
echo ""

uv run python tools/codegen_graph.py \
    --model "$TFLITE" \
    --output-dir "$GENERATED_DIR"

# --------------------------------------------------------------------------
# Step 3: CMake configure + build
# --------------------------------------------------------------------------
echo ""
echo "[Step 3] Building $TARGET target ($BUILD_DIR)"
echo ""

# DOOM_AGENT_HOST enables the SDL2 host binary (x86 only).
# rv32 is bare-metal — no SDL2, no TFLM host dependency.
HOST_FLAG=""
if [[ "$TARGET" == "x86" ]]; then
    HOST_FLAG="-DDOOM_AGENT_HOST=ON"
fi

cmake -B "$BUILD_DIR" -S inference \
    -DDOOM_AGENT_STATIC=ON \
    -DDOOM_AGENT_KERNEL_TARGET="$KERNEL_TARGET" \
    -DDOOM_AGENT_MODEL="$TFLITE" \
    $HOST_FLAG \
    $CMAKE_EXTRA_ARGS

cmake --build "$BUILD_DIR" -j"$(nproc)"

# --------------------------------------------------------------------------
# Step 4: Run bit-accuracy test
# --------------------------------------------------------------------------
echo ""
echo "[Step 4] Running bit-accuracy test"
echo ""

case "$TARGET" in
    x86)
        if [[ -x "$BUILD_DIR/test_x86_bitexact" ]]; then
            "$BUILD_DIR/test_x86_bitexact"
        else
            echo "  (test_x86_bitexact not found — skipping)"
        fi
        ;;
    rv32)
        if [[ -x "$BUILD_DIR/test_rvv_bitexact.elf" ]] && command -v qemu-system-riscv32 &>/dev/null; then
            timeout 60 qemu-system-riscv32 \
                -machine virt \
                -cpu rv32,v=true,vlen=128,zve32f=true \
                -m 128M \
                -nographic \
                -bios none \
                -semihosting-config enable=on,target=native \
                -kernel "$BUILD_DIR/test_rvv_bitexact.elf" || true
        else
            echo "  (test_rvv_bitexact.elf or qemu-system-riscv32 not found — skipping)"
        fi
        ;;
esac

# --------------------------------------------------------------------------
# Step 5: Export library package (rv32 only)
# --------------------------------------------------------------------------
EXPORT_DIR=""
if [[ "$TARGET" == "rv32" && -f "$BUILD_DIR/libcnidoom.a" ]]; then
    EXPORT_DIR="$BUILD_DIR/cnidoom-sdk"
    echo ""
    echo "[Step 5] Exporting library package → $EXPORT_DIR/"
    echo ""

    rm -rf "$EXPORT_DIR"
    mkdir -p "$EXPORT_DIR/include" "$EXPORT_DIR/lib" "$EXPORT_DIR/examples"

    # Merge all static libraries into a single self-contained archive.
    # libcnidoom.a has the engine + glue, but the agent inference chain
    # spans doom_agent_static → doom_agent_core → doom_agent_preprocess.
    # Extract all .o files and re-archive into one libcnidoom.a.
    AR="${AR:-$(find "$BUILD_DIR" -path '*/CMakeFiles' -prune -o -name 'riscv32-*-ar' -print 2>/dev/null | head -1)}"
    if [[ -z "$AR" ]]; then
        # Fall back: read AR from CMake cache.
        AR="$(grep CMAKE_AR "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)"
    fi
    RANLIB="${RANLIB:-${AR/%ar/ranlib}}"

    MERGE_TMP="$(mktemp -d)"
    trap 'rm -rf "$MERGE_TMP"' EXIT
    for lib in "$BUILD_DIR"/libcnidoom.a \
               "$BUILD_DIR"/libdoom_agent_static.a \
               "$BUILD_DIR"/libdoom_agent_core.a \
               "$BUILD_DIR"/libdoom_agent_preprocess.a; do
        if [[ -f "$lib" ]]; then
            # Extract into a per-lib subdir to avoid .o name collisions.
            subdir="$MERGE_TMP/$(basename "$lib" .a)"
            mkdir -p "$subdir"
            (cd "$subdir" && "$AR" x "$REPO_ROOT/$lib")
        fi
    done
    "$AR" rcs "$EXPORT_DIR/lib/libcnidoom.a" "$MERGE_TMP"/*/*.obj "$MERGE_TMP"/*/*.o 2>/dev/null || \
    "$AR" rcs "$EXPORT_DIR/lib/libcnidoom.a" "$MERGE_TMP"/*/*.obj 2>/dev/null || \
    "$AR" rcs "$EXPORT_DIR/lib/libcnidoom.a" "$MERGE_TMP"/*/*.o
    "$RANLIB" "$EXPORT_DIR/lib/libcnidoom.a"
    rm -rf "$MERGE_TMP"
    trap - EXIT

    # Public header (the only header downstream needs)
    cp inference/cnidoom.h "$EXPORT_DIR/include/"

    # Embedded WAD object (optional — user can also use semihosting)
    if [[ -f "$BUILD_DIR/embedded_wad.o" ]]; then
        cp "$BUILD_DIR/embedded_wad.o" "$EXPORT_DIR/lib/"
    fi

    # Example files: QEMU platform (reference for writing your own)
    cp inference/platform/rv32/qemu_main.c     "$EXPORT_DIR/examples/"
    cp inference/platform/rv32/qemu_platform.c "$EXPORT_DIR/examples/"
    cp inference/platform/rv32/startup.S       "$EXPORT_DIR/examples/"
    cp inference/platform/rv32/linker.ld       "$EXPORT_DIR/examples/"
    cp inference/platform/rv32/uart.c          "$EXPORT_DIR/examples/"
    cp inference/platform/rv32/ramfb.c         "$EXPORT_DIR/examples/"
    cp inference/platform/rv32/ramfb.h         "$EXPORT_DIR/examples/"

    # ── Example Makefile (builds QEMU target from examples/) ──────────
    cat > "$EXPORT_DIR/examples/Makefile" << 'EXEMK'
# Example Makefile: builds a QEMU virt firmware from cnidoom SDK.
# Usage: make CNIDOOM_SDK=../  (from within the examples/ directory)

CNIDOOM_SDK ?= ..
include $(CNIDOOM_SDK)/cnidoom.mk

CROSS   ?= riscv32-unknown-elf-
CC      := $(CROSS)gcc
AS      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy

ARCH    := -march=rv32imf_zve32x_zve32f -mabi=ilp32
CFLAGS  := $(ARCH) -Os -ffunction-sections -fdata-sections $(CNIDOOM_CFLAGS)
ASFLAGS := $(ARCH)
LDFLAGS := -nostartfiles -Wl,--gc-sections -Wl,--no-warn-rwx-segments

SRCS := qemu_main.c qemu_platform.c uart.c ramfb.c
OBJS := startup.o $(SRCS:.c=.o)

firmware.elf: $(OBJS)
	$(CC) $(LDFLAGS) -Tlinker.ld $^ $(CNIDOOM_LDFLAGS) -o $@
	@echo "Built $@ ($(shell $(CROSS)size $@ | tail -1 | awk '{print $$1}') bytes text)"

%.o: %.S
	$(AS) $(ASFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o firmware.elf

.PHONY: clean
EXEMK

    # Compiler / arch flags used to build the library (for reproducibility).
    ARCH_FLAGS="-march=rv32imf_zve32x_zve32f -mabi=ilp32"
    COMMON_CFLAGS="$ARCH_FLAGS -fno-exceptions -fno-unwind-tables -ffunction-sections -fdata-sections"
    COMMON_LDFLAGS="-nostartfiles -Wl,--gc-sections"

    # ── Makefile fragment ─────────────────────────────────────────────
    cat > "$EXPORT_DIR/cnidoom.mk" << 'MKEOF'
# cnidoom.mk — Makefile fragment for linking against libcnidoom.a
#
# Include this from your project Makefile:
#   CNIDOOM_SDK := path/to/cnidoom-sdk
#   include $(CNIDOOM_SDK)/cnidoom.mk
#
# Then use the variables it defines:
#   $(CC) $(CNIDOOM_CFLAGS) -c my_platform.c -o my_platform.o
#   $(CC) $(CNIDOOM_LDFLAGS) startup.o main.o my_platform.o -o firmware.elf
#
# The library provides weak defaults for all platform callbacks.
# Override any of these with strong symbols in your own .c file:
#   cnidoom_draw()          — display framebuffer
#   cnidoom_get_ticks_ms()  — monotonic millisecond clock
#   cnidoom_sleep_ms()      — sleep/busy-wait
#   cnidoom_putc()          — console character output
#   cnidoom_platform_init() — one-time hardware init

CNIDOOM_SDK  ?= $(dir $(lastword $(MAKEFILE_LIST)))
CNIDOOM_LIB  := $(CNIDOOM_SDK)/lib/libcnidoom.a
CNIDOOM_WAD  := $(wildcard $(CNIDOOM_SDK)/lib/embedded_wad.o)

CNIDOOM_CFLAGS  := -I$(CNIDOOM_SDK)/include
CNIDOOM_LDFLAGS := -Wl,--whole-archive $(CNIDOOM_LIB) -Wl,--no-whole-archive \
                   $(CNIDOOM_WAD) -lc -lm -lgcc
MKEOF

    # ── Build flags reference ─────────────────────────────────────────
    cat > "$EXPORT_DIR/BUILD_FLAGS" << EOF
# Compiler and linker flags used to build libcnidoom.a.
# Your firmware must use compatible flags (same arch/ABI).

ARCH_FLAGS=$ARCH_FLAGS
CFLAGS=$COMMON_CFLAGS
LDFLAGS=$COMMON_LDFLAGS

# Toolchain: riscv32-unknown-elf-gcc (Kelvin v2, GCC 15, newlib)
# Target:    RV32IMF_Zve32x_Zve32f, ilp32 ABI
EOF

    # ── README ────────────────────────────────────────────────────────
    cat > "$EXPORT_DIR/README" << 'EOF'
cnidoom SDK — Doom + RL Agent for Embedded RISC-V
==================================================

Contents:
  include/cnidoom.h     Public API (the only header you need)
  lib/libcnidoom.a      Static library (Doom engine + agent inference)
  lib/embedded_wad.o    DOOM1.WAD as a linkable object (optional)
  cnidoom.mk            Makefile fragment with compiler/linker flags
  BUILD_FLAGS           Exact flags used to build the library
  examples/             QEMU virt reference platform (startup, linker
                        script, UART, ramfb, platform overrides)

Quick start (bare Makefile):

  1. Write main.c:
       #include "cnidoom.h"
       int main(void) { cnidoom_run(NULL); }

  2. Write my_platform.c with strong overrides for your board:
       void cnidoom_platform_init(void) { my_uart_init(); }
       void cnidoom_putc(char c)        { my_uart_putc(c); }
       void cnidoom_draw(const uint32_t* fb, int w, int h) { ... }

  3. Provide startup.S + linker.ld for your memory map.
     See examples/ for a working reference (QEMU virt).

     Linker section hints (optional — improves perf on real HW):
       .cnidoom.weights  → SRAM / flash   (const, ~200 KB)
       .cnidoom.scratch  → fast SRAM      (r/w,   ~53 KB)
       .cnidoom.wad      → DDR / ext flash (const, ~4 MB)

  4. Build:
       CNIDOOM_SDK=path/to/cnidoom-sdk
       include $(CNIDOOM_SDK)/cnidoom.mk

       riscv32-unknown-elf-gcc $(CNIDOOM_CFLAGS) -c main.c
       riscv32-unknown-elf-gcc $(CNIDOOM_CFLAGS) -c my_platform.c
       riscv32-unknown-elf-as startup.S -o startup.o
       riscv32-unknown-elf-gcc -T linker.ld -nostartfiles -Wl,--gc-sections \
           startup.o main.o my_platform.o $(CNIDOOM_LDFLAGS) -o firmware.elf

  Without embedded_wad.o, the library falls back to semihosting file
  I/O (reads wads/DOOM1.WAD from the host filesystem at runtime).

Platform callbacks (all optional — weak defaults provided):
  cnidoom_platform_init()  — called once at startup
  cnidoom_putc(c)          — character output (default: semihost)
  cnidoom_draw(fb, w, h)   — display frame (default: no-op)
  cnidoom_get_ticks_ms()   — millisecond clock (default: CLINT or semihost)
  cnidoom_sleep_ms(ms)     — sleep (default: busy-wait on get_ticks_ms)
EOF

    echo "  Exported $(du -sh "$EXPORT_DIR" | cut -f1) SDK package"
    ls -la "$EXPORT_DIR"/
    echo ""
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Build complete!"
echo ""
echo "  Model:    $TFLITE"
echo "  Codegen:  $GENERATED_DIR/"

case "$TARGET" in
    x86)
        echo "  Binary:   $BUILD_DIR/doom_agent_static_host"
        echo ""
        echo "  Run:"
        echo "    ./$BUILD_DIR/doom_agent_static_host --model ignored -iwad wads/DOOM1.WAD"
        ;;
    rv32)
        echo "  Binary:   $BUILD_DIR/doom_agent_rv32.elf"
        if [[ -n "$EXPORT_DIR" ]]; then
            echo "  SDK:      $EXPORT_DIR/"
        fi
        echo ""
        echo "  Run (QEMU):"
        echo "    qemu-system-riscv32 -machine virt \\"
        echo "      -cpu rv32,v=true,vlen=128,zve32f=true -m 128M \\"
        echo "      -nographic -bios none \\"
        echo "      -semihosting-config enable=on,target=native \\"
        echo "      -kernel $BUILD_DIR/doom_agent_rv32.elf"
        ;;
esac
echo "================================================================"
