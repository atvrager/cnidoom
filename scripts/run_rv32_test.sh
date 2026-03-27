#!/bin/bash
# run_rv32_test.sh — Build and run bare-metal RISC-V kernel tests under QEMU.
#
# Usage:
#   scripts/run_rv32_test.sh [test_name] [--build-only] [--run-only]
#
# Available tests:
#   rvv_bitexact   RVV vs generic C bit-accuracy (default)
#
# Examples:
#   scripts/run_rv32_test.sh                     # build + run rvv_bitexact
#   scripts/run_rv32_test.sh rvv_bitexact        # same
#   scripts/run_rv32_test.sh --build-only        # build only
#   scripts/run_rv32_test.sh --run-only          # run only (assumes built)
#
# Prerequisites:
#   - RISC-V toolchain (auto-fetched by cmake)
#   - qemu-system-riscv32 (pacman -S qemu-system-riscv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build-rv32"
TIMEOUT=120

# Parse args.
TEST_NAME="rvv_bitexact"
DO_BUILD=true
DO_RUN=true

for arg in "$@"; do
    case "$arg" in
        --build-only) DO_RUN=false ;;
        --run-only)   DO_BUILD=false ;;
        -*)           echo "Unknown flag: $arg"; exit 1 ;;
        *)            TEST_NAME="$arg" ;;
    esac
done

# Map test name to target and ELF.
case "$TEST_NAME" in
    rvv_bitexact)
        TARGET="test_rvv_bitexact"
        ELF="${BUILD_DIR}/test_rvv_bitexact.elf"
        # The test prints "=== SUMMARY:" as its final line.
        SENTINEL="=== SUMMARY:"
        ;;
    *)
        echo "Unknown test: $TEST_NAME"
        echo "Available: rvv_bitexact"
        exit 1
        ;;
esac

# ── Build ────────────────────────────────────────────────────────────
if $DO_BUILD; then
    echo "=== Configuring rv32 build ==="
    cmake -B "$BUILD_DIR" -S "${PROJECT_DIR}/inference" \
        -DCMAKE_TOOLCHAIN_FILE="${PROJECT_DIR}/cmake/riscv32-elf-clang.cmake" \
        -DDOOM_AGENT_STATIC=ON \
        -DDOOM_AGENT_RV32=ON \
        -DDOOM_AGENT_KERNEL_TARGET=riscv \
        -DDOOM_AGENT_BUILD_TESTS=ON

    echo ""
    echo "=== Building $TARGET ==="
    cmake --build "$BUILD_DIR" --target "$TARGET"
    echo ""
fi

# ── Run ──────────────────────────────────────────────────────────────
if $DO_RUN; then
    if [ ! -f "$ELF" ]; then
        echo "Error: ELF not found at $ELF"
        echo "Run without --run-only to build first."
        exit 1
    fi

    if ! command -v qemu-system-riscv32 &>/dev/null; then
        echo "Error: qemu-system-riscv32 not found."
        echo "Install: pacman -S qemu-system-riscv"
        exit 1
    fi

    echo "=== Running $TEST_NAME under QEMU ==="
    echo "  ELF: $ELF"
    echo ""

    # QEMU's rv32 semihosting _exit() doesn't reliably terminate the
    # process, so we stream output through a line reader that detects
    # the summary line and kills QEMU once the test is done.
    OUTFILE=$(mktemp)
    trap 'rm -f "$OUTFILE"' EXIT

    qemu-system-riscv32 \
        -machine virt \
        -cpu rv32,v=true,vlen=128,zve32f=true \
        -m 128M \
        -nographic \
        -bios none \
        -semihosting-config enable=on,target=native \
        -kernel "$ELF" \
        > "$OUTFILE" 2>&1 &
    QEMU_PID=$!

    # Wait for the sentinel line or timeout.
    ELAPSED=0
    FOUND=false
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if grep -q "$SENTINEL" "$OUTFILE" 2>/dev/null; then
            FOUND=true
            break
        fi
        # Check if QEMU exited on its own.
        if ! kill -0 "$QEMU_PID" 2>/dev/null; then
            FOUND=true
            break
        fi
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done

    # Kill QEMU (it won't exit on its own).
    kill "$QEMU_PID" 2>/dev/null || true
    wait "$QEMU_PID" 2>/dev/null || true

    # Print captured output.
    cat "$OUTFILE"
    echo ""

    if ! $FOUND; then
        echo "TIMEOUT: test did not complete within ${TIMEOUT}s"
        exit 1
    fi

    # Check result from output.
    if grep -q "ALL BIT-EXACT" "$OUTFILE"; then
        echo "=== $TEST_NAME: PASSED ==="
        exit 0
    elif grep -q "FAILED" "$OUTFILE"; then
        echo "=== $TEST_NAME: FAILED ==="
        exit 1
    elif grep -q "TRAP" "$OUTFILE"; then
        echo "=== $TEST_NAME: TRAP (illegal instruction or fault) ==="
        exit 1
    else
        echo "=== $TEST_NAME: UNKNOWN RESULT ==="
        exit 1
    fi
fi
