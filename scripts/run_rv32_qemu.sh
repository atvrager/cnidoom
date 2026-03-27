#!/bin/bash
# run_rv32_qemu.sh — Launch the RISC-V Doom agent in QEMU with SDL display.
#
# Usage:
#   scripts/run_rv32_qemu.sh [path/to/doom_agent_rv32.elf]
#
# Prerequisites:
#   pacman -S qemu-system-riscv qemu-system-riscv-firmware riscv32-elf-binutils
#
# The script runs QEMU in system mode with:
#   - ramfb display (renders in QEMU's SDL window)
#   - Semihosting enabled (guest can open host files for WAD loading)
#   - 128 MB RAM (matches linker script)
#   - RV32IMF + vector extensions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ELF="${1:-${PROJECT_DIR}/build-rv32/doom_agent_rv32.elf}"

if [ ! -f "$ELF" ]; then
    echo "Error: ELF not found at $ELF"
    echo "Build it first:"
    echo "  cmake -B build-rv32 -S inference \\"
    echo "    -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv32-elf-clang.cmake \\"
    echo "    -DDOOM_AGENT_STATIC=ON -DDOOM_AGENT_RV32=ON"
    echo "  cmake --build build-rv32"
    exit 1
fi

if ! command -v qemu-system-riscv32 &>/dev/null; then
    echo "Error: qemu-system-riscv32 not found."
    echo "Install: pacman -S qemu-system-riscv"
    exit 1
fi

echo "Starting RISC-V Doom agent in QEMU..."
echo "  ELF: $ELF"
echo "  Display: ramfb (SDL)"
echo ""
echo "Press Ctrl-A X to exit QEMU."

cd "$PROJECT_DIR"

MODE="${2:-display}"

if [ "$MODE" = "headless" ]; then
    # Headless mode — UART output only, no display.
    exec qemu-system-riscv32 \
        -machine virt \
        -cpu rv32,v=true,vlen=128,zve32f=true \
        -m 128M \
        -nographic \
        -bios none \
        -semihosting-config enable=on,target=native \
        -kernel "$ELF"
else
    # Display mode — ramfb renders in QEMU's SDL window.
    exec qemu-system-riscv32 \
        -machine virt \
        -cpu rv32,v=true,vlen=128,zve32f=true \
        -m 128M \
        -bios none \
        -device ramfb \
        -display sdl \
        -semihosting-config enable=on,target=native \
        -serial mon:stdio \
        -kernel "$ELF"
fi
