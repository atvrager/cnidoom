#!/usr/bin/env bash
# Download the Doom shareware WAD (DOOM1.WAD) for training/testing.
# The shareware WAD is freely redistributable and contains E1M1-E1M9.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WAD_DIR="$REPO_ROOT/wads"
WAD_PATH="$WAD_DIR/DOOM1.WAD"

DOOM_SW_URL="https://distro.ibiblio.org/slitaz/sources/packages/d/doom1.wad"
SHA256="1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771"

if [[ -f "$WAD_PATH" ]]; then
    echo "DOOM1.WAD already exists at $WAD_PATH"
    exit 0
fi

mkdir -p "$WAD_DIR"

echo "Downloading DOOM1.WAD (shareware)..."
if command -v curl &>/dev/null; then
    curl -fSL -o "$WAD_PATH" "$DOOM_SW_URL"
elif command -v wget &>/dev/null; then
    wget -q -O "$WAD_PATH" "$DOOM_SW_URL"
else
    echo "ERROR: neither curl nor wget found" >&2
    exit 1
fi

# Verify checksum.
echo "Verifying checksum..."
actual=$(sha256sum "$WAD_PATH" | awk '{print $1}')
if [[ "$actual" != "$SHA256" ]]; then
    echo "ERROR: checksum mismatch" >&2
    echo "  expected: $SHA256" >&2
    echo "  actual:   $actual" >&2
    rm -f "$WAD_PATH"
    exit 1
fi

echo "Downloaded DOOM1.WAD to $WAD_PATH"
