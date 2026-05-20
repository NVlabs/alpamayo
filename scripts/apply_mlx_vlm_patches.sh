#!/bin/bash
# Apply mlx_vlm patches required for Alpamayo-R1 MLX port parity.
# This script locates the installed mlx_vlm package and applies the patch(es)
# in the patches/ directory.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/apply_mlx_vlm_patches.sh
#
# Safe to re-run; patch will skip already-applied hunks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$PROJECT_ROOT/patches"

# Find the installed mlx_vlm location
VLM_PATH=$(python -c "
import mlx_vlm
import os
print(os.path.dirname(mlx_vlm.__file__))
" 2>/dev/null || echo "")

if [[ -z "$VLM_PATH" ]]; then
    echo "ERROR: mlx_vlm is not importable. Activate the virtual environment first."
    exit 1
fi

echo "mlx_vlm found at: $VLM_PATH"

PATCH_FILE="$PATCH_DIR/mlx_vlm_generate.patch"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

echo "Applying patch: $PATCH_FILE"
cd "$VLM_PATH"

# Use -N to ignore already-applied hunks, -p1 because the patch is relative to the package root
patch -N -p1 < "$PATCH_FILE" || true

echo "Patch applied (or already present)."
echo "You may need to restart any running Python processes for the change to take effect."