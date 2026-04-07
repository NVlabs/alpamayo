#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

python scripts/convert_release_config_to_training.py \
  --alpamayo-model nvidia/Alpamayo-R1-10B \
  --output-dir "$ALPAMAYO_MODEL_DIR"

python scripts/download_pai.py \
  --chunk-ids "${PAI_CHUNKS:-3116}" \
  --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --output-dir "$ALPAMAYO_PAI_LOCAL_DIR"

python scripts/curate_pai_samples.py \
  --clip-index-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet" \
  --chunk "${PAI_CHUNKS:-3116}" \
  --num-samples "${PAI_MINI_SAMPLES:-16}" \
  --output-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index_mini.parquet"
