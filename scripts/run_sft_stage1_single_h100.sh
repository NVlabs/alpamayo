#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

torchrun --standalone --nproc_per_node=1 \
  -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage1_single_h100 \
  data.train_dataset.local_dir="$ALPAMAYO_PAI_LOCAL_DIR" \
  data.val_dataset.local_dir="$ALPAMAYO_PAI_LOCAL_DIR" \
  data.train_dataset.chunk_ids="${TRAIN_CHUNKS:-3116-3117}" \
  data.val_dataset.chunk_ids="${VAL_CHUNKS:-3116-3117}" \
  model.checkpoint_path="$ALPAMAYO_MODEL_DIR" \
  paths.output_dir="$ALPAMAYO_STAGE1_OUT"
