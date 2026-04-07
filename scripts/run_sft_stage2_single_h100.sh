#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

if [[ -z "${STAGE1_CKPT_DIR:-}" ]]; then
  echo "请先设置 STAGE1_CKPT_DIR 指向 Stage 1 输出下的 checkpoint 目录。"
  exit 1
fi

torchrun --standalone --nproc_per_node=1 \
  -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage2_single_h100 \
  data.train_dataset.local_dir="$ALPAMAYO_PAI_LOCAL_DIR" \
  data.val_dataset.local_dir="$ALPAMAYO_PAI_LOCAL_DIR" \
  data.train_dataset.chunk_ids="${TRAIN_CHUNKS:-3116-3117}" \
  data.val_dataset.chunk_ids="${VAL_CHUNKS:-3116-3117}" \
  model.pretrained_model_name_or_path="$ALPAMAYO_MODEL_DIR" \
  model.stage1_vlm_checkpoint_path="$STAGE1_CKPT_DIR" \
  paths.output_dir="$ALPAMAYO_STAGE2_OUT"
