#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

if [[ -z "${STAGE2_CKPT_DIR:-}" ]]; then
  echo "请先设置 STAGE2_CKPT_DIR 指向 Stage 2 输出下的 checkpoint 目录。"
  exit 1
fi

torchrun --standalone --nproc_per_node=1 \
  -m finetune.sft.evaluate_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage2_single_h100 \
  data.val_dataset.local_dir="$ALPAMAYO_PAI_LOCAL_DIR" \
  data.val_dataset.chunk_ids="${VAL_CHUNKS:-3116-3117}" \
  evaluate.eval_ckpt="$STAGE2_CKPT_DIR" \
  paths.output_dir="$ALPAMAYO_STAGE2_OUT"
