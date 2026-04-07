#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY:?Please export WANDB_API_KEY first}"

if [[ -z "${STAGE1_CKPT_DIR:-}" ]]; then
  echo "请先设置 STAGE1_CKPT_DIR 指向 overfit Stage 1 的 checkpoint 目录。"
  exit 1
fi

torchrun --standalone --nproc_per_node=1 \
  -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage2_overfit_one_sample_wandb \
  data.train_dataset.local_dir="$ALPAMAYO_PAI_LOCAL_DIR" \
  data.val_dataset.local_dir="$ALPAMAYO_PAI_LOCAL_DIR" \
  data.train_dataset.chunk_ids="${TRAIN_CHUNKS:-3116}" \
  data.val_dataset.chunk_ids="${VAL_CHUNKS:-3116}" \
  model.pretrained_model_name_or_path="$ALPAMAYO_MODEL_DIR" \
  model.stage1_vlm_checkpoint_path="$STAGE1_CKPT_DIR" \
  paths.output_dir="${OVERFIT_STAGE2_OUT:-$WORKSPACE_DIR/artifacts/overfit_stage2_wandb}" \
  wandb.key="$WANDB_API_KEY" \
  wandb.team="${WANDB_TEAM:-moonzerokevin-university-of-british-columbia}" \
  wandb.project="${WANDB_PROJECT:-alpamayo-stage2-overfit}"
