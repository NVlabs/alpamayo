#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY:?Please export WANDB_API_KEY first}"

torchrun --standalone --nproc_per_node=1 \
  -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage1_overfit_scene_wandb \
  data.train_dataset.local_dir="${ALPAMAYO_PAI_LOCAL_DIR}" \
  data.val_dataset.local_dir="${ALPAMAYO_PAI_LOCAL_DIR}" \
  model.checkpoint_path="${ALPAMAYO_MODEL_DIR}" \
  paths.output_dir="${OVERFIT_SCENE_STAGE1_OUT:-$WORKSPACE_DIR/artifacts/overfit_scene_stage1_wandb}" \
  wandb.key="${WANDB_API_KEY}" \
  wandb.team="${WANDB_TEAM:-moonzerokevin-university-of-british-columbia}" \
  wandb.project="${WANDB_PROJECT:-alpamayo-stage1-overfit-scene}"
