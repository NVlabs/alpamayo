#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

if [[ -z "${COSMOS_POLICY_CKPT_DIR:-}" ]]; then
  echo "请先设置 COSMOS_POLICY_CKPT_DIR 指向 Cosmos-RL 导出的 policy checkpoint 目录。"
  exit 1
fi

python scripts/convert_cosmos_rl_checkpoint.py \
  --cosmos-policy-ckpt "$COSMOS_POLICY_CKPT_DIR" \
  --base-hf-ckpt "$ALPAMAYO_MODEL_DIR" \
  --output-dir "$ALPAMAYO_RL_EXPORT_DIR"
