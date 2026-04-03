#!/usr/bin/env bash
set -exu
# E2E RL open loop pipeline for signal node local test


export YOUR_HOME="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/"


export ALPAMAYO_WORKSPACE="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/alpamayo_r1_release"
export ALPAMAYO_MODEL_DIR="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/alpamayo_model_converted_from_hf_331"
export ALPAMAYO_PAI_LOCAL_DIR="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/PAI_mini"
export ALPAMAYO_LOG_DIR="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/alpamayo_cosmos_rl_job/logs"

export UV_CACHE_DIR="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/.cache/uv"


export HF_HOME="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/.cache/huggingface"

export HF_HUB_OFFLINE=1

cd alpamayo/projects/alpamayo_r1_release

# # #Step 1: convert Alpamayo-R1-10B to ReasoningVLA checkpoint
python scripts/convert_release_config_to_training.py  --output-dir /home/bizhao/workspace/alpamayo/hf_model/alpamayo_1.5_converted-from-hf



python scripts/convert_release_config_to_training.py \
  --output-dir /lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/alpamayo_1.5_converted-from-hf-327


# # #Step 2: download PAI dataset
python scripts/download_pai.py --chunk-ids 3116 \
         --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
         --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions --labels egomotion \
         --output-dir /home/bizhao/workspace/alpamayo/dataset/alpamayo/PAI_mini

# # #Step 3: create mini sample index parquet file
python scripts/curate_pai_samples.py \
  --clip-index-path /home/bizhao/workspace/alpamayo/dataset/alpamayo/PAI_mini/clip_index.parquet \
  --chunk 3116 --num-samples 2 \
  --output-path /home/bizhao/workspace/alpamayo/dataset/alpamayo/PAI_mini/clip_index_3116_mini_2.parquet

# #Step 4: Launch RL open loop training
# Note: you must change the model_name_or_path under [policy] section to the output-dir of Step 1 in the toml file
# Note: you must change the data.train.dataset.local_dir in alpamayo_cosmos_rl_post_training_entry_local_test.py
export HYDRA_FULL_ERROR=1
export COSMOS_LOG_LEVEL=INFO
cosmos-rl --config finetune/rl/toml/alpamayo_r1_no_expert_10b_local_test.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir finetune/rl-logs \
  finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry_local_test.py
