"""Cosmos-RL entry point for ReasoningVLA.

Pre-import bootstrap (env vars, monkey patches, vLLM registration) must run
before any cosmos_rl imports, so it stays here. The actual launch logic is in
rl.launcher via REASONING_VLA_SPEC.launch().
"""

# ruff: noqa: E402

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current_dir))))
_src_dir = os.path.join(_repo_root, "src")
_projects_dir = os.path.join(_repo_root, "finetune")

if _src_dir not in sys.path or _projects_dir not in sys.path or _repo_root not in sys.path:
    sys.path[:0] = [_src_dir, _projects_dir]

os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")
os.environ.setdefault("COSMOS_LOG_LEVEL", "DEBUG")

# _PAI_LOCAL_DIR = os.getenv("ALPAMAYO_PAI_LOCAL_DIR")
# if not _PAI_LOCAL_DIR:
#     raise RuntimeError(
#         "Missing required env var ALPAMAYO_PAI_LOCAL_DIR "
#         "(expected PAI dataset root, e.g. /path/to/PAI_mini)."
#     )

_PAI_LOCAL_DIR = "/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/rant/PAI_mini"

# ---------------------------------------------------------------------------
# vLLM registration
# ---------------------------------------------------------------------------
from cosmos_rl.utils.logging import logger

try:
    from vllm import ModelRegistry as vllm_model_registry

    from rl.models.reasoning_vla.vllm_wrapper import ReasoningVLAModelForVLLM

    vllm_model_registry.register_model("ReasoningVLA", ReasoningVLAModelForVLLM)
except Exception as e:
    logger.warning(f"Failed to register ReasoningVLA model with vLLM: {e}")

# Side-effect imports: register trainer and rollout with Cosmos registries
# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------
from rl.models._spec import ModelSpec
from rl.models.reasoning_vla.cosmos_wrapper import RVLACosmos
from rl.models.reasoning_vla.data_packer import RVLADataPacker
from rl.models.reasoning_vla.rollout import ReasoningVlaVllmRollout  # noqa: F401
from rl.models.reasoning_vla.trainer import ReasoningVLAGRPOTrainer  # noqa: F401
from rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper


def _reasoning_vla_reward_fn(to_be_evaluated, reference=None, *args, config=None, **kwargs):
    import rl.state as alp_state
    from rl.rewards.aggregated_reward import compute_reward

    assert reference is not None, "reference is required for Alpamayo reward"
    return compute_reward(
        to_be_evaluated,
        reference,
        tokenizer=alp_state.get_tokenizer(),
        traj_tokenizer=alp_state.get_traj_tokenizer(),
        config=config,
        model_config=alp_state.get_ckpt_cfg(),
    )


REASONING_VLA_SPEC = ModelSpec(
    cosmos_wrapper=RVLACosmos,
    weight_mapper=ReasoningVLAWeightMapper,
    data_packer_cls=RVLADataPacker,
    reward_fn=_reasoning_vla_reward_fn,
    hydra_config_path="hydra_configs",
    hydra_config_name="alpamayo1_5_rvla_rl_pai",
    hydra_overrides=[
        f"data.train.dataset.local_dir={_PAI_LOCAL_DIR}",
        "data.train.dataset.clip_index_metadata=clip_index_mini2.parquet",
        "data.train.dataset.features_metadata=features.csv",
        "data.train.dataset.use_default_keyframe=True",
    ],
)

if __name__ == "__main__":
    REASONING_VLA_SPEC.launch()
