# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ReasoningVLA RL post-training entry point (Cosmos-RL / GRPO).

Sets up env vars and registers the ReasoningVLA model with vLLM before
assembling a ``ModelSpec`` with all Cosmos-RL components — model wrapper,
weight mapper, data packer, and reward function — and launching GRPO
training on the PAI dataset.
"""

# ruff: noqa: E402

import os


os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")
os.environ.setdefault("COSMOS_LOG_LEVEL", "DEBUG")

_PAI_LOCAL_DIR = os.getenv("ALPAMAYO_PAI_LOCAL_DIR")
if not _PAI_LOCAL_DIR:
    raise RuntimeError(
        "Missing required env var ALPAMAYO_PAI_LOCAL_DIR "
        "(expected PAI dataset root, e.g. /path/to/PAI_mini)."
    )

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

# ---------------------------------------------------------------------------
# Model spec components
# ---------------------------------------------------------------------------
from rl.models._spec import ModelSpec
from rl.models.reasoning_vla.cosmos_wrapper import RVLACosmos
from rl.models.reasoning_vla.data_packer import RVLADataPacker
from rl.models.reasoning_vla.rollout import ReasoningVlaVllmRollout  # noqa: F401 (Cosmos registry)
from rl.models.reasoning_vla.trainer import ReasoningVLAGRPOTrainer  # noqa: F401 (Cosmos registry)
from rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper


def _reasoning_vla_reward_fn(to_be_evaluated, reference=None, *args, config=None, **kwargs):
    """Compute aggregated reward for a single ReasoningVLA rollout."""
    import rl.state as alp_state
    from rl.rewards.aggregated_reward import compute_reward

    assert isinstance(reference, dict) and reference, (
        f"Expected a non-empty dict for reference, got {type(reference).__name__}: {reference!r}"
    )
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
        "data.train.dataset.clip_index_metadata=clip_index_mini.parquet",
        "data.train.dataset.features_metadata=features.csv",
        "data.train.dataset.use_default_keyframe=True",
    ],
)

if __name__ == "__main__":
    REASONING_VLA_SPEC.launch()
