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

"""Shared Cosmos-RL launch logic for all Alpamayo models."""

from __future__ import annotations


def _read_ckpt_path_from_toml() -> str:
    """Read policy.model_name_or_path from the Cosmos TOML config.

    Expects --config <path> in sys.argv (passed by Cosmos launch_replica.sh).
    """
    import sys
    import tomllib

    toml_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            toml_path = sys.argv[i + 1]
            break

    if not toml_path:
        raise RuntimeError(
            "No --config <path> found in sys.argv. "
            "The Cosmos orchestrator should pass --config automatically."
        )
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["policy"]["model_name_or_path"]


def launch_alpamayo_model(spec, ckpt_path: str | None = None) -> None:
    """Register *spec* with Cosmos ModelRegistry and call launch_worker().

    Args:
        spec: A ``ModelSpec`` instance describing the model components.
        ckpt_path: Checkpoint path for data/tokenizer init.  If ``None``,
            reads ``[policy].model_name_or_path`` from the TOML config
            pointed to by the ``COSMOS_CONFIG`` env var.
    """
    from cosmos_rl.launcher.worker_entry import main as launch_worker
    from cosmos_rl.policy.model.base import ModelRegistry

    import rl.state as alp_state
    from rl.base_dataset import AlpamayoCosmosDataset

    if ckpt_path is None:
        ckpt_path = _read_ckpt_path_from_toml()

    alp_state.init_once(
        ckpt_path,
        hydra_config_path=spec.hydra_config_path,
        hydra_config_name=spec.hydra_config_name,
        overrides=spec.hydra_overrides,
    )

    ModelRegistry.register_model(
        spec.cosmos_wrapper,
        spec.weight_mapper,
        data_packer_cls=spec.data_packer_cls,
    )

    launch_worker(
        dataset=lambda config: AlpamayoCosmosDataset(split="train"),
        data_packer=spec.data_packer_cls(),
        reward_fns=[spec.reward_fn],
        val_dataset=lambda config: AlpamayoCosmosDataset(split="val"),
        val_data_packer=spec.data_packer_cls(),
        val_reward_fns=[spec.reward_fn],
    )
