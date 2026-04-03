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

"""ModelSpec dataclass -- bundles all Cosmos-RL components for an Alpamayo model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelSpec:
    """Bundles all Cosmos-RL components for an Alpamayo RL post-training model.

    Usage::

        EXPERT_MODEL_SPEC.launch()               # reads ckpt from COSMOS_CONFIG TOML
        EXPERT_MODEL_SPEC.launch(ckpt_path=...)   # explicit override
    """

    cosmos_wrapper: type
    weight_mapper: type
    data_packer_cls: type
    reward_fn: Callable
    hydra_config_path: str
    hydra_config_name: str
    hydra_overrides: list[str]

    def launch(self, ckpt_path: str | None = None) -> None:
        from rl.launcher import launch_alpamayo_model

        launch_alpamayo_model(self, ckpt_path=ckpt_path)
