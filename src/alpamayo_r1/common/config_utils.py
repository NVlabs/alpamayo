# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


def save_config(
    config: DictConfig,
    output_path: str,
    include_hydra_config: bool = True,
    resolve_paths: bool = True,
) -> None:
    """Save the configuration to a file."""
    config = copy.copy(config)
    logger.info(f"Config saved to {output_path}")
    with open_dict(config):
        if resolve_paths:
            config.paths = OmegaConf.to_container(config.paths, resolve=True)
        if include_hydra_config:
            hydra_config = HydraConfig.get()
            config.hydra = {}
            for key in hydra_config:
                if key == "runtime":
                    continue
                config.hydra[key] = hydra_config[key]
    with open(output_path, "w") as f:
        OmegaConf.save(config, f)
