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

import os

import wandb
import yaml

from alpamayo_r1.common import distributed, logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


@distributed.rank_zero_only
def init_wandb(
    key: str,
    team: str,
    project: str,
    group: str,
    name: str,
    output_dir: str,
    **kwargs,
) -> None:
    """Initialize wandb."""
    if key is not None:
        # login wandb, otherwise it will load from the home directory
        wandb.login(key=key)
    # Get an ID for the run
    wandb_id = _check_wandb_id(output_dir)
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()
        _save_wandb_id(output_dir, wandb_id)
        logger.info(f"Starting a new wandb run with ID {wandb_id}")
    else:
        logger.info(f"Resuming wandb run with ID {wandb_id}")

    if os.path.exists(os.path.join(output_dir, "config.yaml")):
        # Assuming the config.yaml is in the output_dir
        with open(os.path.join(output_dir, "config.yaml")) as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    wandb.init(
        force=True,
        id=wandb_id,
        entity=team,
        project=project,
        group=group,
        name=name,
        config=config,
        dir=output_dir,
        resume="allow",
        **kwargs,
    )


def _save_wandb_id(output_dir: str, wandb_id: str):
    """Save the wandb id to the output_dir."""
    wandb_id_path = os.path.join(output_dir, ".wandb_id")
    with open(wandb_id_path, "w") as f:
        f.write(wandb_id)


def _check_wandb_id(output_dir: str) -> str | None:
    """Check if the wandb id exists in the output_dir."""
    wandb_id = None
    wandb_id_path = os.path.join(output_dir, ".wandb_id")
    if os.path.exists(wandb_id_path):
        with open(wandb_id_path) as f:
            wandb_id = f.read().strip()
    return wandb_id
