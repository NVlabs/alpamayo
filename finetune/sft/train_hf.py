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

import os
import hydra
import hydra.utils as hyu
import torch
from omegaconf import DictConfig, OmegaConf

from alpamayo_r1.common import logging
from alpamayo_r1.common import misc

from finetune.sft.trainer import ReasoningVLA_Trainer
from finetune.sft.trainer import TrainingArguments

from alpamayo_r1.common import config_utils
from alpamayo_r1.common import wandb_utils
from alpamayo_r1.common.logging import setup_logging

setup_logging()

logger = logging.RankedLogger("train", rank_zero_only=True)
logger.setLevel("INFO")


@hydra.main(version_base=None, config_path=None, config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training entry point."""
    misc.seed_everything(42)

    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))
    logger.info("Configs:\n" + misc.pformat(OmegaConf.to_container(cfg, resolve=True)))

    model = hyu.instantiate(cfg.model, _convert_="partial")

    train_dataset = hyu.instantiate(
        cfg.data.train_dataset, _convert_="partial", model_config=model.config
    )
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )

    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model.config
    )

    callbacks = []
    for cb_name, cb_cfg in cfg.callbacks.items():
        logger.info(f"Initializing callback {cb_name}")
        callbacks.append(hyu.instantiate(cb_cfg, _convert_="partial"))

    trainer = ReasoningVLA_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    if "deepspeed" in cfg.trainer and cfg.trainer.deepspeed is not None:
        # We should not cast the forward inputs to bfloat16 because our model is mixed
        # precision and trajectory encoder might require float32 input.
        ds_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
        ds_config._dtype = torch.float32

    if cfg.get("wandb", None) is not None:
        wandb_utils.init_wandb(**cfg.wandb)

    if trainer.is_world_process_zero():
        config_utils.save_config(
            cfg,
            os.path.join(cfg.paths.output_dir, "config.yaml"),
            resolve_paths=True,
            include_hydra_config=True,
        )

    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
