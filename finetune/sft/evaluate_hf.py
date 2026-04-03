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
from collections import defaultdict
from itertools import islice

import hydra
import hydra.utils as hyu
import torch

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from finetune.sft.trainer import ReasoningVLA_Trainer
from finetune.sft.trainer import TrainingArguments
from finetune.sft.models.sft_base_model import TrainableReasoningVLA
from finetune.sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1

from alpamayo_r1.common import (
    logging,
    misc,
    wandb_utils,
    distributed,
)

from alpamayo_r1.common.logging import setup_logging

setup_logging()

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@hydra.main(version_base=None, config_path=None, config_name="config")
def evaluate(cfg: DictConfig) -> None:
    distributed.initialize_distributed_simple()

    logger.info(
        "Dataset Configs:\n"
        + misc.pformat(OmegaConf.to_container(cfg.data.val_dataset, resolve=True))
    )
    logger.info(
        "Evaluate Configs:\n" + misc.pformat(OmegaConf.to_container(cfg.evaluate, resolve=True))
    )
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))

    if cfg.evaluate.get("eval_ckpt", None) is not None:
        logger.info(f"Loading model from {cfg.evaluate.eval_ckpt}")
        model_cls = hyu.get_class(cfg.model._target_.rsplit(".", 1)[0])
        if issubclass(model_cls, TrainableReasoningVLA):
            cfg.model.checkpoint_path = cfg.evaluate.eval_ckpt
        elif issubclass(model_cls, TrainableAlpamayoR1):
            cfg.model.pretrained_model_name_or_path = cfg.evaluate.eval_ckpt
        else:
            raise ValueError(f"Unsupported model class: {model_cls}")
    model = hyu.instantiate(cfg.model, _convert_="partial")

    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )
    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model.config
    )

    trainer = ReasoningVLA_Trainer(
        model=model, args=training_args, eval_dataset=eval_dataset, data_collator=collate_fn
    )
    model = trainer.accelerator.prepare_model(model, evaluation_mode=True)
    model.eval()
    accelerator = trainer.accelerator
    is_main_process = accelerator.is_main_process

    if cfg.get("wandb", None) and is_main_process:
        os.makedirs(cfg.wandb.output_dir, exist_ok=True)
        wandb_utils.init_wandb(**cfg.wandb)

    val_dataloader = trainer.get_eval_dataloader()

    metric_runner = hydra.utils.instantiate(cfg.evaluate.metric_runner)

    max_eval_steps = cfg.evaluate.get("max_eval_steps", -1)
    if max_eval_steps == -1 or max_eval_steps is None:
        dataloader_iter = val_dataloader
        total = len(val_dataloader)
    else:
        dataloader_iter = islice(val_dataloader, max_eval_steps)
        total = max_eval_steps

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    val_count = 0

    for data in tqdm(dataloader_iter, total=total, disable=not is_main_process):
        output_batch = {}
        with torch.autocast("cuda", dtype=dtype_map[cfg.evaluate.torch_dtype]):
            metric_runner.run(model, data, output_batch)

        batch_size = len(data["image_frames"])
        gathered_batch_size = accelerator.gather_for_metrics(
            torch.tensor([batch_size], device=accelerator.device, dtype=torch.long)
        )
        if is_main_process:
            val_count += int(gathered_batch_size.sum().item())

        for k, v in output_batch.items():
            if not k.startswith("metric/"):
                continue
            gathered_metric = accelerator.gather_for_metrics(v)
            if is_main_process:
                metric_sums[k] += gathered_metric.float().sum().item()
                metric_counts[k] += gathered_metric.numel()

    if not is_main_process:
        return

    final_metrics_dict = {}
    for key in metric_sums.keys():
        if metric_counts[key] > 0:
            final_metrics_dict["val/" + key] = metric_sums[key] / metric_counts[key]

    padding = 15
    if len(final_metrics_dict) > 0:
        padding = max(padding, *(len(k) for k in final_metrics_dict.keys()))
    logger.info(
        f"Validation @ iteration\n"
        f"{'val/count':<{padding}} {val_count:.4f}\n"
        + "\n".join(
            f"{k:<{padding}} {final_metrics_dict[k]:.4f}" for k in sorted(final_metrics_dict.keys())
        )
        + "\n"
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    evaluate()
