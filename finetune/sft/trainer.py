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

from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled
from transformers import TrainingArguments as HFTrainingArguments

"""Custom TrainingArguments for Alpamayo training."""


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended TrainingArguments with support for custom learning rate multipliers.

    Attributes:
        lr_multiplier: Optional dict mapping parameter name prefixes to learning rate
            multipliers. If None, all parameters use the same learning rate.
            e.g.
                {
                    "vlm.visual": 0.1,  # Vision encoder uses 10% of base LR
                    "vlm.language_model": 1.0,  # Language model uses 100% of base LR
                }
            Parameters are matched using longest prefix match. Unmatched parameters
            use 1.0x multiplier by default.
    """

    lr_multiplier: Optional[dict[str, float]] = field(
        default=None,
        metadata={
            "help": (
                "Dict mapping parameter name prefixes to learning rate multipliers. "
                "Format: {prefix: multiplier}. Example: {'vlm.visual': 0.1} sets "
                "vision encoder parameters to use 0.1x the base learning rate."
            )
        },
    )


class ReasoningVLA_Trainer(Trainer):
    """Trainer for Reasoning VLA models with support for custom learning rate multipliers."""

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with custom learning rate multipliers for different parameter groups.

        If `lr_multiplier` is set in training args, it should be a dict mapping parameter name
        prefixes to learning rate multipliers. For example:
            lr_multiplier: {
                "vlm.visual": 0.1,  # Vision encoder uses 0.1x learning rate
                "vlm.language_model": 1.0,  # Language model uses 1.0x learning rate
            }

        If `lr_multiplier` is None, falls back to the default Trainer behavior.
        """
        # Get lr_multiplier from training args
        lr_multiplier = getattr(self.args, "lr_multiplier", None)

        # If no lr_multiplier specified, use default behavior
        if lr_multiplier is None:
            return super().create_optimizer()

        # Custom optimizer with different learning rates for different parameter groups
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:  # type: ignore[has-type]
            # Group parameters by their prefix match
            param_groups = {}  # type: ignore[var-annotated]  # prefix -> list of (name, param)

            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue

                # Find matching prefix (use longest match)
                matched_prefix = None
                for prefix in lr_multiplier.keys():
                    if name.startswith(prefix):
                        if matched_prefix is None or len(prefix) > len(matched_prefix):
                            matched_prefix = prefix

                # Use matched prefix or "default" for unmatched params
                group_key = matched_prefix if matched_prefix else "default"
                if group_key not in param_groups:
                    param_groups[group_key] = []
                param_groups[group_key].append((name, param))

            # Get decay parameters
            decay_parameters = self.get_decay_parameter_names(opt_model)

            # Build optimizer grouped parameters
            optimizer_grouped_parameters = []

            for group_key, params in param_groups.items():
                # Get learning rate multiplier for this group
                if group_key == "default":
                    lr_mult = 1.0
                else:
                    lr_mult = lr_multiplier[group_key]

                lr = self.args.learning_rate * lr_mult

                # Split into decay and no-decay groups
                decay_params = [p for n, p in params if n in decay_parameters]
                no_decay_params = [p for n, p in params if n not in decay_parameters]

                if decay_params:
                    optimizer_grouped_parameters.append(
                        {
                            "params": decay_params,
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,
                        }
                    )

                if no_decay_params:
                    optimizer_grouped_parameters.append(
                        {
                            "params": no_decay_params,
                            "weight_decay": 0.0,
                            "lr": lr,
                        }
                    )

            # Get optimizer class and kwargs
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )

            # Create optimizer
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # Handle SageMaker MP
            if is_sagemaker_mp_enabled():
                import smdistributed.modelparallel.torch as smp

                self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
