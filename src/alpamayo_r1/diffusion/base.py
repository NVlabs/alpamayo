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

import logging
from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import nn

logger = logging.getLogger(__name__)


class StepFn(Protocol):
    def __call__(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Denoising step function.

        Args:
            x: The input tensor.
            t: The timestep.

        Returns:
            torch.Tensor: The denoised tensor.
        """
        ...


class BaseDiffusion(ABC, nn.Module):
    """Base class for diffusion models."""

    def __init__(
        self,
        x_dims: list[int] | tuple[int] | int,
        *args,
        **kwargs,
    ):
        """Initialize the BaseDiffusion model.

        Args:
            x_dims: The dimension of the input tensor.
        """
        super().__init__()
        self.x_dims = [x_dims] if isinstance(x_dims, int) else list(x_dims)

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample from the diffusion model.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function that takes a noisy x and a
                timestep t and returns either a denoised x, a vector field or noise depending on
                the prediction type of the diffusion model.
            device: The device to use.
            return_all_steps: Whether to return the outputs from all steps.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        raise NotImplementedError
