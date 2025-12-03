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

from typing import Literal

import torch
from alpamayo_r1.diffusion.base import BaseDiffusion, StepFn


class FlowMatching(BaseDiffusion):
    """Flow Matching model.

    References:
    Flow Matching for Generative Modeling
        https://arxiv.org/pdf/2210.02747
    Guided Flows for Generative Modeling and Decision Making
        https://arxiv.org/pdf/2311.13443
    """

    def __init__(
        self,
        int_method: Literal["euler"] = "euler",
        num_inference_steps: int = 10,
        *args,
        **kwargs,
    ):
        """Initialize the FlowMatching model.

        Args:
            int_method: The integration method used in inference.
            num_inference_steps: The number of inference steps.
        """
        super().__init__(*args, **kwargs)
        self.int_method = int_method
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        int_method: Literal["euler"] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample data from the model.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function.
            device: The device to use.
            return_all_steps: Whether to return all steps.
            inference_step: The number of inference steps. (override self.num_inference_steps)
            int_method: The integration method used in inference. (override self.int_method)

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        int_method = int_method or self.int_method
        inference_step = inference_step or self.num_inference_steps
        if int_method == "euler":
            return self._euler(
                batch_size=batch_size,
                step_fn=step_fn,
                device=device,
                return_all_steps=return_all_steps,
                inference_step=inference_step,
            )
        else:
            raise ValueError(f"Invalid integration method: {int_method}")

    def _euler(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Euler integration for flow matching.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function.
            device: The device to use.
            return_all_steps: Whether to return all steps.
            inference_step: The inference step.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        x = torch.randn(batch_size, *self.x_dims, device=device)
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.x_dims)
        if return_all_steps:
            all_steps = [x]

        for i in range(inference_step):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            v = step_fn(x=x, t=t_start)
            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)
        if return_all_steps:
            return torch.stack(all_steps, dim=1), time_steps
        return x
