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

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ActionSpace(ABC, nn.Module):
    """Action space base class for the trajectory generation."""

    @abstractmethod
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Transform the future trajectory to the action space.

        Args:
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
            *args: other data for the action space
            **kwargs: other data for the action space

        Returns:
            action: (..., *action_space_dims)
        """

    @abstractmethod
    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the action space to the trajectory.

        Args:
            action: (..., *action_space_dims)
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            *args: other data for the action space
            **kwargs: other data for the action space

        Returns:
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
        """

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]:
        """Get the dimensions of the action space.

        Returns:
            action_space_dims: the action space dimensions
        """

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Check if the action is within the bounds.

        By default, we assume the action is within bounds (dummy implementation).

        Args:
            action: (..., *action_space_dims)

        Returns:
            is_within_bounds: (...,)
        """
        num_action_dims = len(self.get_action_space_dims())
        batch_shape = action.shape[:-num_action_dims] if num_action_dims > 0 else action.shape
        return torch.ones(batch_shape, dtype=torch.bool, device=action.device)
