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

from typing import Any

import hydra.utils as hyu
import torch

from alpamayo_r1.action_space.action_space import ActionSpace


class DiscreteTrajectoryTokenizer:
    """Discrete trajectory tokenizer."""

    def __init__(
        self,
        action_space_cfg: dict[str, Any],
        dims_min: list[float],
        dims_max: list[float],
        num_bins: int,
        **kwargs: Any,
    ) -> None:
        """Initializes the tokenizer."""
        self.action_space: ActionSpace = hyu.instantiate(action_space_cfg)
        assert len(dims_min) == len(dims_max) == self.action_space.get_action_space_dims()[-1]
        self.dims_min = dims_min
        self.dims_max = dims_max
        self.num_bins = num_bins

    @property
    def vocab_size(self) -> int:
        """Tokens are integers from the set {0, 1, ..., vocab_size - 1}"""
        return self.num_bins

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Encodes the trajectories as discrete tokens.

        We assume the the future tstamp is consistent with the future trajectory.

        Args:
            hist_xyz: The history xyz coordinates.
            hist_rot: The history rotation matrices.
            fut_xyz: The future xyz coordinates.
            fut_rot: The future rotation matrices.
            hist_tstamp: The history timestamps.
            fut_tstamp: The future timestamps.

        Returns:
            tokens: The encoded tokens. Shape: (B, num_tokens_per_trajectory).
        """
        batch_size = fut_xyz.shape[0]
        action = self.action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
        dims_min = torch.tensor(self.dims_min, device=action.device, dtype=action.dtype)
        dims_max = torch.tensor(self.dims_max, device=action.device, dtype=action.dtype)
        action = (action - dims_min) / (dims_max - dims_min)
        action = (action * (self.num_bins - 1)).round().long()
        action = action.clamp(0, self.num_bins - 1)
        return action.reshape(batch_size, -1)

    def decode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        tokens: torch.LongTensor,
        hist_tstamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Decodes the given tokens into future trajectories.

        We assume the the future tstamp is consistent with the future trajectory.

        Args:
            hist_xyz: The history xyz coordinates.
            hist_rot: The history rotation matrices.
            tokens: The tokens to decode.
            hist_tstamp: The history timestamps.

        Returns:
            fut_xyz: The decoded future xyz coordinates.
            fut_rot: The decoded future rotation matrices.
            None: The future timestamps are not decoded.
        """
        action = tokens.reshape(-1, *self.action_space.get_action_space_dims()).to(hist_xyz.dtype)
        dims_min = torch.tensor(self.dims_min, device=action.device, dtype=action.dtype)
        dims_max = torch.tensor(self.dims_max, device=action.device, dtype=action.dtype)
        action = action / (self.num_bins - 1)
        action = action * (dims_max - dims_min) + dims_min
        fut_xyz, fut_rot = self.action_space.action_to_traj(action, hist_xyz, hist_rot)
        return fut_xyz, fut_rot, None
