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

import einops
import numpy as np
import torch


class DeltaTrajectoryTokenizer:
    """Delta trajectory tokenizers."""

    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        ego_yaw_min: float = -np.pi,
        ego_yaw_max: float = np.pi,
        num_bins: int = 1000,
        predict_yaw: bool = False,
        load_weights: bool = False,
    ):
        """Initializes the tokenizer."""
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.num_bins = num_bins
        self._predict_yaw = predict_yaw
        self.ego_yaw_min = ego_yaw_min
        self.ego_yaw_max = ego_yaw_max

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
        """Encodes the trajectories as discrete tokens. The model conditions on the historical
        waypoints to tokenize the future waypoints. Trajectories can be provided in any coordinate
        frame. Timestamps can be provided with any time-origin.

        Args:
            hist_xyz (torch.Tensor): Historical locations XYZ. Shape: (B, Th, 3).
            hist_rot (torch.Tensor): Historical rotations. Shape: (B, Th, 3, 3).
            fut_xyz (torch.Tensor): Future locations XYZ. Shape: (B, Tf, 3).
            fut_rot (torch.Tensor): Future rotations. Shape: (B, Tf, 3, 3).
            hist_tstamp (torch.Tensor): Historical time stamps. Shape: (B, Th).
            fut_tstamp (torch.Tensor): Future time stamps. Shape: (B, Tf).

        Returns:
            torch.LongTensor: The token indices. Shape: (B, num_tokens_per_trajectory).
        """
        del hist_xyz, hist_rot, hist_tstamp, fut_tstamp
        xyz = torch.nn.functional.pad(fut_xyz, [0, 0, 1, 0, 0, 0])
        xyz = xyz[:, 1:] - xyz[:, :-1]
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz - ego_xyz_min) / (ego_xyz_max - ego_xyz_min)
        xyz = (xyz * (self.num_bins - 1)).round().long()
        xyz = xyz.clamp(0, self.num_bins - 1)
        if not self._predict_yaw:
            return einops.rearrange(xyz, "b n m -> b (n m)")
        # Extract yaw angles from rotation matrices
        yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])

        # Calculate delta yaw
        yaw_padded = torch.nn.functional.pad(yaw, [1, 0, 0, 0])
        delta_yaw = yaw_padded[:, 1:] - yaw_padded[:, :-1]

        # Normalize delta yaw to [-pi, pi]
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

        # Scale and quantize delta yaw
        delta_yaw = (delta_yaw - self.ego_yaw_min) / (self.ego_yaw_max - self.ego_yaw_min)
        delta_yaw = (delta_yaw * (self.num_bins - 1)).round().long()
        delta_yaw = delta_yaw.clamp(0, self.num_bins - 1)

        xyzw = torch.cat([xyz, delta_yaw.unsqueeze(-1)], dim=-1)  # Shape: (B, Tf, 4)
        return einops.rearrange(xyzw, "b n m -> b (n m)")

    def decode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        tokens: torch.LongTensor,
        hist_tstamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Decodes the given tokens into future trajectories. The future trajectory is returned in
        the same coordinate frame as the historical trajectory. Timestamps can be provided with any
        time-origin.

        Args:
            hist_xyz (torch.Tensor): Historical locations XYZ. Shape: (B, Th, 3).
            hist_rot (torch.Tensor): Historical rotations. Shape: (B, Th, 3, 3).
            tokens (torch.LongTensor): The token indices. Shape: (B, num_tokens_per_trajectory).
            hist_tstamp (torch.Tensor): Historical time stamps. Shape: (B, Th).

        Returns:
            fut_xyz (torch.Tensor): Future locations XYZ. Shape: (B, Tf, 3).
            fut_rot (torch.Tensor): Future rotations. Shape: (B, Tf, 3, 3).
            fut_tstamp (torch.Tensor): Future time stamps. Shape: (B, Tf).
        """
        del hist_tstamp
        m = 4 if self._predict_yaw else 3
        xyzw = einops.rearrange(tokens, "b (n m) -> b n m", m=m).to(hist_xyz.dtype)
        xyz = xyzw[..., :3]
        xyz = xyz / (self.num_bins - 1)
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = xyz * (ego_xyz_max - ego_xyz_min) + ego_xyz_min
        fut_xyz = torch.cumsum(xyz, dim=1)
        if not self._predict_yaw:
            xyz_cpu = fut_xyz.cpu().numpy().astype(float)
            fut_rot = get_yaw_rotation_matrices(xyz_cpu)
            fut_rot = torch.tensor(fut_rot, device=fut_xyz.device, dtype=fut_xyz.dtype)
            return fut_xyz, fut_rot, None
        yaw_tokens = xyzw[..., 3]
        yaw = yaw_tokens.float() / (self.num_bins - 1)
        yaw = yaw * (self.ego_yaw_max - self.ego_yaw_min) + self.ego_yaw_min
        yaw = torch.cumsum(yaw, dim=1)

        # Convert yaw angles to rotation matrices
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        zeros = torch.zeros_like(cos_yaw)
        ones = torch.ones_like(cos_yaw)

        fut_rot = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
                torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        ).to(device=hist_rot.device, dtype=hist_rot.dtype)
        return fut_xyz, fut_rot, None


def get_yaw_rotation_matrices(trajectory, window_size=10, poly_order=3):
    """Calculate yaw rotation matrices using polynomial fitting for both x(t) and y(t)

    Args:
        trajectory: np.array of shape (B, N, 3) for batch of x,y,z coordinates
        window_size: size of window for polynomial fitting
        poly_order: order of polynomial to fit

    Returns:
        rotation_matrices: rotation matrices at each point, shape (B, N, 3, 3)
    """
    B, N = trajectory.shape[:2]
    rotation_matrices = []

    for b in range(B):
        traj_batch = trajectory[b]  # (N, 3)
        batch_matrices = []
        batch_yaws = []

        for i in range(N):
            # Get window indices with padding for edges
            start_idx = max(0, i - window_size // 2)
            end_idx = min(N, start_idx + window_size)

            # Adjust window if at edges
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)

            # Get points in window
            window_points = traj_batch[start_idx:end_idx]

            # Use time parameter t
            t = np.arange(len(window_points))

            # Fit polynomials to both x(t) and y(t)
            x_coeffs = np.polyfit(t, window_points[:, 0], poly_order)
            y_coeffs = np.polyfit(t, window_points[:, 1], poly_order)

            # Calculate derivatives at center point
            center_t = min(i - start_idx, window_size - 1)
            x_deriv = np.polyder(x_coeffs)
            y_deriv = np.polyder(y_coeffs)

            dx = np.polyval(x_deriv, center_t)
            dy = np.polyval(y_deriv, center_t)

            # Calculate yaw angle from dx, dy
            yaw = np.arctan2(dy, dx)
            batch_yaws.append(yaw)

            # Create 3x3 rotation matrix for yaw
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

            batch_matrices.append(rotation_matrix)

        rotation_matrices.append(batch_matrices)

    return np.array(rotation_matrices)
