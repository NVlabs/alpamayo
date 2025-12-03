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

import math

import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Normalize the input tensor."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Normalize the input tensor."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLPEncoder(nn.Module):
    """Basic MLP encoder."""

    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        assert 1 <= num_enc_layers, f"{num_enc_layers=} must be >= 1"

        enc_layers = [
            nn.Linear(num_input_feats, hidden_size),
            nn.SiLU(),
        ]
        for layeri in range(num_enc_layers):
            if layeri < num_enc_layers - 1:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                    ]
                )
            else:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, outdim),
                    ]
                )

        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C) -> (B, outdim)"""
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    """Improved Fourier feature encoder with logarithmically-spaced frequencies."""

    def __init__(self, dim: int, max_freq: float = 100.0):
        """Initialize the Fourier encoder V2.

        Args:
            dim: Output dimension of the encoder. Must be even as it's split into
                sine and cosine components.
            max_freq: Maximum frequency for the logarithmic frequency spacing.
                Defaults to 100.0.
        """
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])  # (1, half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier encoder V2.

        Args:
            x: Input tensor of arbitrary shape (..., ).

        Returns:
            Fourier-encoded features of shape (..., dim).
        """
        arg = x[..., None] * self.freqs * 2 * torch.pi  # (*, half_dim)
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class PerWaypointActionInProjV2(torch.nn.Module):
    """Improved per-waypoint action input projection module.

    It uses FourierEncoderV2 with logarithmically-spaced frequencies and includes layer normalization. Projects
    action sequences with timestep information into a higher-dimensional representation.
    """

    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        """Initialize the per-waypoint action projection module V2.

        Args:
            in_dims: List of input dimensions. The last element specifies the number
                of action dimensions to encode separately.
            out_dim: Output dimension of the projection.
            num_enc_layers: Number of layers in the MLP encoder. Defaults to 4.
            hidden_size: Hidden dimension size of the MLP encoder. Defaults to 1024.
            max_freq: Maximum frequency for the Fourier encoding. Defaults to 100.0.
            num_fourier_feats: Number of Fourier features for encoding. Defaults to 20.
        """
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        sinus = []
        for _ in range(in_dims[-1]):
            sinus.append(FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq))
        self.sinus = nn.ModuleList(sinus)
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        num_input_feats = sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the per-waypoint action projection V2.

        Args:
            x: Action tensor of shape (batch_size, num_waypoints, action_dim).
            timesteps: Timestep tensor of shape (batch_size, ...). The last dimension
                is used for encoding.

        Returns:
            Normalized projected action features of shape
            (batch_size, num_waypoints, out_dim).
        """
        B, T, _ = x.shape

        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, T, 1)
        x = torch.cat((action_feats, timestep_feats), dim=-1)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))
