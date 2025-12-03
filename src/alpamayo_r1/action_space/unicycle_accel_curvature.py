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

import torch
from alpamayo_r1.action_space.action_space import ActionSpace
from alpamayo_r1.action_space.utils import (
    dxy_theta_to_v,
    dxy_theta_to_v_without_v0,
    solve_xs_eq_y,
    theta_smooth,
    unwrap_angle,
)
from alpamayo_r1.geometry.rotation import (
    rot_2d_to_3d,
    rotation_matrix_torch,
    so3_to_yaw_torch,
)

logger = logging.getLogger(__name__)


class UnicycleAccelCurvatureActionSpace(ActionSpace):
    """Unicycle Kinematic Model with acceleration and curvature as control inputs."""

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),  # min and max bounds for accel
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),  # min and max bounds for curvature
        dt: float = 0.1,
        n_waypoints: int = 64,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ):
        """Initialize the UnicycleAccelCurvatureActionSpace.

        Args:
            accel_mean: Mean for normalizing acceleration.
            accel_std: Std for normalizing acceleration.
            curvature_mean: Mean for normalizing curvature.
            curvature_std: Std for normalizing curvature.
            accel_bounds: Acceleration bounds (min, max).
                This value is used to check if the acceleration is within bounds.
            curvature_bounds: Curvature bounds (min, max).
                This value is used to check if the curvature is within bounds.
            dt: Time step interval.
            n_waypoints: Number of waypoints in the trajectory.
            theta_lambda: Lambda parameter for theta smoothing.
            theta_ridge: Ridge parameter for theta smoothing.
            v_lambda: Lambda parameter for velocity smoothing.
            v_ridge: Ridge parameter for velocity smoothing.
            a_lambda: Lambda parameter for acceleration smoothing.
            a_ridge: Ridge parameter for acceleration smoothing.
            kappa_lambda: Lambda parameter for curvature smoothing.
            kappa_ridge: Ridge parameter for curvature smoothing.
        """
        super().__init__()
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))
        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge
        self.a_lambda = a_lambda
        self.a_ridge = a_ridge
        self.kappa_lambda = kappa_lambda
        self.kappa_ridge = kappa_ridge

    def get_action_space_dims(self) -> tuple[int, int]:
        """Get the dimensions of the action space."""
        return (self.n_waypoints, 2)

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Check if a normalized action is within bounds.

        Args:
            action: (..., N, 2)

        Returns:
            is_within_bounds: (...,)
        """
        accel = action[..., 0]
        kappa = action[..., 1]
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean
        is_accel_within_bounds = (accel >= self.accel_bounds[0]) & (accel <= self.accel_bounds[1])
        is_kappa_within_bounds = (kappa >= self.curvature_bounds[0]) & (
            kappa <= self.curvature_bounds[1]
        )
        return torch.all(is_accel_within_bounds & is_kappa_within_bounds, dim=-1)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the acceleration from the velocity.

        Define:
            Δv_t = v_t+1 - v_t

        According to the kinematic model
            Δv_t = dt * a_t

        => solve it by single-constrained solver

        Args:
            v: (..., N+1)

        Returns:
            a: (..., N)
        """
        dv = (v[..., 1:] - v[..., :-1]) / self.dt  # (..., N)
        # NOTE: for Tikhonov regularization
        # 1st order means we want small jerk
        # 2nd order means we want small difference between jerk
        # We use 2nd order here as we do not want to penalize the jerk itself directly but only
        # smoothness of the jerk.
        a = solve_xs_eq_y(
            s=torch.ones_like(dv),
            y=dv,
            dt=self.dt,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
        )
        return a

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the curvature from the theta, v, a, jerk.

        The kappa is computed by

            s = dt * v + dt^2 * a / 2
            kappa = dtheta / s

        where dtheta is the unwrapped heading difference.

        Args:
            theta: (..., N+1) unwrapped heading
            v: (..., N+1) velocity
            a: (..., N) acceleration

        Returns:
            kappa: (..., N)
        """
        dtheta = theta[..., 1:] - theta[..., :-1]  # (..., N)
        dt = self.dt
        s = dt * v[..., :-1] + (dt**2) / 2.0 * a  # (..., N)

        w = torch.ones_like(dtheta)
        # NOTE: for Tikhonov regularization
        # 1st order means we want small kappa 1st order difference
        # 2nd order means we want small kappa 2nd order difference
        return solve_xs_eq_y(
            s=s,
            y=dtheta,
            w_data=w,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self, traj_history_xyz: torch.Tensor, traj_history_rot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Estimate the t0 states from the trajectory history."""
        full_xy = traj_history_xyz[..., :2]  # (..., N_hist, 2)
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]  # (..., N_hist-1, 2)
        theta = so3_to_yaw_torch(traj_history_rot)
        theta = unwrap_angle(theta)

        v = dxy_theta_to_v_without_v0(
            dxy=dxy, theta=theta, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge
        )  # (..., N+1)
        v_t0 = v[..., -1]
        return {"v": v_t0}

    @torch.no_grad()
    @torch._dynamo.disable()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
        output_all_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Transform the future trajectory to the action space.

        Here we assume the traj_history_xyz[..., -1, :] is the current position and is all zeros.

        Args:
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
            t0_states: initial state estimate
            output_all_states: whether to output all the states

        Returns:
            action: (..., T, 2)
        """
        # Validate inputs
        if traj_future_xyz.shape[-2] != self.n_waypoints:
            raise ValueError(
                f"future trajectory must have length {self.n_waypoints} "
                f"but got {traj_future_xyz.shape[-2]}"
            )

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        # Concatenate last history and future
        # NOTE: we assume the traj_history_xyz[..., -1, :] is the current position and it is all
        # zero.
        full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[
            ..., :2
        ]  # (..., N+1, 2)

        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]  # (..., N, 2)
        theta = theta_smooth(
            traj_future_rot=traj_future_rot,
            dt=self.dt,
            theta_lambda=self.theta_lambda,
            theta_ridge=self.theta_ridge,
        )

        v0 = t0_states["v"]
        v = dxy_theta_to_v(
            dxy=dxy, theta=theta, v0=v0, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge
        )  # (..., N+1)

        accel = self._v_to_a(v)  # (..., N+1), (..., N)

        kappa = self._theta_v_a_to_kappa(theta, v, accel)  # (..., N)

        # normalize acceleration and kappa
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = (accel - accel_mean) / accel_std
        kappa = (kappa - kappa_mean) / kappa_std

        if not output_all_states:
            return torch.stack([accel, kappa], dim=-1)  # (..., N, 2)
        else:
            return torch.stack([accel, kappa], dim=-1), torch.stack(
                [v[:, :-1], accel, theta[:, :-1]], dim=-1
            )

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the action space to the trajectory.

        Args:
            action: (..., T, 2)
            traj_history_xyz: (..., T, 3)
            traj_history_rot: (..., T, 3, 3)
            t0_states: initial state estimate

        Returns:
            traj_future_xyz: (..., T, 3)
            traj_future_rot: (..., T, 3, 3)
        """
        accel, kappa = action[..., 0], action[..., 1]

        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        v0 = t0_states["v"]
        dt = self.dt

        dt_2_term = 0.5 * (self.dt**2)
        velocity = torch.cat(
            [
                v0.unsqueeze(-1),
                (v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)),
            ],
            dim=-1,
        )  # (..., N+1)
        initial_yaw = torch.zeros_like(v0)
        theta = torch.cat(
            [
                initial_yaw.unsqueeze(-1),
                (
                    initial_yaw.unsqueeze(-1)
                    + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                    + torch.cumsum(kappa * accel * dt_2_term, dim=-1)
                ),
            ],
            dim=-1,
        )  # (..., N+1)
        half_dt_term = 0.5 * dt
        initial_x = torch.zeros_like(v0)
        initial_y = torch.zeros_like(v0)
        x = (
            initial_x.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt_term, dim=-1)
        )  # (..., N)
        y = (
            initial_y.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt_term, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt_term, dim=-1)
        )  # (..., N)
        batch_dim = traj_history_xyz.shape[:-2]
        traj_future_xyz = torch.zeros(
            *batch_dim,
            self.n_waypoints,
            3,
            device=traj_history_xyz.device,
            dtype=traj_history_xyz.dtype,
        )
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        # Handle only_xy case for output
        traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]

        traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))

        return traj_future_xyz, traj_future_rot
