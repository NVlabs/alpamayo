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

import einops
import torch
from alpamayo_r1.geometry.rotation import round_2pi_torch, so3_to_yaw_torch

logger = logging.getLogger(__name__)


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """Unwrap the last dimension of the tensor to make sure the diff is in (-pi, pi]."""
    d = torch.diff(phi, dim=-1)
    d = round_2pi_torch(d)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def first_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the banded matrix for the first-order smoothing term."""
    D = torch.zeros(*lead_shape, N - 1, N, dtype=dtype, device=device)
    rows = torch.arange(N - 1, device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 1.0
    return D


def second_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the banded matrix for the second-order smoothing term."""
    D = torch.zeros(*lead_shape, max(N - 2, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 2, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 2.0
    D[..., rows, rows + 2] = -1.0
    return D


def third_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the banded matrix for the third-order smoothing term."""
    D = torch.zeros(*lead_shape, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def construct_DTD(
    N: int,
    lead: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    """Construct the dense matrix D^T s D for multiple orders of smoothing.

    Explanation of the smoothing lambda term:

    For 1st/2nd/3rd smoothing term, we are multiplying the DTD by 1/dt**2, 1/dt**4 and 1/dt**6
    respectively. The reason is that, for example for the 2nd order smoothing term, we are
    minimizing the following term:
        sum_i={0, ..., N-1} lambda * w_smooth2_i * (d^2 x_i / dt^2)**2
    After taking the derivative, we will get the following on the LHS in the normal equation:
        lambda / dt**4 * w_smooth2_i * DTD_2nd.
    Similar explanation applies to the 1st and 3rd order smoothing terms.

    Args:
        N: int, the length of the solving variables.
        lead: tuple, the shape of the leading dimensions of the output matrix.
        device: torch.device, the device of the output matrix.
        dtype: torch.dtype, the dtype of the output matrix.
        w_smooth1: float | torch.Tensor | None, the weight for the first-order smoothing term.
        w_smooth2: float | torch.Tensor | None, the weight for the second-order smoothing term.
        w_smooth3: float | torch.Tensor | None, the weight for the third-order smoothing term.
        lam: float, the weight for the smoothing term.
        dt: float, the time step.

    Returns:
        DTD: torch.Tensor, the dense matrix D^T s D for multiple orders of smoothing.
    """
    DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)
    if w_smooth1 is not None:
        lam_1 = lam / dt**2
        if isinstance(w_smooth1, float):
            w_smooth1_tensor = torch.full(
                (*lead, max(N - 1, 0)), w_smooth1, dtype=dtype, device=device
            )
        else:
            w_smooth1_tensor = w_smooth1
        D1 = first_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_1 * einops.einsum(
            D1 * w_smooth1_tensor.unsqueeze(-1), D1, "... i j, ... i k -> ... j k"
        )

    if w_smooth2 is not None:
        lam_2 = lam / dt**4
        if isinstance(w_smooth2, float):
            w_smooth2_tensor = torch.full(
                (*lead, max(N - 2, 0)), w_smooth2, dtype=dtype, device=device
            )
        else:
            w_smooth2_tensor = w_smooth2
        D2 = second_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_2 * einops.einsum(
            D2 * w_smooth2_tensor.unsqueeze(-1), D2, "... i j, ... i k -> ... j k"
        )

    if w_smooth3 is not None:
        lam_3 = lam / dt**6
        if isinstance(w_smooth3, float):
            w_smooth3_tensor = torch.full(
                (*lead, max(N - 3, 0)), w_smooth3, dtype=dtype, device=device
            )
        else:
            w_smooth3_tensor = w_smooth3
        D3 = third_order_D(N, lead, device=device, dtype=dtype)

        DTD += lam_3 * einops.einsum(
            D3 * w_smooth3_tensor.unsqueeze(-1), D3, "... i j, ... i k -> ... j k"
        )

    return DTD


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Solve a single-point constrained sequence with multiple orders of smoothing.

    This function solves the following problem:
        min_x={x_1, ..., x_N} sum_i={0, ..., N-1} w_data_i (x_i - x_target_i)**2 + smooth_terms
        subject to:
            x_0 = x_init

    Args:
        x_init: the initial value.
        x_target: the target value.
        w_data: the weight for the data term.
        w_smooth1: the weight for the first-order smoothing term.
        w_smooth2: the weight for the second-order smoothing term.
        w_smooth3: the weight for the third-order smoothing term.
        lam: the weight for the smoothing term.
        ridge: the ridge for regularization.
        dt: the time step.

    Returns:
        x: the solved value.
    """
    device, dtype = x_target.device, x_target.dtype
    *lead, N = x_target.shape
    if N <= 0:
        raise ValueError("x_mid must have a positive last-dimension length N.")
    if w_data is None:
        w_data = torch.ones_like(x_target)
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)

    # Solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, x_target, "... i j, ... i -> ... j")

    # The dim is N + 1 because we have x_init as the first element
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * x_init.unsqueeze(-1)

    ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    # strip off the x_init term
    lhs = ATA + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N)

    x = torch.cat([x_init.unsqueeze(-1), x], dim=-1)  # (..., N+1)
    return x


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_xs_eq_y(
    s: torch.Tensor,
    y: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Solve the following problem:

    min_x={x_0, ..., x_N-1} sum_i={0, ..., N-1} w_data_i (x_i * s_i - y_i)**2 + smooth_terms

    Args:
        s (..., N): the slope
        y (..., N): the y-value
        w_data (..., N): the weight for the data term
        w_smooth1: the weight for the first-order smoothing term
        w_smooth2: the weight for the second-order smoothing term
        w_smooth3: the weight for the third-order smoothing term
        lam: the weight for smoothness term
        ridge: the ridge for regularization
        dt: the time step

    Returns:
        x: the solved value.
    """
    device, dtype = y.device, y.dtype
    *lead, N = y.shape
    if w_data is None:
        w_data = torch.ones_like(y)
    if w_data.shape != y.shape:
        raise ValueError("w_data must have the same shape as y")

    # Solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.diag_embed(s)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, y, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )

    # NOTE: Since there is no terminal constraint, we need to handle the singularity case by
    # increasing the ridge term.
    L = None
    while L is None:
        try:
            ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
            lhs = ATA + DTD + ridge_term
            # Ensure dtype consistency for torch.compile fake tensor meta pass
            if rhs.dtype != lhs.dtype:
                rhs = rhs.to(lhs.dtype)
            L = torch.linalg.cholesky(lhs)
        except RuntimeError as e:
            logger.error(f"Error in cholesky decomposition: {e}", exc_info=True)
            ridge *= 10
            logger.warning(f"Resolving singularity using ridge {ridge}")

    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v_without_v0(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    """Given the dxy and theta, compute the velocity.
    The velocity is defined by the trapezoidal integration:

    define:
        u_t = [cos theta_t, sin theta_t]
        Δp_t = p_t+1 - p_t
    We have:
        Δp_t = dt / 2 * (v_t u_t + v_t+1 u_t+1)
        v_t u_t + v_t+1 u_t+1 = 2 * Δp_t / dt
    => Solve v_t, t=0, ..., N by least squares

    This function is shared by the accel and jerk curvature action spaces.

    Args:
        dxy: (..., N, 2) p_t+1 - p_t for t=0, ..., N
        theta: (..., N+1)
        dt: float, the time step
        v_lambda: float, the lambda for the velocity smoothing term
        v_ridge: float, the ridge for the velocity regularization term

    Returns:
        v: (..., N+1) the estimated velocity from 0 to N
    """
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy  # (..., N, 2)

    w = torch.ones_like(dxy[..., 0])

    # solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, b_data, "... i j, ... i -> ... j")

    # The dim is N + 1 because we have x_init as the first element
    # NOTE: for Tikhonov regularization
    # 1st order means we want small acceleration
    # 2nd order means we want small jerk
    # 3rd order means we want small difference between jerk
    # We use 3rd order here as we do not want to penalize the jerk itself directly but only
    # smoothness of the jerk.
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )

    ridge_term = v_ridge * torch.eye(N + 1, dtype=dtype, device=device).expand(*lead, N + 1, N + 1)
    # strip off the x_init term
    lhs = ATA + DTD + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N+1)

    return y  # (..., N+1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    """Given the dxy and theta, compute the velocity.
    The velocity is defined by the trapezoidal integration:

    define:
        u_t = [cos theta_t, sin theta_t]
        Δp_t = p_t+1 - p_t
    We have:
        Δp_t = dt / 2 * (v_t u_t + v_t+1 u_t+1)
        v_t u_t + v_t+1 u_t+1 = 2 * Δp_t / dt
    => Solve v_t, t=1, ..., N by least squares
    Args:
        dxy: (..., N, 2) p_t+1 - p_t for t=0, ..., N
        theta: (..., N+1)
        v0: (...,)

    Returns:
        v: (..., N+1) the estimated velocity from 0 to N
    """
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy  # (..., N, 2)

    w = torch.ones_like(dxy[..., 0])

    # solve the normal equation
    # (A^TA + D^TD + ridge * I) x = A^T b
    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        # rhs is A^T w_data b, but we need to include the x_init terms into the rhs as it is a
        # constant.
        rhs = einops.einsum(Aw_data[..., :, 1:], b_data, "... i j, ... i -> ... j")
    rhs -= ATA[..., 1:, 0] * v0.unsqueeze(-1)

    # The dim is N + 1 because we have x_init as the first element
    # NOTE: for Tikhonov regularization
    # 1st order means we want small acceleration
    # 2nd order means we want small jerk
    # 3rd order means we want small difference between jerk
    # We use 3rd order here as we do not want to penalize the jerk itself directly but only
    # smoothness of the jerk.
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * v0.unsqueeze(-1)

    ridge_term = v_ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    # strip off the x_init term
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (..., N)

    return torch.cat([v0.unsqueeze(-1), y], dim=-1)  # (..., N+1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def theta_smooth(
    traj_future_rot: torch.Tensor,
    dt: float = 1.0,
    theta_lambda: float = 1e-4,
    theta_ridge: float = 1e-4,
) -> torch.Tensor:
    """Smooth the heading of the trajectory.

    Args:
        traj_future_rot: (..., T, 3, 3)
    """
    theta = so3_to_yaw_torch(traj_future_rot)
    theta = unwrap_angle(theta)
    theta_init = torch.zeros_like(theta[..., 0])
    return solve_single_constraint(
        x_init=theta_init,
        x_target=theta,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )
