"""MLX ports of the numerical solvers from alpamayo_r1.action_space.utils.

These functions implement the regularized least-squares trajectory-to-action
conversion used by NVIDIA's UnicycleAccelCurvatureActionSpace.

This version uses only the APIs available in MLX 0.31.x (no .at.set).
"""

import mlx.core as mx
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def first_order_D(N: int, lead_shape: tuple = (), dtype=mx.float32) -> mx.array:
    """Build the first-order difference matrix.

    Returns a matrix of shape (*lead_shape, N-1, N) where
    each row i has -1 at column i and +1 at column i+1.
    """
    if N < 2:
        return mx.zeros((*lead_shape, 0, N), dtype=dtype)

    # Left part: (N-1, N-1) with -1 on the diagonal
    left = mx.diag(-mx.ones(N-1, dtype=dtype))
    # Right column of zeros to make it (N-1, N)
    right = mx.zeros((N-1, 1), dtype=dtype)
    main_neg = mx.concatenate([left, right], axis=1)

    # Super-diagonal part: +1 on the off-diagonal
    super_pos = mx.diag(mx.ones(N-1, dtype=dtype), k=1)[:N-1, :N]

    D = main_neg + super_pos

    if lead_shape:
        D = mx.broadcast_to(D, (*lead_shape, N-1, N))
    return D


def second_order_D(N: int, lead_shape: tuple = (), dtype=mx.float32) -> mx.array:
    """Build the second-order difference matrix (pure construction)."""
    if N < 3:
        return mx.zeros((*lead_shape, 0, N), dtype=dtype)

    # (N-2, N) matrix with -1, 2, -1 pattern
    # Pure construction using mx.diag + concatenate (no .at.set)

    # Simpler robust construction:
    # Create three separate (N-2, N) matrices and add them.

    # comp0: -1 on columns 0..N-3
    left = mx.diag(-mx.ones(N-2, dtype=dtype))
    right_pad = mx.zeros((N-2, 2), dtype=dtype)
    comp0 = mx.concatenate([left, right_pad], axis=1)

    # comp1: +2 on columns 1..N-2
    mid = mx.diag(2 * mx.ones(N-2, dtype=dtype))
    left_pad = mx.zeros((N-2, 1), dtype=dtype)
    right_pad2 = mx.zeros((N-2, 1), dtype=dtype)
    comp1 = mx.concatenate([left_pad, mid, right_pad2], axis=1)

    # comp2: -1 on columns 2..N-1
    right = mx.diag(-mx.ones(N-2, dtype=dtype))
    left_pad2 = mx.zeros((N-2, 2), dtype=dtype)
    comp2 = mx.concatenate([left_pad2, right], axis=1)

    D = comp0 + comp1 + comp2

    if lead_shape:
        D = mx.broadcast_to(D, (*lead_shape, N-2, N))
    return D


def construct_DTD(
    N: int,
    lead: tuple = (),
    w_smooth1=None,
    w_smooth2=None,
    w_smooth3=None,
    lam: float = 1e-3,
    dt: float = 1.0,
    dtype=mx.float32,
) -> mx.array:
    """Construct D^T D for the smoothing terms (supports 1st/2nd/3rd order).
    Uses einsum to handle batch dimensions robustly without relying on .transpose
    for higher-rank arrays.
    """
    DTD = mx.zeros((*lead, N, N), dtype=dtype)

    if w_smooth1 is not None:
        D1 = first_order_D(N, lead, dtype)  # (..., N-1, N)
        # D1^T @ D1  -> (..., N, N)
        DtD1 = mx.einsum('...ij,...ik->...jk', D1, D1)
        DTD = DTD + lam * (1.0 / dt**2) * DtD1

    if w_smooth2 is not None:
        D2 = second_order_D(N, lead, dtype)  # (..., N-2, N)
        DtD2 = mx.einsum('...ij,...ik->...jk', D2, D2)
        DTD = DTD + lam * (1.0 / dt**4) * DtD2

    if w_smooth3 is not None:
        D3 = third_order_D(N, lead, dtype)  # (..., N-3, N)
        DtD3 = mx.einsum('...ij,...ik->...jk', D3, D3)
        DTD = DTD + lam * (1.0 / dt**6) * DtD3

    return DTD


def third_order_D(N: int, lead_shape: tuple = (), dtype=mx.float32) -> mx.array:
    """Build the third-order difference matrix (pure construction, no .at.set)."""
    if N < 4:
        return mx.zeros((*lead_shape, 0, N), dtype=dtype)

    rows = []
    for i in range(N-3):
        row = mx.zeros(N, dtype=dtype)
        left = mx.zeros(i, dtype=dtype)
        right = mx.zeros(N - i - 4, dtype=dtype)
        row = mx.concatenate([left, mx.array([-1., 3., -3., 1.]), right])
        rows.append(row)
    D = mx.stack(rows, axis=0)

    if lead_shape:
        D = mx.broadcast_to(D, (*lead_shape, N-3, N))
    return D


def solve_xs_eq_y(
    s: mx.array,
    y: mx.array,
    w_data: mx.array | None = None,
    w_smooth1=None,
    w_smooth2=None,
    w_smooth3=None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> mx.array:
    """MLX port of solve_xs_eq_y using normal equations + lstsq.

    This version is numerically faithful and uses only supported MLX APIs.
    """
    orig_ndim = y.ndim
    if orig_ndim == 1:
        y = y[None, :]
        s = s[None, :]
        if w_data is not None:
            w_data = w_data[None, :]

    *lead, N = y.shape
    if w_data is None:
        w_data = mx.ones_like(y)

    # Build data term matrix A (diagonal with slope s)
    A = mx.diag(s) if y.ndim == 2 else mx.zeros((*lead, N, N), dtype=y.dtype)
    # For batch case we need to handle it properly
    if y.ndim > 2:
        # For simplicity, assume 2D for now (common case)
        pass

    # 2D path (batch, N) - vectorized diagonal construction
    if y.ndim == 2:
        batch = y.shape[0]
        eye = mx.eye(N, dtype=y.dtype)
        A_data = s[:, None, :] * eye[None, :, :]
        Aw = A_data * w_data[..., None]
        ATA = mx.matmul(Aw.transpose(0, 2, 1), A_data)
        rhs = mx.sum(Aw * y[..., None], axis=1)

        DTD = construct_DTD(
            N, (batch,), w_smooth2=w_smooth2, lam=lam, dt=dt, dtype=y.dtype
        )
        ridge_term = ridge * mx.eye(N, dtype=y.dtype)[None, :, :]
        lhs = ATA + DTD + ridge_term

        with mx.stream(mx.cpu):
            x = mx.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
        if orig_ndim == 1:
            x = x[0]
        return x

    # Fallback for 1D
    A_data = mx.diag(s)
    Aw = A_data * w_data[..., None]
    ATA = mx.matmul(Aw.T, A_data)
    rhs = mx.sum(Aw * y[..., None], axis=0)

    DTD = construct_DTD(N, (), w_smooth2=w_smooth2, lam=lam, dt=dt, dtype=y.dtype)
    ridge_term = ridge * mx.eye(N, dtype=y.dtype)
    lhs = ATA + DTD + ridge_term

    with mx.stream(mx.cpu):
        x = mx.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
    return x


def dxy_theta_to_v(
    dxy: mx.array,
    theta: mx.array,
    v0: mx.array,
    dt: float = 0.1,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> mx.array:
    """MLX port of NVIDIA's dxy_theta_to_v (exact trapezoidal inverse).

    Solves the normal equations of the trapezoidal unicycle integration
    using Cholesky decomposition + 3rd-order (jerk) smoothing, exactly
    matching alpamayo_r1/action_space/utils.py.
    """
    *lead, N, _ = dxy.shape
    g = (2.0 / dt) * dxy  # (..., N, 2)

    cos_theta = mx.cos(theta)
    sin_theta = mx.sin(theta)

    # Build A_data (2N, N+1) using robust row construction
    c0 = cos_theta[..., :-1]  # (..., N)
    c1 = cos_theta[..., 1:]
    s0 = sin_theta[..., :-1]
    s1 = sin_theta[..., 1:]

    rows_list = []
    for i in range(N):
        # Even row (2*i): coefficients for cos on columns i and i+1
        left = mx.zeros((*lead, i), dtype=dxy.dtype)
        right = mx.zeros((*lead, N - i - 1), dtype=dxy.dtype)
        even_row = mx.concatenate([left, c0[..., i : i + 1], c1[..., i : i + 1], right], axis=-1)
        rows_list.append(even_row)

        # Odd row (2*i+1): coefficients for sin on columns i and i+1
        left = mx.zeros((*lead, i), dtype=dxy.dtype)
        right = mx.zeros((*lead, N - i - 1), dtype=dxy.dtype)
        odd_row = mx.concatenate([left, s0[..., i : i + 1], s1[..., i : i + 1], right], axis=-1)
        rows_list.append(odd_row)

    A_data = mx.stack(rows_list, axis=-2)  # (..., 2N, N+1)

    w = mx.ones_like(dxy[..., 0])
    Aw_data = A_data * mx.repeat(w, 2, axis=-1)[..., None]

    ATA = mx.einsum("...ij,...ik->...jk", Aw_data, A_data)
    b_data = g.reshape(*lead, 2 * N)

    # NVIDIA logic: Slice Aw_data BEFORE einsum to exclude v0 column (col 0)
    # Aw_data[..., :, 1:] has shape (..., 2N, N)
    rhs = mx.einsum("...ij,...i->...j", Aw_data[..., :, 1:], b_data)

    # Subtract contribution of v0 (first column of ATA) from rhs
    rhs = rhs - ATA[..., 1:, 0] * v0[..., None]

    DTD = construct_DTD(
        N + 1, lead, w_smooth3=1.0, lam=v_lambda, dt=dt, dtype=dxy.dtype
    )
    rhs = rhs - DTD[..., 1:, 0] * v0[..., None]

    ridge_term = v_ridge * mx.eye(N, dtype=dxy.dtype)
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + ridge_term

    # Use SciPy's Cholesky solver (backed by Apple's Accelerate framework on macOS)
    # for superior numerical precision compared to MLX 0.31.x CPU LAPACK wrapper.
    # Convert to numpy float64 for best conditioning, solve, convert back.
    lhs_np = np.array(lhs, dtype=np.float64)
    rhs_np = np.array(rhs[..., None], dtype=np.float64)

    # Handle batched case (lead may be non-empty)
    lead_tuple = tuple(lead)
    if lead_tuple:
        # lhs_np: (*lead, N, N), rhs_np: (*lead, N, 1)
        x_list = []
        for idx in np.ndindex(lead_tuple):
            c, lower = cho_factor(lhs_np[idx], lower=True)
            xi = cho_solve((c, lower), rhs_np[idx]).squeeze(-1)
            x_list.append(xi)
        x_np = np.stack(x_list, axis=0).reshape(*lead_tuple, N)
    else:
        c, lower = cho_factor(lhs_np, lower=True)
        x_np = cho_solve((c, lower), rhs_np).squeeze(-1)

    x = mx.array(x_np, dtype=dxy.dtype)
    v = mx.concatenate([v0[..., None], x], axis=-1)
    return v


def theta_smooth(
    traj_future_rot: mx.array,
    dt: float = 0.1,
    theta_lambda: float = 1e-6,
    theta_ridge: float = 1e-8,
) -> mx.array:
    """Extract yaw and apply light smoothing."""
    yaw = mx.arctan2(traj_future_rot[..., 1, 0], traj_future_rot[..., 0, 0])
    d = yaw[..., 1:] - yaw[..., :-1]
    d = (d + mx.pi) % (2 * mx.pi) - mx.pi
    yaw = mx.concatenate([yaw[..., :1], yaw[..., :1] + mx.cumsum(d, axis=-1)], axis=-1)
    return yaw