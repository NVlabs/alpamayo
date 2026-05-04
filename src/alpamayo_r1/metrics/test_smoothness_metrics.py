# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for smoothness_metrics.

Run:
    pytest src/alpamayo_r1/metrics/test_smoothness_metrics.py -v
"""

from __future__ import annotations

import math

import torch

from alpamayo_r1.metrics.smoothness_metrics import (
    _SMOOTHNESS_SOURCES,
    compute_smoothness_metrics,
    gather_dynamics,
)


def _identity_rot(*shape: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return identity rotation matrices broadcast to the requested batch shape."""
    return torch.eye(3, dtype=dtype).expand(*shape, 3, 3).contiguous()


def _make_constant_velocity(
    B: int = 2, N: int = 3, K: int = 4, T: int = 16, vx: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant straight-line motion at vx m/s, identity heading.

    A constant-velocity trajectory should have zero acceleration, zero jerk,
    zero lateral motion and zero yaw rate -- everything except v_lon should
    be (numerically) zero. Uses float64 so finite-difference noise stays at
    machine precision instead of the ~1e-4 float32 noise that twice-
    differentiating a 0.1 s step accumulates.
    """
    t = torch.arange(T, dtype=torch.float64) * 0.1
    x = vx * t
    pred_xyz = torch.zeros(B, N, K, T, 3, dtype=torch.float64)
    pred_xyz[..., 0] = x
    pred_rot = _identity_rot(B, N, K, T)
    return pred_xyz, pred_rot


def _make_constant_accel(
    B: int = 1, N: int = 1, K: int = 1, T: int = 20, ax: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant longitudinal acceleration ax m/s^2 from rest, identity heading.

    Longitudinal accel should be ~ax everywhere; lateral / yaw signals zero.
    """
    dt = 0.1
    t = torch.arange(T, dtype=torch.float64) * dt
    x = 0.5 * ax * t * t
    pred_xyz = torch.zeros(B, N, K, T, 3, dtype=torch.float64)
    pred_xyz[..., 0] = x
    pred_rot = _identity_rot(B, N, K, T)
    return pred_xyz, pred_rot


def test_gather_dynamics_returns_expected_keys_and_shapes() -> None:
    pred_xyz, pred_rot = _make_constant_velocity()
    dyn = gather_dynamics(pred_xyz, pred_rot)
    expected = {
        "yaw_rate", "yaw_accel", "v_lon", "v_lat",
        "accel_lon", "accel_lat", "jerk_lon",
    }
    assert set(dyn.keys()) == expected
    for k, v in dyn.items():
        assert v.shape == pred_xyz.shape[:-1], f"{k}: {v.shape}"


def test_constant_velocity_has_zero_accel_jerk_yaw() -> None:
    pred_xyz, pred_rot = _make_constant_velocity(vx=5.0)
    dyn = gather_dynamics(pred_xyz, pred_rot)

    # v_lon should be ~5 m/s everywhere.
    assert torch.allclose(dyn["v_lon"], torch.full_like(dyn["v_lon"], 5.0), atol=1e-4)
    # All derived signals should be (numerically) zero.
    for k in ("v_lat", "accel_lon", "accel_lat", "jerk_lon", "yaw_rate", "yaw_accel"):
        assert torch.allclose(dyn[k], torch.zeros_like(dyn[k]), atol=1e-4), k


def test_constant_acceleration_recovers_accel_value() -> None:
    pred_xyz, pred_rot = _make_constant_accel(ax=2.0)
    dyn = gather_dynamics(pred_xyz, pred_rot)

    # accel_lon = diff(v_lon) = diff(diff(x)). Each _diff_pad_last call repeats
    # the last delta, so the LAST two timesteps are unreliable after two
    # nested differentiations. Check the genuine interior only.
    interior_accel = dyn["accel_lon"][..., 1:-2]
    assert torch.allclose(
        interior_accel, torch.full_like(interior_accel, 2.0), atol=1e-2
    ), interior_accel


def test_yaw_rate_handles_pi_wraparound() -> None:
    """Heading jumping from +pi to -pi must yield ~0 yaw_rate, not ~2*pi."""
    pred_xyz = torch.zeros(1, 1, 1, 4, 3, dtype=torch.float64)
    # Build rotations from yaw angles that wrap.
    yaws = torch.tensor(
        [math.pi - 0.05, math.pi - 0.01, -math.pi + 0.02, -math.pi + 0.04],
        dtype=torch.float64,
    )
    cos = torch.cos(yaws)
    sin = torch.sin(yaws)
    pred_rot = _identity_rot(1, 1, 1, 4)
    pred_rot[..., 0, 0] = cos
    pred_rot[..., 1, 0] = sin
    pred_rot[..., 0, 1] = -sin
    pred_rot[..., 1, 1] = cos

    dyn = gather_dynamics(pred_xyz, pred_rot)

    # Yaw is moving forward in small increments; yaw_rate magnitude must be
    # small (radians/s), not ~ 2*pi*10 from a naive subtraction.
    assert torch.all(dyn["yaw_rate"].abs() < 1.0), dyn["yaw_rate"]


def test_compute_smoothness_metrics_emits_all_keys() -> None:
    pred_xyz, pred_rot = _make_constant_velocity()
    out = compute_smoothness_metrics(pred_xyz, pred_rot)
    for prefix in _SMOOTHNESS_SOURCES:
        for suffix in ("_rms", "_max"):
            key = prefix + suffix
            assert key in out, f"missing key {key!r}"
            assert out[key].shape == (pred_xyz.shape[0],), f"{key}: {out[key].shape}"


def test_compute_smoothness_metrics_constant_velocity_is_smooth() -> None:
    """Straight-line motion at constant speed reports near-zero everywhere."""
    pred_xyz, pred_rot = _make_constant_velocity()
    out = compute_smoothness_metrics(pred_xyz, pred_rot)
    for prefix in _SMOOTHNESS_SOURCES:
        for suffix in ("_rms", "_max"):
            v = out[prefix + suffix]
            assert torch.all(v.abs() < 1e-3), f"{prefix}{suffix} = {v}"


def test_compute_smoothness_adds_std_when_N_gt_1() -> None:
    pred_xyz, pred_rot = _make_constant_velocity(N=3)
    out = compute_smoothness_metrics(pred_xyz, pred_rot)
    # _std variants get added by summarize_metric.
    for prefix in _SMOOTHNESS_SOURCES:
        for suffix in ("_rms", "_max"):
            assert (prefix + suffix + "_std") in out


def test_compute_smoothness_drops_std_when_disable_summary() -> None:
    pred_xyz, pred_rot = _make_constant_velocity(N=3)
    out = compute_smoothness_metrics(pred_xyz, pred_rot, disable_summary=True)
    for k in list(out.keys()):
        assert not k.endswith("_std"), f"{k} should not be present"


def test_compute_smoothness_rejects_wrong_shapes() -> None:
    bad_xyz = torch.zeros(2, 3, 4, 10, dtype=torch.float64)  # missing trailing 3
    rot = _identity_rot(2, 3, 4, 10)
    try:
        compute_smoothness_metrics(bad_xyz, rot)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on missing xyz channel dim")

    pred_xyz, _ = _make_constant_velocity()
    bad_rot = torch.zeros(2, 3, 4, 10, 3, 4, dtype=torch.float64)  # 3x4 instead of 3x3
    try:
        compute_smoothness_metrics(pred_xyz, bad_rot)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-3x3 rotation")
