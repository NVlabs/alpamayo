# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FDE / minFDE metrics in ``alpamayo_r1.metrics.distance_metrics``.

Run:
    pytest src/alpamayo_r1/metrics/test_fde.py -v
"""

from __future__ import annotations

import torch

from alpamayo_r1.metrics.distance_metrics import compute_ade, compute_fde, compute_minfde


def _make_pred_gt(
    B: int = 2, N: int = 3, K: int = 4, T: int = 60
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic pred/gt tensors of the canonical shapes."""
    torch.manual_seed(0)
    pred = torch.randn(B, N, K, T, 3)
    gt = torch.randn(B, T, 3)
    return pred, gt


def test_compute_fde_returns_per_sample_shape() -> None:
    """FDE returns ``[B, N, K]`` (no time axis: it is a single per-sample scalar)."""
    pred, gt = _make_pred_gt()

    out = compute_fde(pred, gt)

    assert out.shape == (pred.shape[0], pred.shape[1], pred.shape[2])
    # FDE is non-negative (it's an L2 norm).
    assert torch.all(out >= 0)


def test_compute_fde_matches_explicit_final_step_l2() -> None:
    """FDE must equal the L2 distance at the last timestep on XY by default."""
    pred, gt = _make_pred_gt(B=1, N=1, K=1, T=10)

    out = compute_fde(pred, gt).squeeze()
    expected = torch.linalg.norm(pred[0, 0, 0, -1, :2] - gt[0, -1, :2])

    assert torch.allclose(out, expected)


def test_compute_fde_only_xy_false_uses_xyz() -> None:
    """only_xy=False must use the full 3D vector, not just XY."""
    pred, gt = _make_pred_gt(B=1, N=1, K=1, T=5)

    fde_xy = compute_fde(pred, gt, only_xy=True).squeeze()
    fde_xyz = compute_fde(pred, gt, only_xy=False).squeeze()

    expected_xyz = torch.linalg.norm(pred[0, 0, 0, -1, :] - gt[0, -1, :])
    assert torch.allclose(fde_xyz, expected_xyz)
    # XYZ distance is at least as large as the XY-projection distance.
    assert fde_xyz >= fde_xy


def test_compute_fde_timestep_horizon_picks_that_timestep() -> None:
    """``timestep_horizon=H`` evaluates FDE at index H-1 (the H-th timestep)."""
    pred, gt = _make_pred_gt(B=1, N=1, K=1, T=20)

    fde_at_5 = compute_fde(pred, gt, timestep_horizon=5).squeeze()
    expected = torch.linalg.norm(pred[0, 0, 0, 4, :2] - gt[0, 4, :2])

    assert torch.allclose(fde_at_5, expected)


def test_compute_fde_horizon_too_large_raises() -> None:
    """Horizons larger than T must raise ValueError, mirroring compute_ade."""
    pred, gt = _make_pred_gt(B=1, N=1, K=1, T=10)

    try:
        compute_fde(pred, gt, timestep_horizon=11)
    except ValueError:
        return
    raise AssertionError("compute_fde should reject timestep_horizon > T")


def test_compute_fde_independent_of_intermediate_steps() -> None:
    """FDE depends only on the last timestep -- intermediate noise must not affect it."""
    pred1, gt = _make_pred_gt(B=1, N=1, K=1, T=10)
    pred2 = pred1.clone()
    # Perturb every timestep except the last.
    pred2[..., :-1, :] += 100.0

    fde1 = compute_fde(pred1, gt).squeeze()
    fde2 = compute_fde(pred2, gt).squeeze()

    assert torch.allclose(fde1, fde2)


def test_compute_fde_does_not_equal_ade_in_general() -> None:
    """Sanity: FDE and ADE differ for non-trivial trajectories.

    Locks in that compute_fde is genuinely a different metric, not an alias.
    """
    pred, gt = _make_pred_gt()

    ade = compute_ade(pred, gt)
    fde = compute_fde(pred, gt)

    assert ade.shape == fde.shape
    # With random data the two metrics agree at exactly zero points.
    assert not torch.allclose(ade, fde)


def test_compute_minfde_emits_documented_keys() -> None:
    """``compute_minfde`` returns ``min_fde`` plus a per-horizon key per valid t."""
    pred, gt = _make_pred_gt(B=2, N=3, K=4, T=60)

    out = compute_minfde(pred, gt, timestep_horizons=[5, 10, 30, 50])

    assert "min_fde" in out
    assert out["min_fde"].shape == (pred.shape[0],)
    for t in [5, 10, 30, 50]:
        assert f"min_fde/by_t={t * 0.1:.1f}" in out
    # _std variants because N > 1.
    assert "min_fde_std" in out
    for t in [5, 10, 30, 50]:
        assert f"min_fde/by_t={t * 0.1:.1f}_std" in out


def test_compute_minfde_skips_horizons_exceeding_T() -> None:
    """Horizons larger than T are silently dropped (mirrors compute_minade)."""
    pred, gt = _make_pred_gt(T=10)

    out = compute_minfde(pred, gt, timestep_horizons=[5, 30, 100])

    assert f"min_fde/by_t={5 * 0.1:.1f}" in out
    assert f"min_fde/by_t={30 * 0.1:.1f}" not in out
    assert f"min_fde/by_t={100 * 0.1:.1f}" not in out


def test_compute_minfde_disable_summary_drops_std() -> None:
    """``disable_summary=True`` returns only the means (no ``_std`` keys)."""
    pred, gt = _make_pred_gt(B=1, N=3, K=2, T=10)

    out = compute_minfde(pred, gt, timestep_horizons=[5], disable_summary=True)

    assert "min_fde" in out
    assert "min_fde_std" not in out


def test_compute_minfde_takes_min_over_K() -> None:
    """min_fde must equal min over K of compute_fde (then averaged over N)."""
    pred, gt = _make_pred_gt(B=1, N=1, K=4, T=10)

    fde = compute_fde(pred, gt)  # [1, 1, 4]
    expected_min_fde = fde.min(dim=2).values.mean(dim=1)  # [1]

    out = compute_minfde(pred, gt, timestep_horizons=[])

    assert torch.allclose(out["min_fde"], expected_min_fde)
