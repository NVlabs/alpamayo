# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``alpamayo_r1.metrics.distance_metrics`` and ``metric_utils``.

Run:
    pytest src/alpamayo_r1/metrics/test_distance_metrics.py -v
"""

from __future__ import annotations

import torch

from alpamayo_r1.metrics.distance_metrics import compute_ade, compute_minade
from alpamayo_r1.metrics.metric_utils import summarize_metric


def _make_pred_gt(B: int = 2, N: int = 3, K: int = 4, T: int = 60) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic pred/gt tensors of the canonical shapes."""
    torch.manual_seed(0)
    pred = torch.randn(B, N, K, T, 3)
    gt = torch.randn(B, T, 3)
    return pred, gt


def test_compute_ade_returns_per_sample_shape() -> None:
    """``compute_ade`` returns ``[B, N, K]`` mean L2 over time."""
    pred, gt = _make_pred_gt()

    out = compute_ade(pred, gt)

    assert out.shape == (pred.shape[0], pred.shape[1], pred.shape[2])


def test_compute_ade_timestep_horizon_truncates_time_axis() -> None:
    """When ``timestep_horizon`` is set, the L2 average uses only the first H steps."""
    pred, gt = _make_pred_gt(T=20)

    full = compute_ade(pred, gt)
    head_5 = compute_ade(pred, gt, timestep_horizon=5)

    # Head-only mean differs from the full-trajectory mean.
    assert head_5.shape == full.shape
    assert not torch.allclose(head_5, full)


def test_compute_minade_returns_documented_keys() -> None:
    """The returned dict has the keys advertised in the docstring.

    Locks in the contract documented in PR #86: ``min_ade`` plus one
    ``min_ade/by_t={H:.1f}`` per valid horizon, plus the matching ``_std``
    entries when N > 1.
    """
    pred, gt = _make_pred_gt(B=2, N=3, K=4, T=60)

    out = compute_minade(pred, gt, timestep_horizons=[5, 10, 30, 50])

    # Headline mean key.
    assert "min_ade" in out
    assert out["min_ade"].shape == (pred.shape[0],)
    # Per-horizon keys at the documented precision.
    for t in [5, 10, 30, 50]:
        key = f"min_ade/by_t={t * 0.1:.1f}"
        assert key in out, f"missing horizon key {key!r}"
    # _std variants because N > 1.
    assert "min_ade_std" in out
    for t in [5, 10, 30, 50]:
        assert f"min_ade/by_t={t * 0.1:.1f}_std" in out


def test_compute_minade_skips_horizons_exceeding_T() -> None:
    """Horizons larger than the available timesteps are silently dropped."""
    pred, gt = _make_pred_gt(T=10)

    out = compute_minade(pred, gt, timestep_horizons=[5, 30, 100])

    assert f"min_ade/by_t={5 * 0.1:.1f}" in out
    assert f"min_ade/by_t={30 * 0.1:.1f}" not in out
    assert f"min_ade/by_t={100 * 0.1:.1f}" not in out


def test_compute_minade_disable_summary_drops_std() -> None:
    """``disable_summary=True`` returns only the means, no ``_std`` keys."""
    pred, gt = _make_pred_gt(B=1, N=3, K=2, T=10)

    out = compute_minade(pred, gt, timestep_horizons=[5], disable_summary=True)

    assert "min_ade" in out
    assert "min_ade_std" not in out


def test_summarize_metric_adds_std_when_N_gt_1() -> None:
    """Each input key gets a matching ``<key>_std`` entry when N > 1."""
    metric = {"x": torch.randn(2, 5)}

    out = summarize_metric(metric)

    assert "x" in out
    assert "x_std" in out
    assert out["x"].shape == (2,)
    assert out["x_std"].shape == (2,)


def test_summarize_metric_no_std_when_N_eq_1() -> None:
    """N == 1 means there is no spread to summarize: no ``_std`` is added."""
    metric = {"x": torch.randn(2, 1)}

    out = summarize_metric(metric)

    assert "x" in out
    assert "x_std" not in out


def test_summarize_metric_no_std_when_disable_summary_true() -> None:
    """``disable_summary=True`` suppresses ``_std`` even with N > 1."""
    metric = {"x": torch.randn(2, 5)}

    out = summarize_metric(metric, disable_summary=True)

    assert "x" in out
    assert "x_std" not in out


def test_summarize_metric_rejects_wrong_rank() -> None:
    """Input tensors must be 2-D ``[B, N]``; otherwise raise ValueError."""
    bad = {"x": torch.randn(2, 3, 4)}

    try:
        summarize_metric(bad)
    except ValueError:
        return
    raise AssertionError("summarize_metric should reject non-2D inputs")
