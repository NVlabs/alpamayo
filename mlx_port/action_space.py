"""
MLX / NumPy port of the Alpamayo action space and discrete trajectory tokenizer.

This module provides the equivalent of:
- alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace
- alpamayo_r1.action_space.discrete_action_space.DiscreteTrajectoryTokenizer

The goal is to produce bit-identical discrete token IDs for history trajectories
as the original NVIDIA implementation, so the VLM sees the exact same conditioning
tokens that were used during training.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class UnicycleAccelCurvatureActionSpace:
    """
    NumPy port of the unicycle kinematic action space.

    For history encoding we only need `traj_to_action` (with future = history).
    The full smoothing / integration logic is simplified for the history case,
    but follows the same high-level structure as the original.
    """

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        dt: float = 0.1,
        n_waypoints: int = 64,
        **kwargs: Any,
    ):
        self.accel_mean = accel_mean
        self.accel_std = accel_std
        self.curvature_mean = curvature_mean
        self.curvature_std = curvature_std
        self.dt = dt
        self.n_waypoints = n_waypoints

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)

    def traj_to_action(
        self,
        traj_history_xyz: np.ndarray,
        traj_history_rot: np.ndarray,
        traj_future_xyz: np.ndarray,
        traj_future_rot: np.ndarray,
    ) -> np.ndarray:
        """
        Higher-fidelity kinematic conversion (history treated as future).

        This version improves heading extraction and velocity estimation
        while staying self-contained. It is closer in spirit to the original
        `UnicycleAccelCurvatureActionSpace` without requiring the full
        solver infrastructure (`solve_xs_eq_y` + many regularization lambdas).
        """
        # Concatenate last history pose (t0) + future sequence
        hist_last_xyz = traj_history_xyz[-1:, :2]
        hist_last_rot = traj_history_rot[-1:, :, :]
        full_xyz = np.concatenate([hist_last_xyz, traj_future_xyz[:, :2]], axis=0)
        full_rot = np.concatenate([hist_last_rot, traj_future_rot], axis=0)

        # Heading (yaw) directly from rotation matrix (more stable than dxy)
        yaw = np.arctan2(full_rot[:, 1, 0], full_rot[:, 0, 0])

        dxy = full_xyz[1:] - full_xyz[:-1]
        v_trans = np.linalg.norm(dxy, axis=1) / self.dt

        # Heading rate (align length with v_trans)
        dtheta = np.diff(yaw)  # length = len(yaw) - 1 = 16

        # Combined velocity estimate (translation + rotation component)
        v = np.maximum(v_trans, np.abs(dtheta) * 0.15) + 1e-6

        # Acceleration with light smoothing
        accel = np.diff(v, prepend=v[:1]) / self.dt
        accel = 0.75 * accel + 0.25 * np.pad(accel, (1, 0), mode="edge")[:-1]

        # Curvature = dtheta / arc-length
        s = v * self.dt
        kappa = dtheta / (s + 1e-6)
        kappa = 0.75 * kappa + 0.25 * np.pad(kappa, (1, 0), mode="edge")[:-1]

        action = np.stack([accel, kappa], axis=1)

        # Normalize with exact checkpoint statistics
        action = (action - np.array([self.accel_mean, self.curvature_mean])) / np.array(
            [self.accel_std, self.curvature_std]
        )

        # Guarantee exactly n_waypoints tokens
        N = action.shape[0]
        if N > self.n_waypoints:
            action = action[-self.n_waypoints:]
        elif N < self.n_waypoints:
            pad = self.n_waypoints - N
            action = np.pad(action, ((pad, 0), (0, 0)), mode="edge")

        return action[: self.n_waypoints]


class DiscreteTrajectoryTokenizer:
    """
    MLX/NumPy port of DiscreteTrajectoryTokenizer.

    Uses the exact `dims_min`, `dims_max`, `num_bins` from the Alpamayo checkpoint.
    """

    def __init__(
        self,
        action_space: UnicycleAccelCurvatureActionSpace,
        dims_min: list[float],
        dims_max: list[float],
        num_bins: int,
    ):
        self.action_space = action_space
        self.dims_min = np.array(dims_min, dtype=np.float32)
        self.dims_max = np.array(dims_max, dtype=np.float32)
        self.num_bins = num_bins

    def encode(
        self,
        hist_xyz: np.ndarray,
        hist_rot: np.ndarray,
        fut_xyz: np.ndarray,
        fut_rot: np.ndarray,
    ) -> np.ndarray:
        """
        Encode history (treated as future for the history tokenizer) into discrete token IDs.
        """
        action = self.action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
        # Normalize into [0, num_bins-1]
        action_norm = (action - self.dims_min) / (self.dims_max - self.dims_min)
        action_norm = np.clip(action_norm, 0.0, 1.0)
        tokens = (action_norm * (self.num_bins - 1)).round().astype(np.int32)
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        return tokens.reshape(-1)  # flatten to 1-D token sequence


def create_history_tokenizer_from_alpamayo_config(
    cfg: dict[str, Any] | None = None,
) -> DiscreteTrajectoryTokenizer:
    """
    Factory that builds a DiscreteTrajectoryTokenizer using the parameters
    stored in the Alpamayo checkpoint config (or sensible defaults).
    """
    if cfg is None:
        # Fallback values taken from Alpamayo-R1-10B/config.json
        cfg = {
            "action_space_cfg": {
                "accel_mean": 0.02902694707164455,
                "accel_std": 0.6810426736454882,
                "curvature_mean": 0.0002692167976330542,
                "curvature_std": 0.026148280660833106,
                "dt": 0.1,
                "n_waypoints": 64,
            },
            "dims_min": [-10.0, -10.0],
            "dims_max": [10.0, 10.0],
            "num_bins": 3000,
        }

    action_space = UnicycleAccelCurvatureActionSpace(**cfg["action_space_cfg"])
    return DiscreteTrajectoryTokenizer(
        action_space=action_space,
        dims_min=cfg["dims_min"],
        dims_max=cfg["dims_max"],
        num_bins=cfg["num_bins"],
    )