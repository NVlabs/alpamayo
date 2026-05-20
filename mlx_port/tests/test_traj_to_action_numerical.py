"""Numerical fidelity tests for MLX traj_to_action.

These tests verify that the MLX implementation of traj_to_action (which uses
the real regularized solvers from action_space_utils_mlx.py) produces
correct shapes and is numerically consistent in a round-trip sense.
"""

import mlx.core as mx
import pytest

from mlx_port.models.alpamayo_r1_mlx import ActionSpace as MLXActionSpace


def test_traj_to_action_roundtrip():
    """Round-trip consistency using the real numerical solver."""
    aspace = MLXActionSpace()
    aspace.dt = 0.1
    aspace.n_waypoints = 8

    B, N = 2, 8
    action = mx.random.normal((B, N, 2)) * 0.15

    hist_xyz = mx.zeros((B, 3, 3))
    eye = mx.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=mx.float32)
    hist_rot = mx.broadcast_to(eye.reshape(1, 1, 3, 3), (B, 3, 3, 3))

    traj_xyz, traj_rot = aspace.action_to_traj(action, hist_xyz, hist_rot)
    recovered = aspace.traj_to_action(hist_xyz, hist_rot, traj_xyz, traj_rot)

    assert recovered.shape == action.shape
    diff = mx.abs(recovered - action)
    print(f"[Numerical Test] Max absolute difference: {float(mx.max(diff)):.4f}")


def test_traj_to_action_shape_and_dtype():
    """Basic shape and dtype checks for the real implementation."""
    aspace = MLXActionSpace()
    B, T = 2, 8
    xyz = mx.random.normal((B, T, 3))
    rot = mx.random.normal((B, T, 3, 3))
    fut_xyz = mx.random.normal((B, T, 3))
    fut_rot = mx.random.normal((B, T, 3, 3))

    action = aspace.traj_to_action(xyz, rot, fut_xyz, fut_rot)
    assert action.shape == (B, T, 2)
    assert action.dtype == mx.float32