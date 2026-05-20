"""Unit tests using analytically known solutions for the unicycle kinematics.

These tests verify that traj_to_action and action_to_traj are consistent
with the expected unicycle (non-holonomic) kinematic model using
closed-form analytical trajectories.
"""

import mlx.core as mx
import pytest

from mlx_port.models.alpamayo_r1_mlx import ActionSpace as MLXActionSpace


def generate_straight_line_trajectory(
    v: float = 1.0,
    dt: float = 0.1,
    n_future: int = 8,
    batch_size: int = 1,
) -> tuple[mx.array, mx.array, mx.array, mx.array, float]:
    """Generate an analytically exact straight-line trajectory.

    This corresponds to:
        accel = 0, kappa = 0  (constant velocity, zero curvature)

    Returns:
        hist_xyz, hist_rot, fut_xyz, fut_rot, v0
        where v0 is the initial speed (scalar per batch item, or float).
    """
    # History: 3 past steps, all at origin with identity rotation
    hist_xyz = mx.zeros((batch_size, 3, 3))
    eye = mx.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=mx.float32)
    hist_rot = mx.broadcast_to(eye[None, None, :, :], (batch_size, 3, 3, 3))

    # Future trajectory: straight line along +x at constant speed v
    t = mx.arange(1, n_future + 1, dtype=mx.float32) * dt   # time steps
    x = v * t
    y = mx.zeros_like(x)
    z = mx.zeros_like(x)

    fut_xyz_single = mx.stack([x, y, z], axis=-1)[None, :, :]
    fut_xyz = mx.broadcast_to(fut_xyz_single, (batch_size, n_future, 3))

    # Rotation: identity (yaw = 0) for all future steps
    fut_rot = mx.broadcast_to(eye[None, None, :, :], (batch_size, n_future, 3, 3))

    return hist_xyz, hist_rot, fut_xyz, fut_rot, v


def test_analytical_straight_line_zero_action():
    """Test that a perfect straight-line constant-velocity trajectory
    recovers (near) zero accel and zero curvature when given the correct v0.
    """
    aspace = MLXActionSpace()
    aspace.dt = 0.1
    aspace.n_waypoints = 8

    v0 = 1.0
    hist_xyz, hist_rot, fut_xyz, fut_rot, _ = generate_straight_line_trajectory(
        v=v0, dt=0.1, n_future=8, batch_size=2
    )

    # Supply the correct initial velocity so the solver has the right starting point
    t0_states = {"v": mx.full((2,), v0, dtype=mx.float32)}
    action = aspace.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot, t0_states=t0_states)

    accel = action[..., 0]
    kappa = action[..., 1]

    print(f"[Analytical Test] Recovered accel mean: {float(mx.mean(mx.abs(accel))):.6f}")
    print(f"[Analytical Test] Recovered kappa mean: {float(mx.mean(mx.abs(kappa))):.6f}")

    # With correct v0 the analytical solution (accel=0, kappa=0) should be recovered
    # to high numerical precision. The regularized solver achieves ~5e-5 in float32.
    assert float(mx.mean(mx.abs(accel))) < 1e-4, f"Accel mean abs too large: {float(mx.mean(mx.abs(accel))):.4f}"
    assert float(mx.mean(mx.abs(kappa))) < 1e-4, f"Kappa mean abs too large: {float(mx.mean(mx.abs(kappa))):.4f}"


def test_action_to_traj_analytical_zero_action():
    """Test that zero action produces a (near) stationary or straight trajectory."""
    aspace = MLXActionSpace()
    aspace.dt = 0.1
    aspace.n_waypoints = 8

    B, N = 2, 8
    zero_action = mx.zeros((B, N, 2))

    hist_xyz = mx.zeros((B, 3, 3))
    eye = mx.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=mx.float32)
    hist_rot = mx.broadcast_to(eye.reshape(1, 1, 3, 3), (B, 3, 3, 3))

    fut_xyz, fut_rot = aspace.action_to_traj(zero_action, hist_xyz, hist_rot)

    # With zero action and v0=0, the trajectory should stay near the origin
    # (or move very little). We mainly check shape and basic sanity.
    assert fut_xyz.shape == (B, N, 3)
    assert fut_rot.shape == (B, N, 3, 3)

    # With zero action and v0=0 the unicycle must produce exactly zero displacement
    displacement = mx.sqrt(mx.sum(fut_xyz[..., :2] ** 2, axis=-1))
    print(f"[Analytical Test] Max displacement with zero action: {float(mx.max(displacement)):.10f}")
    assert mx.all(displacement < 1e-6), f"Zero action must produce near-zero displacement, got max={float(mx.max(displacement)):.2e}"


# =============================================================================
# Constant Turn (Circular Arc) Analytical Solution
# =============================================================================

def generate_constant_turn_trajectory(
    v: float = 1.0,
    kappa: float = 0.2,
    dt: float = 0.1,
    n_future: int = 8,
    batch_size: int = 1,
) -> tuple[mx.array, mx.array, mx.array, mx.array, float]:
    """Generate an analytically exact constant-curvature circular arc trajectory.

    For constant speed v and constant curvature κ:
        accel = 0
        kappa = κ  (constant)

    The path is a circular arc of radius R = 1/|κ|.

    Returns:
        hist_xyz, hist_rot, fut_xyz, fut_rot, v0
    """
    R = 1.0 / kappa if abs(kappa) > 1e-6 else 1e6

    # Arc length at each future step
    s = mx.arange(1, n_future + 1, dtype=mx.float32) * (v * dt)
    theta = kappa * s   # heading change = κ * s

    # Parametric equations for circle starting at (0,0) with initial heading +x
    # (left turn for positive κ)
    x = R * mx.sin(theta)
    y = R * (1.0 - mx.cos(theta))

    fut_xyz_single = mx.stack([x, y, mx.zeros_like(x)], axis=-1)[None, :, :]
    fut_xyz = mx.broadcast_to(fut_xyz_single, (batch_size, n_future, 3))

    # Rotation matrices (yaw = theta)
    c = mx.cos(theta)
    s_ = mx.sin(theta)  # sin is a builtin, so use s_
    eye = mx.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=mx.float32)
    # Build rotation matrices for each time step
    rot_list = []
    for i in range(n_future):
        R_i = mx.array([
            [c[i], -s_[i], 0.0],
            [s_[i],  c[i], 0.0],
            [0.0,    0.0,  1.0],
        ])
        rot_list.append(R_i)
    fut_rot_single = mx.stack(rot_list, axis=0)[None, :, :, :]  # (1, N, 3, 3)
    fut_rot = mx.broadcast_to(fut_rot_single, (batch_size, n_future, 3, 3))

    # History: 3 past steps at origin, identity rotation
    hist_xyz = mx.zeros((batch_size, 3, 3))
    hist_rot = mx.broadcast_to(eye.reshape(1, 1, 3, 3), (batch_size, 3, 3, 3))

    return hist_xyz, hist_rot, fut_xyz, fut_rot, v


def test_analytical_constant_turn():
    """Test that a perfect constant-curvature circular trajectory recovers
    (near) zero accel and the correct constant curvature when given proper v0.
    """
    aspace = MLXActionSpace()
    aspace.dt = 0.1
    aspace.n_waypoints = 8

    v0 = 1.0
    kappa_target = 0.15
    hist_xyz, hist_rot, fut_xyz, fut_rot, _ = generate_constant_turn_trajectory(
        v=v0, kappa=kappa_target, dt=0.1, n_future=8, batch_size=2
    )

    t0_states = {"v": mx.full((2,), v0, dtype=mx.float32)}
    action = aspace.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot, t0_states=t0_states)

    accel = action[..., 0]
    kappa_rec = action[..., 1]

    print(f"[Analytical Turn] Recovered accel mean: {float(mx.mean(mx.abs(accel))):.6f}")
    print(f"[Analytical Turn] Recovered kappa mean: {float(mx.mean(kappa_rec)):.6f}")
    print(f"[Analytical Turn] Recovered kappa std:  {float(mx.std(kappa_rec)):.6f}")

    # With correct v0 we expect:
    #   accel ≈ 0          (secondary quantity, small residual from regularization)
    #   kappa ≈ kappa_target (constant) to machine precision
    # Use a practical bound of 1e-4 to account for regularization effects in solve_xs_eq_y.
    assert float(mx.mean(mx.abs(accel))) < 1e-4, f"Accel mean abs too large: {float(mx.mean(mx.abs(accel))):.4f}"
    assert mx.all(mx.isfinite(action)), "Action must be finite"

    # Kappa must match the target to high numerical precision (~1e-5 to 1e-6).
    # Small tolerance accounts for regularization effects and float32 round-trip.
    mean_k = float(mx.mean(kappa_rec))
    abs_err = abs(mean_k - kappa_target)
    assert abs_err < 1e-5, f"Recovered kappa {mean_k:.8f} deviates from target {kappa_target:.8f} by {abs_err:.2e}"

    # And it must be constant across the horizon to high numerical precision (~1e-5).
    std_k = float(mx.std(kappa_rec))
    assert std_k < 1e-5, f"Kappa not constant (std={std_k:.2e}, mean={mean_k:.8f})"


# =============================================================================
# Combined Linear + Turning Motion (accel and kappa both non-zero)
# =============================================================================

def test_analytical_combined_accel_and_turn():
    """Test simultaneous constant linear acceleration and constant curvature.

    Uses a deterministic action profile:
        accel = 0.10 (constant linear acceleration)
        kappa = 0.12 (constant left turn)

    The forward kinematics (action_to_traj) produces a trajectory that both
    speeds up and turns (a spiral-like path). The inverse (traj_to_action)
    must recover the exact same constant action profile.
    """
    aspace = MLXActionSpace()
    aspace.dt = 0.1
    aspace.n_waypoints = 8

    B = 2
    N = 8
    # Deterministic mixed action: constant linear acceleration + constant curvature
    accel_profile = mx.full((N,), 0.10, dtype=mx.float32)         # (8,)
    kappa_profile = mx.full((N,), 0.12, dtype=mx.float32)         # (8,)
    action = mx.stack([accel_profile, kappa_profile], axis=-1)    # (8, 2)
    action = mx.broadcast_to(action[None, :, :], (B, N, 2))       # (B, 8, 2)

    v0 = 0.8  # non-zero initial speed makes the coupling non-trivial
    hist_xyz = mx.zeros((B, 3, 3))
    eye = mx.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=mx.float32)
    hist_rot = mx.broadcast_to(eye.reshape(1, 1, 3, 3), (B, 3, 3, 3))

    # Forward: action -> trajectory (ground truth for this action), with correct initial velocity
    t0_states = {"v": mx.full((B,), v0, dtype=mx.float32)}
    fut_xyz, fut_rot = aspace.action_to_traj(action, hist_xyz, hist_rot, dt=aspace.dt, t0_states=t0_states)

    # Inverse: trajectory -> action, with correct initial velocity
    recovered = aspace.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot, t0_states=t0_states)

    # Explicit numerical difference check between ground truth and recovered action.
    # Ground truth: accel = 0.10 (constant), kappa = 0.12 (constant)
    gt_accel = 0.10
    gt_kappa = 0.12

    accel_err = mx.abs(recovered[..., 0] - gt_accel)
    kappa_err = mx.abs(recovered[..., 1] - gt_kappa)

    mean_accel_err = float(mx.mean(accel_err))
    max_accel_err = float(mx.max(accel_err))
    mean_kappa_err = float(mx.mean(kappa_err))
    max_kappa_err = float(mx.max(kappa_err))

    print(f"[Combined Motion] Accel error  mean={mean_accel_err:.6f}  max={max_accel_err:.6f}")
    print(f"[Combined Motion] Kappa error  mean={mean_kappa_err:.6f}  max={max_kappa_err:.6f}")

    assert mx.all(mx.isfinite(recovered)), "Recovered action must be finite"

    # With dv/dt scaling fix + SciPy Cholesky, expect ~1e-4 to 1e-3 fidelity due to
    # jerk-smoothing regularization in solve_xs_eq_y (even for exact constant profiles).
    assert mean_kappa_err < 1e-3, f"Kappa reconstruction error too large: mean={mean_kappa_err:.6f}"
    assert mean_accel_err < 1e-3, f"Accel reconstruction error too large: mean={mean_accel_err:.6f}"


# =============================================================================
# Constant Linear Acceleration, Straight Line (kappa = 0)
# =============================================================================

def test_analytical_constant_accel_straight_line():
    """Test constant linear acceleration with zero curvature (pure straight-line acceleration).

    Uses a deterministic action profile:
        accel = 0.15 (constant linear acceleration)
        kappa = 0.0   (straight line, no turning)

    The forward kinematics (action_to_traj) produces a trajectory that
    accelerates in a straight line. The inverse (traj_to_action) must recover
    the exact constant accel and exactly zero kappa.

    This test isolates the Cholesky solver behavior when there is no heading
    coupling from turning, to contrast with the combined accel+turn case.
    """
    aspace = MLXActionSpace()
    aspace.dt = 0.1
    aspace.n_waypoints = 8

    B = 2
    N = 8
    # Deterministic straight-line accelerating action
    accel_profile = mx.full((N,), 0.15, dtype=mx.float32)         # (8,)
    kappa_profile = mx.zeros((N,), dtype=mx.float32)              # (8,)
    action = mx.stack([accel_profile, kappa_profile], axis=-1)    # (8, 2)
    action = mx.broadcast_to(action[None, :, :], (B, N, 2))       # (B, 8, 2)

    v0 = 0.5  # non-zero initial speed
    hist_xyz = mx.zeros((B, 3, 3))
    eye = mx.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=mx.float32)
    hist_rot = mx.broadcast_to(eye.reshape(1, 1, 3, 3), (B, 3, 3, 3))

    # Forward: action -> trajectory (ground truth), with correct initial velocity
    t0_states = {"v": mx.full((B,), v0, dtype=mx.float32)}
    fut_xyz, fut_rot = aspace.action_to_traj(action, hist_xyz, hist_rot, dt=aspace.dt, t0_states=t0_states)

    # Inverse: trajectory -> action, with correct initial velocity
    recovered = aspace.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot, t0_states=t0_states)

    # Ground truth values
    gt_accel = 0.15
    gt_kappa = 0.0

    accel_err = mx.abs(recovered[..., 0] - gt_accel)
    kappa_err = mx.abs(recovered[..., 1] - gt_kappa)

    mean_accel_err = float(mx.mean(accel_err))
    max_accel_err = float(mx.max(accel_err))
    mean_kappa_err = float(mx.mean(kappa_err))
    max_kappa_err = float(mx.max(kappa_err))

    print(f"[Constant Accel Straight] Accel error  mean={mean_accel_err:.6f}  max={max_accel_err:.6f}")
    print(f"[Constant Accel Straight] Kappa error  mean={mean_kappa_err:.6f}  max={max_kappa_err:.6f}")

    assert mx.all(mx.isfinite(recovered)), "Recovered action must be finite"

    # With dv/dt scaling fix + SciPy Cholesky, expect ~1e-4 to 1e-3 fidelity due to
    # jerk-smoothing regularization in solve_xs_eq_y (even for exact constant profiles).
    assert mean_kappa_err < 1e-3, f"Kappa should be near-zero: mean={mean_kappa_err:.6f}"
    assert mean_accel_err < 1e-3, f"Accel reconstruction error too large: mean={mean_accel_err:.6f}"
