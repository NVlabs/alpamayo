"""MLX port of NVIDIA's DiscreteTrajectoryTokenizer and DeltaTrajectoryTokenizer.

These are used to convert ego history/future trajectories into discrete tokens
for the VLM prompt and to decode actions back to trajectories.
"""

from typing import Any, Tuple

import mlx.core as mx

from mlx_port.models.alpamayo_r1_mlx import ActionSpace


class DiscreteTrajectoryTokenizerMLX:
    """MLX port of DiscreteTrajectoryTokenizer.

    This tokenizer uses an ActionSpace (UnicycleAccelCurvature) to convert
    trajectories to actions, then discretizes the actions into tokens.
    """

    def __init__(
        self,
        action_space: ActionSpace | None = None,
        dims_min: list[float] | None = None,
        dims_max: list[float] | None = None,
        num_bins: int = 1000,
    ):
        self.action_space = action_space or ActionSpace()
        # Default normalization bounds for accel/curvature (typical values)
        self.dims_min = mx.array(dims_min or [-5.0, -1.0])
        self.dims_max = mx.array(dims_max or [5.0, 1.0])
        self.num_bins = num_bins

    def encode(
        self,
        hist_xyz: mx.array,
        hist_rot: mx.array,
        fut_xyz: mx.array,
        fut_rot: mx.array,
        hist_tstamp: mx.array | None = None,
        fut_tstamp: mx.array | None = None,
    ) -> mx.array:
        """Encode trajectories into discrete action tokens.

        For history tokenization we treat the history as "future" for the
        tokenizer (as done in NVIDIA's tokenize_history_trajectory).
        """
        # Flatten batch
        B = hist_xyz.shape[0]
        hist_xyz_flat = hist_xyz.reshape(B, -1, 3)
        hist_rot_flat = hist_rot.reshape(B, -1, 3, 3)
        fut_xyz_flat = fut_xyz.reshape(B, -1, 3)
        fut_rot_flat = fut_rot.reshape(B, -1, 3, 3)

        # Real implementation: convert history trajectory to actions via traj_to_action
        # Always provide t0_states with zero v0 for history tokenization (no ego velocity in prompt)
        B = hist_xyz_flat.shape[0]
        t0_states = {"v": mx.zeros((B,), dtype=mx.float32)}
        action = self.action_space.traj_to_action(
            hist_xyz_flat, hist_rot_flat, fut_xyz_flat, fut_rot_flat, t0_states=t0_states
        )
        # Discretize (simple linear quantization into num_bins)
        action = mx.clip(action, self.dims_min, self.dims_max)
        scale = self.dims_max - self.dims_min
        tokens = ((action - self.dims_min) / scale * (self.num_bins - 1)).astype(mx.int32)
        # Flatten last two dims (accel + kappa per waypoint)
        tokens = tokens.reshape(B, -1)
        return tokens

    def decode(
        self,
        hist_xyz: mx.array,
        hist_rot: mx.array,
        tokens: mx.array,
        hist_tstamp: mx.array | None = None,
    ) -> Tuple[mx.array, mx.array, Any]:
        """Decode tokens back to future trajectories."""
        # This would use action_space.action_to_traj after denormalization.
        # For now we delegate to the ActionSpace we already implemented.
        action = tokens.reshape(-1, *self.action_space.get_action_space_dims())
        fut_xyz, fut_rot = self.action_space.action_to_traj(
            action, hist_xyz, hist_rot
        )
        return fut_xyz, fut_rot, None