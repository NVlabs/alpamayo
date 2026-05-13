"""
MLX implementation of Flow Matching (Euler integrator) for the Alpamayo action expert.

This is the inference-time sampler that repeatedly calls the expert's step function
to denoise a noisy action trajectory into a clean future trajectory.
"""

from typing import Callable, Optional

import mlx.core as mx


class FlowMatching:
    """
    Flow Matching sampler using Euler integration.

    This is a direct port of the logic in src/alpamayo_r1/diffusion/flow_matching.py
    but written for MLX tensors.
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        int_method: str = "euler",
    ):
        self.num_inference_steps = num_inference_steps
        self.int_method = int_method

    def sample(
        self,
        batch_size: int,
        step_fn: Callable,
        x_dims: tuple[int, ...] = (64, 2),   # (n_waypoints, action_dim)
        device: str = "cpu",
        return_all_steps: bool = False,
    ):
        """
        Sample a trajectory using Flow Matching.

        Args:
            batch_size: Number of trajectories to sample.
            step_fn: Callable that takes (x, t) and returns the vector field / noise prediction.
            x_dims: Shape of one action trajectory (n_waypoints, action_dim).
            return_all_steps: If True, also return the intermediate states.

        Returns:
            Sampled action trajectories of shape (batch_size, *x_dims)
        """
        # Start from Gaussian noise
        x = mx.random.normal((batch_size, *x_dims))

        # Time steps from 1.0 -> 0.0
        dt = 1.0 / self.num_inference_steps
        timesteps = mx.linspace(1.0, 0.0, self.num_inference_steps + 1)[:-1]

        all_steps = [x] if return_all_steps else None

        for t in timesteps:
            t_tensor = mx.full((batch_size, 1), t, dtype=x.dtype)
            pred = step_fn(x=x, t=t_tensor)
            x = x + dt * pred

            if return_all_steps:
                all_steps.append(x)

        if return_all_steps:
            return x, all_steps
        return x
