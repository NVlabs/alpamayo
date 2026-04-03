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

"""Visualization utilities for waypoint projection and trajectory plotting."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import textwrap
import scipy.spatial.transform as spt
import cv2


def project_waypoints_ftheta(wp, cam_rot, cam_t, intr):
    """Project 3D waypoints onto 2D image plane using the f-theta camera model.

    Args:
        wp: 3D waypoints in world coordinates, shape (N, 3).
        cam_rot: Camera rotation matrix, shape (3, 3).
        cam_t: Camera translation vector, shape (3,).
        intr: Intrinsic parameters [width, height, cx, cy, fw_poly_0..4].

    Returns:
        Projected 2D pixel coordinates, shape (M, 2) for visible points.
    """
    width, height, cx, cy, fw_poly_0, fw_poly_1, fw_poly_2, fw_poly_3, fw_poly_4 = intr
    cam_points = (wp - cam_t) @ cam_rot

    x, y, z = cam_points.T
    r_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r_xy, z)

    radius = (
        fw_poly_0
        + fw_poly_1 * theta
        + fw_poly_2 * theta**2
        + fw_poly_3 * theta**3
        + fw_poly_4 * theta**4
    )
    scale = radius / (r_xy + 1e-8)
    u = cx + x * scale
    v = cy + y * scale

    projected = np.stack([u, v], axis=1)
    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    return projected[valid]


def viz_waypoints_pai(
    intr_df: pd.DataFrame,
    extr_df: pd.DataFrame,
    image,
    waypoints_gt: np.ndarray,
    waypoints_pred: np.ndarray,
):
    """Overlay ground-truth and predicted waypoints on the front-wide camera image.

    Args:
        intr_df: Camera intrinsics DataFrame indexed by sensor name.
        extr_df: Sensor extrinsics DataFrame indexed by sensor name.
        image: BGR image array of shape (H, W, 3). Modified in-place.
        waypoints_gt: Ground-truth waypoint tensor, or None to skip.
        waypoints_pred: Predicted waypoint tensor, or None to skip.

    Returns:
        The input image with waypoint circles drawn on it.
    """
    cam_intr = intr_df.loc["camera_front_wide_120fov"]
    cam_extr = extr_df.loc["camera_front_wide_120fov"]

    width = cam_intr["width"]
    height = cam_intr["height"]
    cx = cam_intr["cx"]
    cy = cam_intr["cy"]
    fw_poly_0 = cam_intr["fw_poly_0"]
    fw_poly_1 = cam_intr["fw_poly_1"]
    fw_poly_2 = cam_intr["fw_poly_2"]
    fw_poly_3 = cam_intr["fw_poly_3"]
    fw_poly_4 = cam_intr["fw_poly_4"]

    intr_list = [width, height, cx, cy, fw_poly_0, fw_poly_1, fw_poly_2, fw_poly_3, fw_poly_4]

    # extr parameters
    qx = cam_extr["qx"]
    qy = cam_extr["qy"]
    qz = cam_extr["qz"]
    qw = cam_extr["qw"]
    tx = cam_extr["x"]
    ty = cam_extr["y"]
    tz = cam_extr["z"]

    # convert qx, qy, qz, qw to rotation matrix
    rot = spt.Rotation.from_quat([qx, qy, qz, qw])
    rot_matrix = rot.as_matrix()
    cam_t = np.array([tx, ty, tz], dtype=np.float64)

    if waypoints_gt is not None:
        projected_waypoints_gt = project_waypoints_ftheta(
            waypoints_gt.cpu().numpy().squeeze(), rot_matrix, cam_t, intr_list
        )
        for p in projected_waypoints_gt:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

    if waypoints_pred is not None:
        projected_waypoints_pred = project_waypoints_ftheta(
            waypoints_pred.cpu().numpy().squeeze(), rot_matrix, cam_t, intr_list
        )
        for p in projected_waypoints_pred:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    return image


def make_image_grid(images: np.ndarray, columns: int = 4) -> np.ndarray:
    """Tile multiple images into a single row-major grid.

    Args:
        images: Image batch of shape (N, H, W, C).
        columns: Number of columns in the output grid.

    Returns:
        A single image array of shape (rows * H, columns * W, C).
    """
    num = images.shape[0]
    rows = int(np.ceil(num / columns))
    h, w, c = images.shape[1:]
    grid = np.zeros((rows * h, columns * w, c), dtype=images.dtype)
    for idx in range(num):
        r = idx // columns
        col = idx % columns
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = images[idx]
    return grid


def rotate_90cc(xy):
    """Rotate 2D coordinates 90 degrees counter-clockwise: (x, y) -> (-y, x)."""
    return np.stack([-xy[1], xy[0]], axis=0)


def _plot_trajectory_with_fade(
    ax,
    xy_rot: np.ndarray,
    color: str,
    label: str,
    fade_in: bool = True,
) -> None:
    """Plot a 2D trajectory with fading markers to indicate temporal order.

    Args:
        ax: Matplotlib Axes to draw on.
        xy_rot: Coordinate array of shape (2, N).
        color: Matplotlib color specification.
        label: Legend label for this trajectory.
        fade_in: If True, alpha increases over time; otherwise decreases.
    """
    x = xy_rot[0]
    y = xy_rot[1]
    if x.size == 0:
        return

    # Keep line visible, and encode temporal order with sparse, small markers.
    ax.plot(x, y, "-", color=color, alpha=0.45, linewidth=1.2, label=label)
    alphas = np.linspace(0.2, 1.0, x.size) if fade_in else np.linspace(1.0, 0.2, x.size)
    marker_stride = max(1, x.size // 14)
    for idx in range(0, x.size, marker_stride):
        ax.scatter([x[idx]], [y[idx]], c=color, s=7, alpha=float(alphas[idx]))

    # Highlight start/end waypoints.
    ax.scatter([x[0]], [y[0]], c=color, marker="x", s=24, alpha=0.9)
    ax.scatter(
        [x[-1]], [y[-1]], c=color, marker="o", s=18, alpha=1.0, edgecolors="black", linewidths=0.4
    )


def _set_tight_trajectory_limits(
    ax,
    trajectories: list[np.ndarray],
    pad_ratio: float = 0.12,
    min_span: float = 2.0,
) -> None:
    """Set compact, equal-scale axis limits around plotted trajectories.

    Args:
        ax: Matplotlib Axes to configure.
        trajectories: List of arrays, each of shape (2, N).
        pad_ratio: Extra padding as a fraction of the span.
        min_span: Minimum span to avoid degenerate limits.
    """
    if not trajectories:
        return

    xs = np.concatenate([traj[0] for traj in trajectories if traj.size > 0], axis=0)
    ys = np.concatenate([traj[1] for traj in trajectories if traj.size > 0], axis=0)
    if xs.size == 0 or ys.size == 0:
        return

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))

    x_span = max(x_max - x_min, min_span)
    y_span = max(y_max - y_min, min_span)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    # Use a shared span for x/y so metric distance looks isotropic.
    span = max(x_span, y_span)
    half_span = 0.5 * span
    pad = span * pad_ratio

    ax.set_xlim(x_center - half_span - pad, x_center + half_span + pad)
    ax.set_ylim(y_center - half_span - pad, y_center + half_span + pad)
    ax.set_aspect("equal", adjustable="box")


def visualize_data(
    image_frames,
    ego_future_xyz_gt=None,
    ego_future_xyz_pred=None,
    cot_text=None,
    show_waypoint_pai: bool = False,
    extr=None,
    intr=None,
    save_path: str = None,
):
    """Create a composite figure with camera grid, projected waypoints, and BEV trajectories.

    Args:
        image_frames: Camera image tensor of shape (T, num_cams, C, H, W).
        ego_future_xyz_gt: Ground-truth future ego waypoints tensor, or None.
        ego_future_xyz_pred: Predicted future ego waypoints tensor, or None.
        cot_text: Optional chain-of-thought text to display at the bottom.
        show_waypoint_pai: Whether to project waypoints on the front-wide camera image.
        extr: Sensor extrinsics DataFrame (required if show_waypoint_pai is True).
        intr: Camera intrinsics DataFrame (required if show_waypoint_pai is True).
        save_path: If provided, save the figure to this path and close it.
    """
    frames = image_frames.flatten(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    waypoint_viz = None
    if show_waypoint_pai:
        t0_front_wide = frames[7]  # 7 = t0 frame of front wide
        waypoint_viz = viz_waypoints_pai(
            intr,
            extr,
            t0_front_wide,
            ego_future_xyz_gt,
            ego_future_xyz_pred,
        )
    grid = make_image_grid(frames, columns=4)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1, 1])
    ax_grid = fig.add_subplot(gs[:, 0])
    if waypoint_viz is not None:
        ax_waypoint = fig.add_subplot(gs[0, 1])
        ax_pred = fig.add_subplot(gs[1, 1])
    else:
        ax_waypoint = None
        ax_pred = fig.add_subplot(gs[:, 1])

    ax_grid.imshow(grid)
    ax_grid.axis("off")

    if waypoint_viz is not None:
        ax_waypoint.imshow(waypoint_viz)
        ax_waypoint.axis("off")
    plotted_trajectories: list[np.ndarray] = []

    if ego_future_xyz_pred is not None:
        for i in range(ego_future_xyz_pred.shape[2]):
            pred_xy = ego_future_xyz_pred.cpu()[0, 0, i, :, :2].T.numpy()
            pred_xy_rot = rotate_90cc(pred_xy)
            plotted_trajectories.append(pred_xy_rot)
            _plot_trajectory_with_fade(
                ax_pred,
                pred_xy_rot,
                color="b",
                label=f"Predicted Trajectory #{i + 1}",
                fade_in=True,
            )
    if ego_future_xyz_gt is not None:
        gt_xy = ego_future_xyz_gt.squeeze().cpu()[:, :2].T.numpy()
        gt_xy_rot = rotate_90cc(gt_xy)
        plotted_trajectories.append(gt_xy_rot)
        _plot_trajectory_with_fade(
            ax_pred,
            gt_xy_rot,
            color="r",
            label="Ground Truth Trajectory",
            fade_in=True,
        )
    ax_pred.set_ylabel("y coordinate (meters) x for start, o for end")
    ax_pred.set_xlabel("x coordinate (meters)")
    ax_pred.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )

    _set_tight_trajectory_limits(ax_pred, plotted_trajectories)

    if cot_text is not None:
        wrapped_cot = textwrap.fill(cot_text, width=110)
        fig.text(0.5, 0.01, wrapped_cot, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
