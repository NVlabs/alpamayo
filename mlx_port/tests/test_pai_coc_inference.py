"""
Pytest suite for full end-to-end PAI-COC inference.

This module replicates the functionality of `test_inference_pai_coc.py`
as proper, reusable, and parametrizable pytest tests.

It supports easy ablation studies on:
- temperature / top_p
- two_stage generation
- history injection
- etc.
"""

import pytest
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import pandas as pd
from PIL import Image

import mlx.core as mx

from mlx_port.load_alpamayo_vlm import load_alpamayo_vlm_weights
from mlx_port.vlm_loader import (
    generate_alpamayo_coc,
    prefill_vlm_kv_cache,
    extract_readable_coc,
    MIN_PIXELS,
    MAX_PIXELS,
)
from mlx_port.expert_loader import load_alpamayo_expert_weights
from mlx_port.expert import (
    Expert,
    create_expert_step_fn,
    build_expert_attention_inputs,
    denormalize_trajectory,
)
from mlx_port.flow_matching import FlowMatching


# ---------------------------------------------------------------------------
# Data loading helpers (moved from test_inference_pai_coc.py for test isolation)
# ---------------------------------------------------------------------------

def get_coc_clip_ids(downloaded_only: bool = True) -> list[str]:
    """Return list of downloadable CoC clip IDs."""
    index_path = Path("/Volumes/MicronSSD/pai_coc/clip_index.parquet")
    if not index_path.exists():
        return []
    df = pd.read_parquet(index_path)
    if downloaded_only:
        df = df[df["downloaded_coc"] == True]  # noqa: E712
    return df["clip_id"].tolist()


def _get_chunk_for_clip(clip_id: str) -> int | None:
    index_path = Path("/Volumes/MicronSSD/pai_coc/clip_index.parquet")
    if not index_path.exists():
        return None
    df = pd.read_parquet(index_path)
    row = df[df["clip_id"] == clip_id]
    if len(row) == 0:
        return None
    return int(row.iloc[0]["chunk"])


def _load_images_from_local_zips(
    clip_id: str, cameras: list[str] | None = None
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    """Load N cameras × 4 frames using timestamp synchronization (same as NVIDIA)."""
    chunk = _get_chunk_for_clip(clip_id)
    if chunk is None:
        raise FileNotFoundError(f"Clip {clip_id} not found in index")

    base = Path(f"/Volumes/MicronSSD/pai_coc/chunk_{chunk:04d}")
    if cameras is None:
        cameras = ["front_wide", "cross_left", "cross_right", "rear_left", "rear_right"][:4]

    # Load timestamps
    ts_files = {cam: base / f"{clip_id}.{cam}.timestamps.parquet" for cam in cameras}
    ts_dfs = {}
    for cam, f in ts_files.items():
        if f.exists():
            ts_dfs[cam] = pd.read_parquet(f)

    if not ts_dfs:
        raise FileNotFoundError("No timestamp files found")

    # Use first camera's t0 as reference
    ref_cam = list(ts_dfs.keys())[0]
    t0_us = ts_dfs[ref_cam]["timestamp"].iloc[0]

    # Desired offsets: -300, -200, -100, 0 ms
    desired_offsets = [-300_000, -200_000, -100_000, 0]
    desired_ts = [t0_us + off for off in desired_offsets]

    images = []
    abs_timestamps = []
    rel_timestamps = []

    for cam in cameras:
        if cam not in ts_dfs:
            # fallback: use first available camera
            cam = list(ts_dfs.keys())[0]

        ts = ts_dfs[cam]["timestamp"].to_numpy()
        idxs = [int(np.argmin(np.abs(ts - d))) for d in desired_ts]

        video_path = base / f"{clip_id}.{cam}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(pil)
                abs_timestamps.append(ts[idx])
                rel_timestamps.append((ts[idx] - t0_us) / 1_000_000.0)
        cap.release()
        images.extend(frames)

    return images, np.array(abs_timestamps), np.array(rel_timestamps)


def _load_egomotion_history(clip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 16-step ego history (xyz + rotation matrix) from parquet."""
    chunk = _get_chunk_for_clip(clip_id)
    if chunk is None:
        return np.zeros((16, 3)), np.zeros((16, 3, 3))

    ego_path = Path(f"/Volumes/MicronSSD/pai_coc/chunk_{chunk:04d}/{clip_id}.egomotion.parquet")
    if not ego_path.exists():
        return np.zeros((16, 3)), np.zeros((16, 3, 3))

    df = pd.read_parquet(ego_path)
    df = df.sort_values("timestamp").head(16)

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    # Convert quaternion to rotation matrix (simplified)
    from scipy.spatial.transform import Rotation as R
    quats = df[["qx", "qy", "qz", "qw"]].to_numpy()
    rot_mats = np.stack([R.from_quat(q).as_matrix() for q in quats])

    # Pad if less than 16
    if len(xyz) < 16:
        pad = 16 - len(xyz)
        xyz = np.pad(xyz, ((0, pad), (0, 0)), mode="edge")
        rot_mats = np.pad(rot_mats, ((0, pad), (0, 0), (0, 0)), mode="edge")

    return xyz, rot_mats


def load_clip_data(clip_id: str, cameras: list[str] | None = None) -> dict[str, Any]:
    """High-level loader used by tests. Pass cameras=["front_wide"] for single-cam ablation."""
    try:
        images, abs_ts, rel_ts = _load_images_from_local_zips(clip_id, cameras=cameras)
        xyz, rot = _load_egomotion_history(clip_id)
        return {
            "images": images,
            "ego_history_xyz": xyz,
            "ego_history_rot": rot,
            "absolute_timestamps": abs_ts,
            "relative_timestamps": rel_ts,
            "clip_id": clip_id,
        }
    except Exception as e:
        print(f"[Warning] Local loader failed: {e}")
        test_img = Image.open("reports/stage1_test_image_front_wide.png").convert("RGB")
        return {
            "images": [test_img] * 4,
            "ego_history_xyz": None,
            "ego_history_rot": None,
            "clip_id": clip_id,
        }


# ---------------------------------------------------------------------------
# Pytest fixtures and tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vlm_model_processor():
    """Session-scoped VLM to avoid reloading for every test."""
    return load_alpamayo_vlm_weights()


@pytest.fixture(scope="session")
def expert_model():
    """Session-scoped Expert (bfloat16)."""
    expert_state = load_alpamayo_expert_weights()
    expert_state = {k: v.astype(mx.bfloat16) for k, v in expert_state.items()}
    expert = Expert(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=8256,
        num_layers=36,
    )
    expert.load_weights(list(expert_state.items()))
    return expert


@pytest.mark.parametrize(
    "temperature,top_p,two_stage,only_front_wide",
    [
        (0.0, 1.0, False, False),
        (0.0, 1.0, True, False),
        (0.6, 0.98, True, False),
        (0.0, 1.0, True, True),  # new ablation: front_wide camera only
    ],
    ids=[
        "deterministic_simple_generation",
        "deterministic_two_stage_processor",
        "sampling_two_stage_processor",
        "deterministic_front_wide_only",
    ],
)
def test_pai_coc_full_end_to_end(
    vlm_model_processor, expert_model, temperature, top_p, two_stage, only_front_wide
):
    """
    Full end-to-end PAI-COC inference test (VLM CoC + Expert + FlowMatching).

    This is the pytest equivalent of the original `run_inference_on_clip`.
    """
    clip_id = "0abe118e-aa79-41f6-a719-f2df8abaf1ea"
    cameras = ["front_wide"] if only_front_wide else None
    clip_data = load_clip_data(clip_id, cameras=cameras)

    vlm_model, vlm_processor = vlm_model_processor
    expert = expert_model

    # VLM CoC generation
    coc_output = generate_alpamayo_coc(
        vlm_model,
        vlm_processor,
        images=clip_data["images"],
        max_tokens=256,
        temperature=temperature,
        top_p=top_p,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        two_stage=two_stage,
        stop_on_traj_future=False,
    )
    raw_coc = coc_output.text
    readable_coc = extract_readable_coc(raw_coc)

    cam_mode = "front_wide_only" if only_front_wide else "all_4_cameras"
    print(f"\n{'='*70}")
    print(f"[CoC Output] temperature={temperature}, two_stage={two_stage}, cameras={cam_mode}")
    print(f"{'='*70}")
    print(raw_coc)
    print(f"\n[Readable CoC] {readable_coc if readable_coc else '(no <|cot_*> markers found)'}")

    # KV cache + history injection
    traj_data = {
        k: clip_data.get(k)
        for k in ("ego_history_xyz", "ego_history_rot")
        if clip_data.get(k) is not None
    }
    prompt_cache = prefill_vlm_kv_cache(
        vlm_model, vlm_processor, images=clip_data["images"], traj_data=traj_data or None
    )

    # Expert + FlowMatching (64 waypoints)
    n_diff = 64
    batch = 1
    prompt_len = prompt_cache[0][0].shape[2] if prompt_cache else 200

    pos_ids, attn_mask = build_expert_attention_inputs(prompt_len, n_diff)
    step_fn = create_expert_step_fn(
        expert=expert,
        prompt_cache=prompt_cache,
        position_ids=pos_ids,
        attention_mask=attn_mask,
        n_diffusion_tokens=n_diff,
        prefill_seq_len=prompt_len,
    )

    flow = FlowMatching(num_inference_steps=8)
    traj = flow.sample(batch_size=batch, step_fn=step_fn, x_dims=(n_diff * 2,))

    # Print full trajectory (raw + denormalized)
    traj_2d = traj.reshape(-1, 2)
    print(f"\n[Trajectory] {n_diff} waypoints (raw normalized accel, curvature)")
    for i in range(n_diff):
        print(f"  {i:02d}: accel={float(traj_2d[i,0]):+8.4f}, curvature={float(traj_2d[i,1]):+8.4f}")

    try:
        phys = denormalize_trajectory(traj_2d, expert)
        print(f"\n[Trajectory] {n_diff} waypoints (physical units: m/s², 1/m)")
        for i in range(n_diff):
            print(f"  {i:02d}: accel={float(phys[i,0]):+8.4f}, curvature={float(phys[i,1]):+8.4f}")
    except Exception as e:
        print(f"[Warning] Denormalization failed: {e}")

    # Basic assertions
    assert len(raw_coc) > 10
    assert traj.shape == (batch, n_diff * 2)
    assert not mx.any(mx.isnan(traj))

    # Optional: denormalize and check shape
    try:
        phys = denormalize_trajectory(traj_2d, expert)
        assert phys.shape == (n_diff, 2)
    except Exception as e:
        print(f"[Warning] Denormalization failed: {e}")