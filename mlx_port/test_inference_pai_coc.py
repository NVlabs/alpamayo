"""
MLX port of src/alpamayo_r1/test_inference.py

End-to-end inference on a real PAI-COC clip:
- Load multi-camera frames + ego history from /Volumes/MicronSSD/pai_coc
- Run VLM for CoC + discrete trajectory tokens
- Run Expert + FlowMatching for continuous trajectory
- Print CoC and predicted future trajectory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx
from PIL import Image

# MLX port modules
from mlx_port.vlm_loader import (
    generate_alpamayo_coc,
    extract_readable_coc,
    prefill_vlm_kv_cache,
    MIN_PIXELS,
    MAX_PIXELS,
)
from mlx_port.expert import denormalize_trajectory
from mlx_port.load_alpamayo_vlm import load_alpamayo_vlm_weights
from mlx_port.expert_loader import load_alpamayo_expert_weights
from mlx_port.expert import Expert, create_expert_step_fn, build_expert_attention_inputs
from mlx_port.flow_matching import FlowMatching

# Try to import the official loader (requires physical_ai_av package)
try:
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper as nvidia_helper
    HAS_PHYSICAL_AI_AV = True
except ImportError:
    HAS_PHYSICAL_AI_AV = False
    print("[Warning] physical_ai_av not installed. Using placeholder data loader.")


DATA_ROOT = Path("/Volumes/MicronSSD/pai_coc")

def get_coc_clip_ids(downloaded_only: bool = True) -> list[str]:
    """
    Return CoC clip IDs.

    Args:
        downloaded_only: If True (default), return only the 370 CoC clips whose
                         camera chunk data has been downloaded locally (chunks 0-249).
                         If False, return all 1,740 CoC clips from the reasoning parquet.
    """
    reasoning_path = DATA_ROOT / "reasoning" / "ood_reasoning.parquet"
    if not reasoning_path.exists():
        raise FileNotFoundError(f"CoC reasoning file not found: {reasoning_path}")

    import pandas as pd
    coc_df = pd.read_parquet(reasoning_path)
    all_coc_ids = coc_df.index.unique().tolist()

    if not downloaded_only:
        return all_coc_ids

    # Compute intersection with actually downloaded chunks
    clip_index = pd.read_parquet(DATA_ROOT / "clip_index.parquet")
    camera_dir = DATA_ROOT / "camera" / "camera_front_wide_120fov"
    downloaded_chunks = set()
    for f in camera_dir.glob("*.zip"):
        name = f.name
        if "chunk_" in name:
            chunk_str = name.split("chunk_")[1].split(".zip")[0]
            downloaded_chunks.add(int(chunk_str))

    clip_index_reset = clip_index.reset_index()
    coc_in_index = clip_index_reset[clip_index_reset["clip_id"].isin(all_coc_ids)]
    downloadable = coc_in_index[coc_in_index["chunk"].isin(downloaded_chunks)]

    return downloadable["clip_id"].tolist()


def _get_chunk_for_clip(clip_id: str) -> int:
    """Look up which chunk a clip belongs to using the local clip_index.parquet."""
    import pandas as pd
    df = pd.read_parquet(DATA_ROOT / "clip_index.parquet")
    return int(df.loc[clip_id, "chunk"])


def _load_images_from_local_zips(
    clip_id: str,
    chunk: int,
    t0_us: int = 5_100_000,
    time_step: float = 0.1,
    num_frames_per_cam: int = 4,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    """
    Extract real camera frames + timestamps for a clip from the downloaded chunk zips,
    using timestamp synchronization exactly as in NVIDIA's load_physical_aiavdataset.

    Returns:
        images: list of PIL images (N_cameras * num_frames)
        absolute_timestamps: (N_cameras, num_frames) int64 microseconds
        relative_timestamps: (N_cameras, num_frames) float32 seconds (relative to camera min)
    """
    import zipfile
    import tempfile
    import os
    import cv2
    import numpy as np
    import pandas as pd   # needed for reading the .timestamps.parquet files

    camera_slugs = [
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    ]

    dt_us = int(time_step * 1_000_000)
    # Desired timestamps for the 4 frames: t0-0.3s, t0-0.2s, t0-0.1s, t0
    image_timestamps = np.array(
        [t0_us - (num_frames_per_cam - 1 - i) * dt_us for i in range(num_frames_per_cam)],
        dtype=np.int64,
    )

    images: list[Image.Image] = []
    abs_timestamps_per_cam: list[list[int]] = []

    for cam in camera_slugs:
        zip_path = DATA_ROOT / "camera" / cam / f"{cam}.chunk_{chunk:04d}.zip"
        if not zip_path.exists():
            print(f"[Warning] Camera zip not found: {zip_path}")
            continue

        mp4_name = f"{clip_id}.{cam}.mp4"
        ts_name = f"{clip_id}.{cam}.timestamps.parquet"

        with zipfile.ZipFile(zip_path, "r") as zf:
            if mp4_name not in zf.namelist() or ts_name not in zf.namelist():
                print(f"[Warning] Missing mp4 or timestamps for {cam} in {zip_path.name}")
                continue

            # Load timestamp array
            try:
                ts_df = pd.read_parquet(zf.open(ts_name))
                frame_timestamps = ts_df["timestamp"].to_numpy(dtype=np.int64)
            except Exception as e:
                print(f"[Warning] Failed to read timestamps parquet for {cam}: {e}")
                continue

            # Find closest frame index for each desired timestamp
            indices = []
            selected_ts = []
            for desired in image_timestamps:
                idx = int(np.argmin(np.abs(frame_timestamps - desired)))
                indices.append(idx)
                selected_ts.append(int(frame_timestamps[idx]))
            abs_timestamps_per_cam.append(selected_ts)

            # Extract the selected frames from the mp4
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(zf.read(mp4_name))
                tmp_path = tmp.name

            try:
                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                for idx in indices:
                    if idx >= total_frames:
                        idx = total_frames - 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame_bgr = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        images.append(Image.fromarray(frame_rgb))
                cap.release()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # Build arrays (N_cameras, num_frames) - numpy to avoid torch dependency
    if len(abs_timestamps_per_cam) == 0:
        # No camera succeeded – return empty arrays so caller can fall back
        return images, np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    abs_ts = np.array(abs_timestamps_per_cam, dtype=np.int64)
    # Relative timestamps in seconds, exactly as NVIDIA
    camera_tmin = abs_ts.min()
    rel_ts = (abs_ts - camera_tmin).astype(np.float32) * 1e-6

    return images, abs_ts, rel_ts


def _load_egomotion_history(clip_id: str, chunk: int, t0_us: int = 5_100_000,
                            num_history_steps: int = 16) -> tuple[Any, Any]:
    """
    Load ego history (xyz + rotation matrix) from the local egomotion parquet
    for the given clip. Returns arrays shaped like the NVIDIA loader:
        ego_history_xyz: (1, 1, num_history_steps, 3)
        ego_history_rot: (1, 1, num_history_steps, 3, 3)
    The history is taken from the rows immediately preceding t0_us.
    (Torch-free implementation so the MLX-only venv can still use real data.)
    """
    import zipfile
    import pandas as pd
    from scipy.spatial.transform import Rotation as R

    zip_path = DATA_ROOT / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
    if not zip_path.exists():
        return None, None

    parquet_name = f"{clip_id}.egomotion.parquet"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if parquet_name not in zf.namelist():
                return None, None
            df = pd.read_parquet(zf.open(parquet_name))
    except Exception:
        return None, None

    if "timestamp" not in df.columns:
        return None, None

    # Find the row closest to t0_us
    df = df.sort_values("timestamp")
    idx = (df["timestamp"] - t0_us).abs().idxmin()
    pos = df.index.get_loc(idx)

    start = max(0, pos - num_history_steps + 1)
    hist = df.iloc[start : pos + 1].tail(num_history_steps)

    if len(hist) == 0:
        return None, None

    xyz = hist[["x", "y", "z"]].to_numpy(dtype=np.float32)          # (T, 3)
    quats = hist[["qx", "qy", "qz", "qw"]].to_numpy()
    rot_mats = np.stack([R.from_quat(q).as_matrix() for q in quats], axis=0)  # (T, 3, 3)

    T = xyz.shape[0]
    if T < num_history_steps:
        pad = num_history_steps - T
        xyz = np.pad(xyz, ((pad, 0), (0, 0)), mode="edge")
        rot_mats = np.pad(rot_mats, ((pad, 0), (0, 0), (0, 0)), mode="edge")

    # Return numpy arrays with the expected leading (1,1,...) batch dims
    xyz_arr = xyz.reshape(1, 1, -1, 3)
    rot_arr = rot_mats.reshape(1, 1, -1, 3, 3)
    return xyz_arr, rot_arr


def load_clip_data(clip_id: str, t0_us: int = 5_100_000) -> dict[str, Any]:
    """
    Load a PAI-COC clip and return images + ego history in the format expected
    by the MLX Alpamayo pipeline.
    """
    if HAS_PHYSICAL_AI_AV:
        data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
        # Convert torch tensors to PIL images for the VLM
        # data["image_frames"] is (N_cameras, num_frames, C, H, W)
        images = []
        for cam in range(data["image_frames"].shape[0]):
            for f in range(data["image_frames"].shape[1]):
                img = data["image_frames"][cam, f].permute(1, 2, 0).numpy()
                images.append(Image.fromarray(img.astype(np.uint8)))

        return {
            "images": images,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
            "clip_id": clip_id,
        }
    else:
        # Direct local loader using the downloaded chunk zips (no physical_ai_av needed)
        print(f"[Info] Loading real multi-camera frames for CoC clip {clip_id} from local zips...")
        try:
            chunk = _get_chunk_for_clip(clip_id)
            images, abs_ts, rel_ts = _load_images_from_local_zips(
                clip_id, chunk, t0_us=t0_us, num_frames_per_cam=4
            )
            if len(images) == 0:
                raise RuntimeError("No frames could be extracted from any camera zip")
            print(f"[Info] Successfully loaded {len(images)} real frames "
                  f"({len(images)//4} cameras × 4 frames) from chunk {chunk}.")
            ego_xyz, ego_rot = _load_egomotion_history(clip_id, chunk, t0_us)
            if ego_xyz is not None:
                print(f"[Info] Loaded ego history ({ego_xyz.shape[2]} steps) from egomotion parquet.")
            return {
                "images": images,
                "ego_history_xyz": ego_xyz,
                "ego_history_rot": ego_rot,
                "absolute_timestamps": abs_ts,
                "relative_timestamps": rel_ts,
                "clip_id": clip_id,
            }
        except Exception as e:
            print(f"[Warning] Local zip loader failed ({e}). Falling back to placeholder image.")
            test_img = Image.open("reports/stage1_test_image_front_wide.png").convert("RGB")
            return {
                "images": [test_img] * 4,
                "ego_history_xyz": None,
                "ego_history_rot": None,
                "clip_id": clip_id,
            }


def run_inference_on_clip(clip_id: str):
    print(f"\n=== MLX Alpamayo Inference on PAI-COC clip: {clip_id} ===")

    # 1. Load VLM + Expert
    vlm_model, vlm_processor = load_alpamayo_vlm_weights()
    expert_state = load_alpamayo_expert_weights()
    # Cast expert weights to bfloat16 for inference parity
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

    # 2. Load clip data
    clip_data = load_clip_data(clip_id)

    # Clean single run (use defaults for production)
    coc_output = generate_alpamayo_coc(
        vlm_model,
        vlm_processor,
        images=clip_data["images"],
        max_tokens=256,
        temperature=0.6,
        top_p=0.98,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        two_stage=True,
        stop_on_traj_future=False,
    )
    raw_coc = coc_output.text
    readable_coc = extract_readable_coc(raw_coc)
    print("\n[CoC Output]\n", raw_coc)
    print("\n[Readable CoC]\n", readable_coc if readable_coc != raw_coc else "(no <|cot_*> markers found in output)")

    # 4. Get real KV cache from the same prompt (now with numeric history injection)
    traj_data = {
        k: clip_data.get(k)
        for k in ("ego_history_xyz", "ego_history_rot")
        if clip_data.get(k) is not None
    }
    if traj_data:
        print(f"[Info] Ego history available for injection: xyz shape {getattr(traj_data.get('ego_history_xyz'), 'shape', None)}")
    # Request exactly 16 tokens per history trajectory (tokens_per_history_traj from Alpamayo config)
    prompt_cache = prefill_vlm_kv_cache(
        vlm_model, vlm_processor, images=clip_data["images"], traj_data=traj_data or None
    )

    # 5. Run Expert + FlowMatching
    #    NVIDIA produces 64 future waypoints. We match that here.
    #    The VLM prompt now receives numeric ego_history_xyz / ego_history_rot
    #    via the injected discrete <iN> tokens (MLX equivalent of fuse_traj_tokens).
    n_diff = 64
    batch = 1
    prompt_len = prompt_cache[0][0].shape[2] if prompt_cache else 200

    past_kv = prompt_cache  # already in correct format

    pos_ids, attn_mask = build_expert_attention_inputs(prompt_len, n_diff)

    step_fn = create_expert_step_fn(
        expert=expert,
        prompt_cache=past_kv,
        position_ids=pos_ids,
        attention_mask=attn_mask,
        n_diffusion_tokens=n_diff,
        prefill_seq_len=prompt_len,
    )

    flow = FlowMatching(num_inference_steps=8)
    traj = flow.sample(batch_size=batch, step_fn=step_fn, x_dims=(n_diff * 2,))

    print(f"\n[Predicted Trajectory – all {n_diff} waypoints (raw normalized values)]")
    # Ensure we have a (n_waypoints, 2) tensor for both printing and denormalization
    traj_2d = traj.reshape(-1, 2)
    for i in range(n_diff):
        print(f"  waypoint {i:02d}: accel={float(traj_2d[i,0]):+7.4f}, curvature={float(traj_2d[i,1]):+7.4f}")

    # Denormalize to physical units (m/s² and 1/m) using the expert's action_space stats
    try:
        phys = denormalize_trajectory(traj_2d, expert)
        phys_reshaped = phys.reshape(n_diff, 2)
        print(f"\n[Predicted Trajectory – 64 waypoints (physical units: accel m/s², curvature 1/m)]")
        for i in range(n_diff):
            print(f"  waypoint {i:02d}: accel={float(phys_reshaped[i,0]):+8.4f}, curvature={float(phys_reshaped[i,1]):+8.4f}")
    except Exception as e:
        print(f"[Warning] Could not denormalize trajectory: {e}")

    print("\n=== Inference complete ===")
    return coc_output.text, traj


if __name__ == "__main__":
    # Use a clip from the *downloaded subset* of the CoC dataset.
    # Only 370 of the 1,740 CoC clips have their camera data (chunks 0-249) locally available.
    coc_clips = get_coc_clip_ids(downloaded_only=True)
    clip_id = coc_clips[0]  # first downloadable CoC clip: 0abe118e-aa79-41f6-a719-f2df8abaf1ea
    print(f"Using downloadable CoC clip: {clip_id}")
    print(f"Total downloadable CoC clips: {len(coc_clips)} / 1740")
    run_inference_on_clip(clip_id)