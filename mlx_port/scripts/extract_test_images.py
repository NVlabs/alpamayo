"""Extract the 4 camera images used in the end-to-end inference test.

Saves one representative frame (the current observation at t0) per camera
to mlx_port/data/ as PNG files.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


CLIP_ID = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def main():
    print(f"Loading dataset for clip_id: {CLIP_ID}...")
    data = load_physical_aiavdataset(
        CLIP_ID,
        t0_us=5_100_000,
        maybe_stream=True,
    )
    print("Dataset loaded.")

    image_frames = data["image_frames"]  # (N_cameras, num_frames, 3, H, W)
    camera_indices = data.get("camera_indices", list(range(image_frames.shape[0])))

    print(f"image_frames shape: {image_frames.shape}")
    print(f"camera_indices: {camera_indices}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert to numpy once
    image_frames_np = image_frames.numpy() if hasattr(image_frames, "numpy") else np.array(image_frames)

    # Save the last frame of each camera (the current observation at t0)
    for cam_idx, cam_id in enumerate(camera_indices):
        # Last frame of this camera's sequence
        frame = image_frames_np[cam_idx, -1]  # (3, H, W)

        # Convert to (H, W, C) and uint8
        frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Camera name mapping (best effort)
        cam_name = {
            0: "cross_left_120fov",
            1: "front_wide_120fov",
            2: "cross_right_120fov",
            3: "front_tele_30fov",
        }.get(cam_id, f"camera_{cam_id}")

        filename = OUTPUT_DIR / f"test_image_cam{cam_idx}_{cam_name}.png"
        Image.fromarray(frame).save(filename)
        print(f"Saved: {filename}")

    print(f"\nAll 4 camera images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()