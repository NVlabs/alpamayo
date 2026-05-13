"""
Extract a random frame from camera_front_wide_120fov in the PAI-COC dataset.

This replaces the previous clip-specific extraction with a truly random frame
from the data stored in /Volumes/MicronSSD/pai_coc/.

Usage (on host with the volume mounted):

    python mlx_port/extract_random_front_wide.py

Output:
    reports/stage1_test_image_front_wide.png
"""

import random
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = Path("/Volumes/MicronSSD/pai_coc")
CAMERA = "camera_front_wide_120fov"
OUTPUT_PATH = Path("reports/stage1_test_image_front_wide.png")


def main():
    print("Loading clip index...")
    index = pd.read_parquet(DATA_ROOT / "clip_index.parquet")
    valid_clips = index[index.clip_is_valid].reset_index()

    # Pick a random valid clip
    row = valid_clips.sample(1, random_state=None).iloc[0]
    clip_id = row["clip_id"]
    chunk = int(row["chunk"])

    print(f"Selected random clip: {clip_id} (chunk {chunk})")

    zip_path = DATA_ROOT / "camera" / CAMERA / f"{CAMERA}.chunk_{chunk:04d}.zip"
    if not zip_path.exists():
        print(f"Zip not found: {zip_path}")
        return

    print(f"Opening zip: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        mp4_name = f"{clip_id}.{CAMERA}.mp4"
        if mp4_name not in z.namelist():
            print(f"MP4 not found inside zip: {mp4_name}")
            return

        # Extract MP4 to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(z.read(mp4_name))
            tmp_path = Path(tmp.name)

    print(f"Extracted MP4 to temp: {tmp_path}")

    # Open video and pick a random frame
    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        print("Failed to open video")
        tmp_path.unlink(missing_ok=True)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Video has no frames")
        cap.release()
        tmp_path.unlink(missing_ok=True)
        return

    frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    tmp_path.unlink(missing_ok=True)

    if not ret:
        print("Failed to read frame")
        return

    # Convert BGR (OpenCV) -> RGB and save
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH)

    print(f"\nSaved random front-wide frame to: {OUTPUT_PATH.resolve()}")
    print(f"Frame index in video: {frame_idx} / {total_frames}")
    print(f"Image size: {img.size}")


if __name__ == "__main__":
    main()