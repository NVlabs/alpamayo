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


"""Download the Physical AI AV dataset from Hugging Face.

example: for downloading 4camera + egomotion for AR1 finetuning
python scripts/download_pai.py --chunk-ids 0-2 \
         --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
         --calibration camera_intrinsics sensor_extrinsics --labels egomotion
"""

from __future__ import annotations
import argparse
from pathlib import Path

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MANDATORY_PATTERNS = [
    "features.csv",
    "clip_index.parquet",
    "metadata/**",
]
OPTIONAL_COMPONENTS = ("camera", "calibration", "labels", "lidar", "radar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Physical AI AV dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nvidia/PhysicalAI-Autonomous-Vehicles"),
        help="Local directory to store downloaded files.",
    )
    parser.add_argument(
        "--chunk-ids",
        type=str,
        default=None,
        help="Chunk IDs to download. Supports: single '0', multi '0 1', or range '0-3' (exclusive end, downloads 0,1,2). Downloads all if not specified.",
    )
    parser.add_argument(
        "--camera",
        nargs="+",
        default=None,
        help="Camera subparts, e.g. camera_front_wide_120fov camera_cross_left_120fov.",
    )
    parser.add_argument(
        "--calibration",
        nargs="+",
        default=None,
        help="Calibration subparts, e.g. camera_intrinsics sensor_extrinsics.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels subparts, e.g. egomotion.",
    )
    parser.add_argument(
        "--lidar",
        nargs="+",
        default=None,
        help="Lidar subparts, e.g. lidar_top_360fov.",
    )
    parser.add_argument(
        "--radar",
        nargs="+",
        default=None,
        help="Radar subparts, e.g. radar_front_center_mrr_2.",
    )
    args = parser.parse_args()
    return args


def parse_component_subparts(args: argparse.Namespace) -> list[tuple[str, str]]:
    component_pairs: list[tuple[str, str]] = []
    for component in OPTIONAL_COMPONENTS:
        subparts = getattr(args, component) or []
        for subpart in subparts:
            cleaned = subpart.strip().strip("/")
            if not cleaned:
                continue
            component_pairs.append((component, cleaned))
    return component_pairs


def build_allow_patterns(
    component_pairs: list[tuple[str, str]],
    chunk_ids: list[str] | None,
) -> list[str]:
    patterns: list[str] = list(MANDATORY_PATTERNS)

    normalized_chunks = [f"chunk_{int(chunk):04d}" for chunk in (chunk_ids or [])]

    for component, subpart in component_pairs:
        if normalized_chunks:
            for chunk in normalized_chunks:
                patterns.append(f"{component}/{subpart}/{subpart}.{chunk}.*")
        else:
            patterns.append(f"{component}/{subpart}/{subpart}.*")

    # De-duplicate while preserving order.
    return list(dict.fromkeys(patterns))


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.chunk_ids is None:
        args.chunk_ids = []
    elif " " in args.chunk_ids:
        args.chunk_ids = args.chunk_ids.split(" ")
    elif "-" in args.chunk_ids:
        start = int(args.chunk_ids.split("-")[0])
        end = int(args.chunk_ids.split("-")[1])
        args.chunk_ids = list(range(start, end))
    else:
        args.chunk_ids = [int(args.chunk_ids)]

    print("downloading chunks: ", args.chunk_ids if args.chunk_ids else "all")

    try:
        component_pairs = parse_component_subparts(args)
        allow_patterns = build_allow_patterns(component_pairs, args.chunk_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("download patterns", allow_patterns)

    downloaded_path = snapshot_download(
        repo_id=DEFAULT_REPO_ID,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    print(f"Downloaded dataset snapshot to: {downloaded_path}")
    print("Included mandatory patterns: " + ", ".join(MANDATORY_PATTERNS))


if __name__ == "__main__":
    main()
