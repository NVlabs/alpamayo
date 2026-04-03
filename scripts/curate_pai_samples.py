# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Curate target number of samples from Physical AI dataset so that you don't need to download the entire dataset.

example:
python scripts/curate_pai_samples.py \
  --clip-index-path /path/to/PAI_datset/clip_index.parquet \
  --chunk 3116-3119 --num-samples 16 \
  --output-path /path/to/PAI_datset/clip_index_3116_mini.parquet
"""

import pandas as pd
import argparse

DEFAULT_CLIP_INDEX_PATH = "dataset/alpamayo/PAI_mini/clip_index.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate PAI samples")
    parser.add_argument(
        "--clip-index-path",
        "-p",
        type=str,
        default=DEFAULT_CLIP_INDEX_PATH,
        help="Path to clip index parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk",
        "-c",
        type=str,
        required=True,
        help="chunk_id(s): single (e.g. 3116), space-separated (e.g. '3116 3117'), or range (e.g. 3116-3119)",
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, required=True, help="Number of samples to curate"
    )
    parser.add_argument(
        "--output-path", "-o", type=str, required=True, help="Output path for the curated parquet"
    )
    return parser.parse_args()


def _parse_chunk_ids(chunk_arg: str) -> list[str]:
    """Parse --chunk like download_pai: single, space-separated, or start-end range."""
    if " " in chunk_arg:
        return [s.strip() for s in chunk_arg.split(" ") if s.strip()]
    if "-" in chunk_arg:
        start_s, end_s = chunk_arg.split("-", 1)
        start, end = int(start_s.strip()), int(end_s.strip())
        return [str(i) for i in range(start, end)]
    return [chunk_arg.strip()]


def main():
    args = parse_args()
    chunk_ids = _parse_chunk_ids(args.chunk)
    print(f"Curate samples from chunk IDs: {chunk_ids}")
    chunk_set = set(chunk_ids)

    clip_index = pd.read_parquet(args.clip_index_path)
    col = "chunk_id" if "chunk_id" in clip_index.columns else "chunk"
    chunk_subset = clip_index[clip_index[col].astype(str).isin(chunk_set)]
    if len(chunk_subset) == 0:
        raise SystemExit(
            f"No rows with {col} in {sorted(chunk_set)}. "
            f"Available chunks (sample): {clip_index[col].dropna().unique()[:10].tolist()}"
        )
    n = min(args.num_samples, len(chunk_subset))
    if n < args.num_samples:
        print(f"Warning: only {len(chunk_subset)} rows for chunk(s) {chunk_ids}, sampling {n}.")
    curated_clip_index = chunk_subset.sample(n=n, random_state=11)
    curated_clip_index.to_parquet(args.output_path)


if __name__ == "__main__":
    main()
