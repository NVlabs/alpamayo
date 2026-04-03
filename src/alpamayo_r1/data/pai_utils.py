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

"""Local interface for loading PAI data from a local directory."""

import os
import json
import io
import pathlib
import zipfile
from typing import Any, Iterable
import pandas as pd
import numpy as np

from physical_ai_av import egomotion, video
from physical_ai_av.dataset import Features

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


class PhysicalAIAVDatasetLocalInterface:
    def __init__(
        self,
        local_dir: str | pathlib.Path,
        chunk_ids: list[int] | None = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        start_safe_margin_seconds: float = 1.6,
        end_safe_margin_seconds: float = 6.4,
    ) -> None:
        """Initialize the local PAI dataset interface.

        Args:
            local_dir: Path to the local directory containing the PAI dataset.
            chunk_ids: List of chunk IDs to load, or a range string (e.g. "0-9").
                      If None, all available chunks will be loaded.
        """
        self.local_dir = local_dir
        self.chunk_ids = None
        if chunk_ids is not None:
            if isinstance(chunk_ids, str) and "-" in chunk_ids:
                chunk_start = int(chunk_ids.split("-")[0])
                chunk_end = int(chunk_ids.split("-")[1])
                self.chunk_ids = list(range(chunk_start, chunk_end))
            elif isinstance(chunk_ids, (list, tuple, Iterable)):
                self.chunk_ids = list(chunk_ids)
            elif isinstance(chunk_ids, int):
                self.chunk_ids = [chunk_ids]
            else:
                logger.error(f"Invalid chunk_ids: {chunk_ids} {type(chunk_ids)}")
        else:
            logger.info("Loading all chunks")

        logger.info(f"Loading from {local_dir} with chunk_ids: {self.chunk_ids}")

        self.start_safe_margin_seconds = start_safe_margin_seconds
        self.end_safe_margin_seconds = end_safe_margin_seconds

        features_df = pd.read_csv(
            os.path.join(self.local_dir, features_metadata), index_col="feature"
        )
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(os.path.join(self.local_dir, clip_index_metadata))

        self.filter_clips_by_event_t0s()

        self.sensor_presence = pd.read_parquet(
            os.path.join(self.local_dir, "metadata/feature_presence.parquet")
        )
        self.chunk_sensor_presence = (
            pd.concat(
                [self.clip_index[["chunk"]], self.sensor_presence.select_dtypes(include=bool)],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )

    def filter_clips_by_event_t0s(self) -> None:
        """Filter clip_index by event_t0s: keep only events where
        event_t0 >= start_safe_margin_seconds (in us) and
        event_t0 + end_safe_margin_seconds (in us) <= end_timestamp.
        Remove rows whose event_t0s become empty after filtering.
        """
        if "event_t0s" not in self.clip_index.columns:
            return
        start_margin_us = int(self.start_safe_margin_seconds * 1_000_000)
        end_margin_us = int(self.end_safe_margin_seconds * 1_000_000)
        has_end = "end_timestamp" in self.clip_index.columns

        def filter_events(row: pd.Series) -> np.ndarray:
            et0s = row["event_t0s"]
            if et0s is None or (hasattr(et0s, "__len__") and len(et0s) == 0):
                return np.array([], dtype=np.int64)
            arr = np.asarray(et0s, dtype=np.int64)
            mask = arr >= start_margin_us
            if has_end:
                end_ts = int(row["end_timestamp"])
                mask &= (arr + end_margin_us) <= end_ts
            return arr[mask]

        self.clip_index["event_t0s"] = self.clip_index.apply(filter_events, axis=1)
        non_empty = self.clip_index["event_t0s"].apply(lambda x: x is not None and len(x) > 0)
        removed = (~non_empty).sum()
        self.clip_index = self.clip_index.loc[non_empty]
        if removed > 0:
            logger.info(
                "filter_clip_index_by_event_t0s: removed %d rows with empty event_t0s after filtering",
                removed,
            )

    def get_all_clip_ids(self):
        if self.chunk_ids is not None:
            return self.clip_index.loc[self.clip_index["chunk"].isin(self.chunk_ids)].index.tolist()
        else:
            return self.clip_index.index.tolist()

    def get_clip_chunk(self, clip_id: str) -> int:
        """Returns the chunk index for `clip_id`."""
        return self.clip_index.at[clip_id, "chunk"]

    def get_clip_key_frame(self, clip_id: str, sample_index_in_clip: int = 0) -> np.int64:
        t0 = self.clip_index.at[clip_id, "event_t0s"][sample_index_in_clip]
        return np.asarray(t0, dtype=np.int64)

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        if feature not in self.features.features_df.index:
            logger.warning(
                "Feature %r is not in features_df (available: %s). Returning None.",
                feature,
                list(self.features.features_df.index),
            )
            return None
        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        chunk_filename = os.path.join(self.local_dir, chunk_filename)
        with open(chunk_filename, "rb") as f:
            if chunk_filename.endswith(".parquet"):
                return pd.read_parquet(f).loc[clip_id]
            elif chunk_filename.endswith(".zip"):
                clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(f, "r") as zf:
                    if feature == "egomotion":
                        egomotion_df = pd.read_parquet(
                            io.BytesIO(zf.read(clip_files_in_zip["egomotion"]))
                        )
                        return egomotion.EgomotionState.from_egomotion_df(
                            egomotion_df
                        ).create_interpolator(egomotion_df["timestamp"].to_numpy())
                    elif feature.startswith("camera"):
                        return video.SeekVideoReader(
                            video_data=io.BytesIO(zf.read(clip_files_in_zip["video"])),
                            timestamps=pd.read_parquet(
                                io.BytesIO(zf.read(clip_files_in_zip["frame_timestamps"]))
                            )["timestamp"].to_numpy(),
                        )
                    else:
                        logger.warning(
                            f"Feature-specific data reader for {feature=} not implemented yet."
                        )
                        return {
                            k: pd.read_parquet(io.BytesIO(zf.read(v)))
                            if v.endswith(".parquet")
                            else io.BytesIO(zf.read(v))
                            for k, v in self.features.get_clip_files_in_zip(
                                clip_id, feature
                            ).items()
                        }
            else:
                raise ValueError(f"Unexpected file extension: {chunk_filename=}.")
