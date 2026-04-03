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

from typing import Any

import torch
from alpamayo_r1.data.pai_utils import PhysicalAIAVDatasetLocalInterface
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset


class PAIDataset(Dataset):
    """Dataset for loading and processing Alpamayo samples."""

    DEFAULT_T0_US = 5_100_000

    def __init__(
        self,
        local_dir: str,
        chunk_ids: list[int] | None = None,
        include_extr_intr: bool = False,
        reshape_tensors_for_rl: bool = False,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        use_default_keyframe: bool = False,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
    ):
        """Initialize dataset.

        Args:
            local_dir: Path to the local directory containing the PAI dataset.
            chunk_ids: List of chunk IDs to load, or a range string (e.g. "0-9").
                      If None, all available chunks will be loaded.
            include_extr_intr: Whether to include extrinsics, intrinsics,
                and vehicle dimensions in samples.
            reshape_tensors_for_rl: If True, reshape cameras/timestamps and squeeze ego tensors
                for RL training.
            model_config: Optional config (dict is converted with OmegaConf) passed as
                ``model_config`` into ``hydra.utils.instantiate`` when building the preprocessor.
            vla_preprocess_args: If set, Hydra config dict to instantiate a preprocessor; its result
                is stored as ``tokenized_data`` on each sample.
            use_default_keyframe: If True, clip time is ``DEFAULT_T0_US``; else use the per-clip
                keyframe from the dataset index.
            features_metadata: Filename under ``local_dir`` for the features CSV.
            clip_index_metadata: Filename under ``local_dir`` for the clip index parquet.
            num_history_steps: History length for ``load_physical_aiavdataset``.
            num_future_steps: Future horizon for ``load_physical_aiavdataset``.
            time_step: Seconds per step between trajectory samples.
        """
        self.avdi = PhysicalAIAVDatasetLocalInterface(
            local_dir=local_dir,
            chunk_ids=chunk_ids,
            features_metadata=features_metadata,
            clip_index_metadata=clip_index_metadata,
        )
        self.clip_ids = self.avdi.get_all_clip_ids()
        self.include_extr_intr = include_extr_intr
        self.use_default_keyframe = use_default_keyframe
        self.reshape_tensors_for_rl = reshape_tensors_for_rl

        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step

        self.vla_preprocess_func = None
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    def __len__(self) -> int:
        """Return the number of clips in the dataset."""
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """Load and process a single sample.

        Returns:
            Dictionary with processed inputs for the model, or None if loading fails
        """
        clip_id = self.clip_ids[idx]
        t0_us = (
            self.DEFAULT_T0_US
            if self.use_default_keyframe
            else self.avdi.get_clip_key_frame(clip_id)
        )

        sample_data = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        # squeeze ego motion shape
        for key in sample_data.keys():
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        if self.include_extr_intr:
            sample_data["extr"] = self.avdi.get_clip_feature(clip_id, "sensor_extrinsics")
            sample_data["intr"] = self.avdi.get_clip_feature(clip_id, "camera_intrinsics")

            vehicle_dimensions = self.avdi.get_clip_feature(clip_id, "vehicle_dimensions")
            sample_data["ego_lwh"] = torch.tensor(
                [vehicle_dimensions.length, vehicle_dimensions.width, vehicle_dimensions.height]
            )
            sample_data["ego_length_offset"] = torch.tensor(
                vehicle_dimensions.rear_axle_to_bbox_center / vehicle_dimensions.length
            )

        if self.reshape_tensors_for_rl:
            image_frames = sample_data["image_frames"]
            camera_indices = sample_data["camera_indices"]
            absolute_timestamps = sample_data["absolute_timestamps"]
            relative_timestamps = sample_data["relative_timestamps"]

            n_cam, n_frame = image_frames.shape[0], image_frames.shape[1]
            # [N, G, C, H, W]
            image_frames = image_frames.reshape(n_cam * n_frame, *image_frames.shape[2:]).unsqueeze(
                1
            )
            camera_indices = camera_indices.repeat_interleave(n_frame)
            absolute_timestamps = absolute_timestamps.reshape(-1)
            relative_timestamps = relative_timestamps.reshape(-1)

            sample_data["image_frames"] = image_frames
            sample_data["camera_indices"] = camera_indices
            sample_data["absolute_timestamps"] = absolute_timestamps
            sample_data["relative_timestamps"] = relative_timestamps

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
