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

from abc import ABC, abstractmethod
from typing import Mapping, MutableMapping
import torch

from alpamayo_r1.common import logging
from alpamayo_r1.metrics.metric_utils import apply_prefix
from alpamayo_r1.metrics import distance_metrics

# default vehicle size in meters
EGO_VEHICLE_LWH = (4.0, 3.0, 2.0)

logger = logging.RankedLogger(__name__, rank_zero_only=False)
logger.setLevel("INFO")


class Metric(ABC):
    """Base class for metrics, subclass and implement the evaluate function."""

    @abstractmethod
    def evaluate(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluates metric(s) on a batch of data.

        Args:
            model (BaseModel): the model being trained
            data_batch (dict[str,torch.Tensor]): batch of data
            output_batch (dict[str,torch.Tensor]): model outputs from model.training_step()
                or model.validation_step()

        Returns:
            per_sample_metrics (dict[str,torch.Tensor]): metrics computed per sample where each
                key is a metric name and the value is the metric value of size [B]
        """
        ...


class ReasoningSampler(Metric):
    """Helper metric for use in `alpamayo.callbacks.metric_callback.MetricRunnerCallback` which
    samples reasoning process from the model and adds them to output_batch for use in later metrics
    and vis.

    Does not compute any actual metric values.
    """

    def __init__(
        self,
        top_p: float = 0.98,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        prefix: str = "",
        max_generation_length: int = 256,
        traj_only_generation: bool = False,
        **kwargs,
    ) -> None:
        """Reasoning sampler module.

        Args:
            top_p (float, optional): top probability to sample. Defaults to 0.98.
            temperature (float, optional): sampling temperature. Defaults to 1.0.
            num_traj_samples (int, optional): number of trajectory samples per set. Defaults to 6.
            num_traj_sets (int, optional): number of trajectory sets, used to compute variance.
                Defaults to 1.
            prefix (str, optional): prefix to attach to the predictions which are added to the
                output_batch. Defaults to "".
            kwargs (Mapping[str, object], optional): additional arguments to pass to
                `model.sample_trajectories_from_data`.
        """
        super().__init__()
        self.top_p = top_p
        self.temperature = temperature
        self.num_traj_samples = num_traj_samples
        self.num_traj_sets = num_traj_sets
        self.prefix = prefix
        self.max_generation_length = max_generation_length
        self.kwargs = kwargs if kwargs is not None else {}

    def evaluate(
        self,
        model,
        data_batch: Mapping[str, torch.Tensor],
        output_batch: MutableMapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Generates predictions from the model and adds them to the output_batch.

        Outputs dict contains:
            pred_xyz: [B, N, K, Tf, 3] predicted trajectory
            pred_rot: [B, N, K, Tf, 3, 3] predicted rotations
            logprob: [B, N, K, Tf] log probabilities of predicted tokens
            cot: [B, ns, nj] predicted Cot with number of set (ns) and number of traj (nj)
            meta_action_string: [B, ns, nj] predicted meta action strings
            pred_answer: [B, ns, nj] predicted answers
        """
        pred_xyz, pred_rot = model.sample_trajectories_from_data(
            data=data_batch,
            num_traj_samples=self.num_traj_samples,
            num_traj_sets=self.num_traj_sets,
            top_p=self.top_p,
            temperature=self.temperature,
            traj_only_generation=False,
            max_generation_length=self.max_generation_length,
            return_extra=False,
            **self.kwargs,
        )

        # dict used for later metrics
        output_batch.update(
            apply_prefix(
                self.prefix,
                {
                    "pred_xyz": pred_xyz,
                    "pred_rot": pred_rot,
                },
            )
        )

        # dict for data input (no need to add prefix)
        output_batch.update(
            {
                "absolute_timestamps": data_batch.get("absolute_timestamps", None),
                "relative_timestamps": data_batch.get("relative_timestamps", None),
                "ego_history_xyz": data_batch.get("ego_history_xyz", None),
                "ego_history_rot": data_batch.get("ego_history_rot", None),
                "ego_future_xyz": data_batch.get("ego_future_xyz", None),
                "ego_future_rot": data_batch.get("ego_future_rot", None),
            }
        )
        return {}


class DistanceMetrics(Metric):
    """Computes distance metrics for `alpamayo.callbacks.metric_callback.MetricRunnerCallback`."""

    def __init__(
        self,
        prefix: str = "",
        time_step: float = 0.1,
    ):
        """Args:
        prefix (str, optional): prefix of trajectory samples to use. Computes metrics on
            trajectories in output_batch with keys f"{prefix}pred_rot"
            Defaults to "".
        group_by_scenarios (list[str], optional): Defaults to None.
            if not None, then the metrics will be grouped by the scenario names in this list.
        """
        self.prefix = prefix
        self.time_step = time_step

    def evaluate(
        self,
        model: torch.nn.Module | None,
        data_batch: Mapping[str, torch.Tensor],
        output_batch: MutableMapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Computes distance metrics (corner distance, minADE, ADE)

        Assumes TrajSampler has been run already, and trajectory samples are
        in the output_batch with keys:
            (prefix)+pred_xyz: [B, N, K, Tf, 3] predicted trajectory
            (prefix)+pred_rot [B, N, K, Tf, 3, 3] predicted rotations
            (prefix)+logprob: [B, N, K, Tf] log probabilities of predicted tokens

        Returns dict[str,Tensor]:
            # distance metrics
            min_ade: [B] average min_ade (min over K, average over N)
            corner_distance: [B] average min corner distance (min over K, average over N)

            # if N > 1, we have the following extra data item:
            # for each value above, we collect the statistics _sq and _std, for example:
            min_ade_sq: [B] average min_ade^2 (min over K, average over N)
            min_ade_std: [B] std of min_ade
        """
        del model  # unused

        if self.prefix + "pred_xyz" not in output_batch:
            logger.warning(f"No predictions with prefix {self.prefix} found in output_batch.")
            return {}

        pred_xyz = output_batch[self.prefix + "pred_xyz"]
        pred_rot = output_batch[self.prefix + "pred_rot"]

        num_traj_sets = pred_xyz.shape[1]

        if data_batch["ego_future_xyz"].shape[1] > 1:
            logger.info("Multiple traj group provided, only evaluating the last one.")
        gt_xyz = data_batch["ego_future_xyz"][:, -1]
        gt_rot = data_batch["ego_future_rot"][:, -1]

        # TODO: move this to a config
        timestep_horizons_in_seconds = [0.5, 1, 3, 5]
        timestep_horizons = [int(t / self.time_step) for t in timestep_horizons_in_seconds]

        metric_dict = distance_metrics.compute_minade(
            pred_xyz,
            gt_xyz,
            disable_summary=(num_traj_sets == 1),
            timestep_horizons=timestep_horizons,
            time_step=self.time_step,
        )

        # compute per-sample ADE for later visualization
        sample_ade = distance_metrics.compute_ade(pred_xyz, gt_xyz)
        output_batch.update({self.prefix + "sample_ade": sample_ade})

        # dummy logprob for now
        logprob = torch.zeros_like(pred_xyz[..., 0])

        # compute ADE, select pred_xyz of highest logprob
        # logprob: [B, N, K, Tf]
        idx = logprob.sum(dim=-1).argmax(dim=-1)  # [B, N]
        top_xyz = torch.take_along_dim(pred_xyz, idx[..., None, None, None], dim=2)
        ade = distance_metrics.compute_ade(top_xyz, gt_xyz).squeeze(2).mean(-1)  # [B]
        metric_dict.update({"ade": ade})
        # shape [B]
        timestep_horizon = int(3.0 / self.time_step)
        if timestep_horizon <= pred_xyz.shape[3]:
            ade_3s = (
                distance_metrics.compute_ade(top_xyz, gt_xyz, timestep_horizon=timestep_horizon)
                .squeeze(2)
                .mean(-1)
            )
            metric_dict.update({"ade/by_t=3.0": ade_3s})

        metric_dict.update(
            distance_metrics.compute_grouped_corner_distance(
                pred_xyz,
                pred_rot,
                gt_xyz,
                gt_rot,
                # TODO: change this to true ego vehicle size
                torch.tensor(EGO_VEHICLE_LWH, dtype=torch.float32, device=gt_xyz.device),
                disable_summary=(num_traj_sets == 1),
            )
        )

        return apply_prefix(self.prefix, metric_dict)
