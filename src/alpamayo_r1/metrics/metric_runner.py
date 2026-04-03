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

from alpamayo_r1.metrics.metric_api import Metric


class MetricRunner:
    """
    Runner for metrics.
    This class is used to hold all of the metric instances and run them one-by-one on a batch of data.
    Each metric value will be prefixed with "metric/" and added to the output batch.
    """

    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics

    def run(
        self, model: Any, data_batch: dict[str, Any], output_batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the metrics one-by-one by the order of the metrics list."""
        per_sample_metrics = {}
        for metric in self.metrics:
            per_sample = metric.evaluate(model, data_batch, output_batch)
            per_sample_metrics.update(per_sample)

        output_batch.update({"metric/" + k: v for k, v in per_sample_metrics.items()})
