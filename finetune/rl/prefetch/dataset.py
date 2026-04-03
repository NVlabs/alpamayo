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

from __future__ import annotations

from typing import Any, Literal

from rl.prefetch.server import _fetch_sample, init_dataset_prefetch

SampleRole = Literal["raw", "policy", "rollout"]


class NodePrefetchDatasetWrapper:
    """Wrap a map-style dataset so reads are served by the node prefetch server.

    Intended usage: replace `alp_state.get_dataloaders()[split].dataset` with this wrapper
    when TOML enables prefetch. Then the existing packer path `dataset[n]` is routed through
    the node server, and `get_prefetched(n, role)` enables role-specific fetch.
    """

    def __init__(self, base: Any, *, server_key: str = "default"):
        self._base = base
        self._server_key = str(server_key or "default")
        # Ensure server exists (no-op if prefetch.capacity <= 0).
        init_dataset_prefetch(dataset=self._base, server_key=self._server_key)

    @property
    def base(self) -> Any:
        return self._base

    @property
    def server_key(self) -> str:
        return self._server_key

    def __len__(self) -> int:
        return int(len(self._base))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def get_prefetched(self, *, n: int, role: SampleRole) -> Any:
        sample, _mapped_idx = _fetch_sample(n=int(n), dataset=self._base, role=str(role))
        return sample

    def __getitem__(self, idx: int) -> Any:
        # Keep raw semantics for any code that still calls dataset[idx] directly.
        return self.get_prefetched(n=int(idx), role="raw")
