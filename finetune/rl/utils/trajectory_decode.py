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

"""Trajectory decoding helpers for Alpamayo Cosmos-RL."""

from __future__ import annotations

import torch

from alpamayo_r1.models.base_model import SPECIAL_TOKENS
from alpamayo_r1.models.token_utils import extract_traj_tokens


def decode_rollout_trajectory(
    to_be_evaluated: str,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    *,
    tokenizer,
    traj_tokenizer,
    model_config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode predicted future trajectory from generated completion text.

    Passing `tokenizer`/`traj_tokenizer` does NOT copy them in-process (cheap ref pass).
    """
    # Extract inner text between traj start/end markers
    traj_text = to_be_evaluated.rsplit(SPECIAL_TOKENS["traj_future_start"], maxsplit=1)[-1].split(
        SPECIAL_TOKENS["traj_future_end"], maxsplit=1
    )[0]

    token_id_offset = model_config.traj_token_start_idx
    expected_len = model_config.tokens_per_future_traj

    # If the model stops right after emitting `<|traj_future_start|>`, the inner segment is empty.
    # Fall back to an all-zero token sequence so downstream decode/reward doesn't crash.
    if traj_text.strip() == "":
        traj_token_ids = torch.zeros(
            (1, expected_len), dtype=torch.long, device=ego_history_xyz.device
        )
    else:
        generated_tokens = tokenizer(traj_text)

        special_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()
        }
        generated_tokens_t = torch.tensor(generated_tokens["input_ids"]).unsqueeze(0)

        traj_token_ids = extract_traj_tokens(
            generated_tokens_t,
            special_token_ids,
            expected_len,
            token_id_offset,
            traj_tokenizer.vocab_size,
        )

    # Align batch dimensions
    ego_hist_xyz = ego_history_xyz
    ego_hist_rot = ego_history_rot
    B = min(ego_hist_xyz.shape[0], traj_token_ids.shape[0])
    if B != traj_token_ids.shape[0]:
        traj_token_ids = traj_token_ids[:B]
    if B != ego_hist_xyz.shape[0]:
        ego_hist_xyz = ego_hist_xyz[:B]
        ego_hist_rot = ego_hist_rot[:B]

    predicted_fut_xyz, predicted_fut_rot, _ = traj_tokenizer.decode(
        hist_xyz=ego_hist_xyz,
        hist_rot=ego_hist_rot,
        tokens=traj_token_ids,
    )
    return predicted_fut_xyz, predicted_fut_rot
