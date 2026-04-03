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

import numpy as np
import torch
from transformers import AutoTokenizer

from alpamayo_r1.models.base_model import SPECIAL_TOKENS


def fill_masks_between_special_tokens(
    start_token: str,
    end_token: str,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """Fill the masks between the start and end tokens in the labels mask.

    Args:
        start_token (str): the start token to look for in the input ids.
        end_token (str): the end token to look for in the input ids.
        input_ids (B, seq_len): tensor of input ids.
        tokenizer (AutoTokenizer): The tokenizer used to convert tokens to ids.
        labels_mask (B, seq_len): tensor of labels mask, indicating the positions that should

    Returns:
        (B, seq_len): updated labels mask.
    """
    start_idx = (input_ids == tokenizer.convert_tokens_to_ids(start_token)).nonzero()
    end_idx = (input_ids == tokenizer.convert_tokens_to_ids(end_token)).nonzero()
    for start, end in zip(start_idx, end_idx):
        # NOTE: we should include the end token in the mask
        labels_mask[start[0], start[1] : end[1] + 1] = True
    return labels_mask


def get_label_mask(
    input_ids: torch.Tensor, tokenizer: AutoTokenizer, label_components: list[str]
) -> torch.Tensor:
    """Get the labels mask for the input ids.

    Args:
        input_ids: (B, seq_len) tensor of input ids.
        tokenizer: The tokenizer used to convert tokens to ids.
        label_components (list[str]): The components that should be used for computing the loss.

    Returns:
        labels_mask: (B, seq_len) tensor of labels mask, indicating the positions that should
            be used for computing the loss.
    """
    labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for label in label_components:
        assert f"{label}_start" in SPECIAL_TOKENS, (
            f"{label}_start must be in SPECIAL_TOKENS if {label} is"
        )
        assert f"{label}_end" in SPECIAL_TOKENS, (
            f"{label}_end must be in SPECIAL_TOKENS if {label} is"
        )
        labels_mask = fill_masks_between_special_tokens(
            start_token=SPECIAL_TOKENS[f"{label}_start"],
            end_token=SPECIAL_TOKENS[f"{label}_end"],
            input_ids=input_ids,
            tokenizer=tokenizer,
            labels_mask=labels_mask,
        )

    return labels_mask


def get_assistant_mask(
    tokenizer: AutoTokenizer,
    tokens: torch.Tensor | list[int],
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>",
    role: str = "assistant",
) -> torch.Tensor:
    """Generate a boolean mask indicating which tokens correspond to the assistant's response.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used to convert tokens to IDs.
        tokens (torch.Tensor | list[int]): The sequence of token IDs.
        bos_token (str, optional): The beginning-of-sequence token. Defaults to "<|im_start|>".
        eos_token (str, optional): The end-of-sequence token. Defaults to "<|im_end|>".
        role (str, optional): The assistant role string. Defaults to "assistant".

    Returns:
        torch.Tensor: A boolean mask with True for assistant tokens, False otherwise.

    Reference:
        Adapted from:
    """
    # Offsets: skip the bos + "assistant\n" (always 3 tokens) and include the eos (+1)
    # for supervision
    START_OFFSET = 3
    END_OFFSET = 1

    np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)

    # Retrieve token IDs for the markers and the role.
    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    role_id = tokenizer.convert_tokens_to_ids(role)

    # Locate all positions where the start and end markers appear.
    start_indices = np.where(np_tokens == bos_token_id)[0]
    end_indices = np.where(np_tokens == eos_token_id)[0]

    # Initialize the mask with False values.
    masks = np.zeros_like(np_tokens, dtype=bool)
    assert len(start_indices) == len(end_indices), (
        f"Number of bos ({len(start_indices)}) does not match eos ({len(end_indices)})"
    )
    # For each pair of bos/eos, check if the role is 'assistant'
    # and apply the mask accordingly.
    for start, end in zip(start_indices, end_indices):
        if np_tokens[start + 1] == role_id:
            # Mask tokens from after the assistant header (start+3) to include the end
            # marker (end+1)
            masks[start + START_OFFSET : end + END_OFFSET] = True

    assert masks.shape == np_tokens.shape
    if isinstance(tokens, torch.Tensor):
        return torch.from_numpy(masks)
    else:
        return masks.tolist()


def get_role_eos_mask(
    input_ids: torch.Tensor,
    tokenizer,
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>",
    role: str = "assistant",
) -> torch.Tensor:
    """Get mask that is True only for EOS tokens that end assistant responses.

    Args:
        input_ids: (B, seq_len) tensor of input ids.
        tokenizer: The tokenizer used to convert tokens to ids.
        bos_token: The beginning-of-sequence token.
        eos_token: The end-of-sequence token.
        role: The role string to match (e.g., "assistant").

    Returns:
        eos_mask: (B, seq_len) boolean tensor with True only at assistant EOS positions.
    """
    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    role_id = tokenizer.convert_tokens_to_ids(role)
    eos_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # Find all bos and eos positions: (batch_idx, seq_idx)
    bos_positions = torch.where(input_ids == bos_token_id)
    eos_positions = torch.where(input_ids == eos_token_id)

    num_bos = bos_positions[0].shape[0]
    num_eos = eos_positions[0].shape[0]

    # Handle inference case: if there are more BOS than EOS tokens,
    # it means the last role (e.g., assistant) doesn't have an EOS yet.
    # We only consider the BOS tokens that have corresponding EOS tokens.
    if num_eos < num_bos:
        # Truncate bos_positions to match the number of eos tokens
        bos_positions = (bos_positions[0][:num_eos], bos_positions[1][:num_eos])

    # Get the token after each bos position (the role token)
    role_tokens = input_ids[bos_positions[0], bos_positions[1] + 1]

    # Get the corresponding eos positions for assistant blocks
    assistant_batch_idx = eos_positions[0][role_tokens == role_id]
    assistant_eos_idx = eos_positions[1][role_tokens == role_id]

    # Set mask to True at those positions
    eos_mask[assistant_batch_idx, assistant_eos_idx] = True

    return eos_mask
