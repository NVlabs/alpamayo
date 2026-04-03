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

"""Cosmos-RL DataPacker(s) for Alpamayo ReasoningVLA."""

from __future__ import annotations

from typing import Any, cast

import rl.state as alp_state
import torch
from alpamayo_r1.models.base_model import SPECIAL_TOKENS
from alpamayo_r1.processor.qwen_processor import basic_collation_fn
from rl.base_data_packer import BaseRLDataPacker
from rl.utils.trajectory_decode import decode_rollout_trajectory
from vllm.inputs import TokensPrompt


class RVLADataPacker(BaseRLDataPacker):
    """Bridges Alpamayo samples to vLLM prompts + GRPO trainer batches."""

    def _process_rollout_sample(self, sample: dict) -> TokensPrompt:
        """Convert raw sample to vLLM rollout prompt.

        Hook called by BaseRLDataPacker._get_sample.
        """
        return self._sample_to_rollout_prompt(sample)

    def _sample_to_rollout_prompt(self, sample: dict) -> TokensPrompt:
        """Convert one Alpamayo sample dict into a vLLM-compatible prompt dict."""
        alp_tok = alp_state.get_tokenizer()
        ckpt_cfg = alp_state.get_ckpt_cfg()
        traj_fuser = alp_state.get_traj_fuser()

        td = sample.get("tokenized_data")
        if td is None:
            raise KeyError("Expected key 'tokenized_data' in Alpamayo sample for rollout.")

        input_ids = td.get("input_ids")
        if input_ids is None:
            raw_text = td.get("text")
            if raw_text is None:
                raise KeyError(
                    "Expected key 'input_ids' or 'text' inside sample['tokenized_data'] "
                    "for rollout."
                )
            input_ids = torch.tensor(alp_tok.encode(cast(str, raw_text)), dtype=torch.long)

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Fuse trajectory tokens into input_ids
        fused_input_ids = input_ids
        ego_history_xyz = sample.get("ego_history_xyz", None)
        ego_history_rot = sample.get("ego_history_rot", None)
        ego_future_xyz = sample.get("ego_future_xyz", None)
        ego_future_rot = sample.get("ego_future_rot", None)

        if isinstance(ego_history_xyz, torch.Tensor) and isinstance(ego_history_rot, torch.Tensor):
            if ego_history_xyz.dim() == 3:
                ego_history_xyz = ego_history_xyz.unsqueeze(1)
            if ego_history_rot.dim() == 4:
                ego_history_rot = ego_history_rot.unsqueeze(1)
            if isinstance(ego_future_xyz, torch.Tensor) and ego_future_xyz.dim() == 3:
                ego_future_xyz = ego_future_xyz.unsqueeze(1)
            if isinstance(ego_future_rot, torch.Tensor) and ego_future_rot.dim() == 4:
                ego_future_rot = ego_future_rot.unsqueeze(1)

            if fused_input_ids.dim() == 1:
                fused_input_ids = fused_input_ids.unsqueeze(0)

            traj_data = {
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
                "ego_future_xyz": ego_future_xyz,
                "ego_future_rot": ego_future_rot,
            }
            fused_input_ids = traj_fuser.fuse_traj_tokens(fused_input_ids, traj_data)
        else:
            raise ValueError("Expected ego_history_xyz/ego_history_rot tensors for traj fusion.")

        # fused_input_ids is expected to be [1, T]
        if not (isinstance(fused_input_ids, torch.Tensor) and fused_input_ids.ndim == 2):
            raise ValueError("Expected fused_input_ids to be a 2D tensor [1, T].")
        token_ids = fused_input_ids[0].tolist()

        # Compress image placeholders back to a single token per run.
        img_id = alp_tok.convert_tokens_to_ids(SPECIAL_TOKENS["image_pad"])
        if img_id is not None:
            token_ids = [
                tid
                for i, tid in enumerate(token_ids)
                if not (tid == img_id and i > 0 and token_ids[i - 1] == img_id)
            ]

        prompt_dict: TokensPrompt = {"prompt_token_ids": token_ids}

        # Build multi-modal data processor config
        mm_data: dict[str, Any] = {}
        image_frames = sample.get("image_frames", None)
        if image_frames is not None:
            frames = image_frames.flatten(0, 1)  # (N, C, H, W)
            images = [f.cpu() for f in frames]
            if len(images) > 0:
                mm_data["image"] = images

        if mm_data:
            prompt_dict["multi_modal_data"] = mm_data
            mm_kwargs = {
                "do_rescale": True,
                "max_pixels": ckpt_cfg.max_pixels,
                "min_pixels": ckpt_cfg.min_pixels,
            }
            mm_kwargs = {k: v for k, v in mm_kwargs.items() if v is not None}
            prompt_dict["hf_processor_mm_kwargs"] = mm_kwargs
            prompt_dict["mm_processor_kwargs"] = mm_kwargs

        return prompt_dict

    def get_rollout_input(self, item: dict[str, str]) -> Any:
        """Get a rollout-ready item (already processed into a vLLM prompt dict).

        Args:
            item: Lightweight index dict ``{"idx": str, "split": str}`` produced by
                :class:`~rl.base_dataset.AlpamayoCosmosDataset` and carried as
                ``RLPayload.prompt`` through the cosmos-rl pipeline.  The actual
                sample data is fetched on-demand via :meth:`_get_sample`.
        """
        idx = int(item["idx"])
        split = item["split"]
        return self._get_sample(split=split, n=idx, role="rollout")

    def get_cot_input(self, item: dict[str, str]) -> Any:
        """Extract chain-of-thought fields from a policy sample."""
        idx = int(item["idx"])
        split = item["split"]
        # IMPORTANT: Avoid routing through `_get_sample_raw` in the prefetch packer, because that
        # would use `_fetch_sample`'s default tag ("raw") and create a raw-tag prefetch queue.
        # We want prefetch queue tags to be role-specific (policy/rollout) only.
        sample = self._get_sample(split=split, n=idx, role="policy")
        return {"cot": sample.get("cot", "")}

    def rollout_collate_fn(self, items: list[Any]) -> Any:
        """Convert rollout items into vLLM prompts.

        Fast path: if items already are vLLM prompt dicts (produced by get_rollout_input),
        we return them directly.
        """
        prompts: list[TokensPrompt] = []
        for item in items:
            if isinstance(item, dict) and "prompt_token_ids" in item:
                prompts.append(cast(TokensPrompt, item))
                continue
            if not isinstance(item, dict):
                raise TypeError(f"Expected rollout item to be a dict, but got {type(item)}")
            prompts.append(self._sample_to_rollout_prompt(item))

        return prompts

    def get_policy_input(
        self,
        sample: dict,
        rollout_output: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """Process samples & rollout output before collating them into a mini-batch."""
        idx = int(sample["idx"])
        split = sample["split"]
        data_dict = cast(dict, self._get_sample(split=split, n=idx, role="policy"))
        assert isinstance(data_dict, dict), "data_dict must be a dict"

        tokenized = data_dict["tokenized_data"]
        alp_tok = alp_state.get_tokenizer()
        traj_tok = alp_state.get_traj_tokenizer()

        if tokenized is None:
            raise KeyError(f"Missing tokenized_data for sample idx={idx}, split={split}")

        input_ids = tokenized.get("input_ids")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise ValueError("tokenized_data.input_ids must be a torch.Tensor of shape [1, L]")

        end_id = alp_tok.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future_end"])
        ids_row = input_ids[0]

        gen_ids = alp_tok(rollout_output, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]

        if "text" in tokenized and isinstance(tokenized["text"], str):
            tokenized["text"] += rollout_output
            tokenized["text"] += SPECIAL_TOKENS["traj_future_end"]

        if gen_ids.numel() > 0:
            end_token_tensor = torch.tensor([end_id], dtype=ids_row.dtype, device=ids_row.device)
            new_ids_row = torch.cat([ids_row, gen_ids.to(ids_row.device), end_token_tensor], dim=0)

            new_mask = torch.zeros_like(new_ids_row, dtype=torch.bool)
            new_mask[-(gen_ids.numel() + 1) :] = True  # include the end token

            tokenized["labels_mask"] = new_mask.unsqueeze(0)
            tokenized["input_ids"] = new_ids_row.unsqueeze(0)

        predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
            rollout_output,
            data_dict["ego_history_xyz"],
            data_dict["ego_history_rot"],
            tokenizer=alp_tok,
            traj_tokenizer=traj_tok,
            model_config=alp_state.get_ckpt_cfg(),
        )
        data_dict["ego_rollout_xyz"] = predicted_fut_xyz
        data_dict["ego_rollout_rot"] = predicted_fut_rot

        return data_dict

    def policy_compute_max_len(self, processed_samples: list[Any]) -> int:
        """Return the maximum sequence length across processed policy samples."""
        return max(x["tokenized_data"]["input_ids"].shape[1] for x in processed_samples)

    def policy_collate_fn(
        self,
        processed_samples: list[Any],
        computed_max_len: int,
    ) -> dict[str, Any]:
        """Collate the mini-batch into the kwargs required by the policy model."""
        batch: dict[str, Any] = basic_collation_fn(
            processed_samples,
            unstackable_keys=["image_frames"],
        )

        tokenized_data = {}
        for k in batch["tokenized_data"][0].keys():
            if k not in ["text"]:
                tokenized_data[k] = torch.cat([row[k] for row in batch["tokenized_data"]])
        batch["tokenized_data"] = tokenized_data

        label_components = batch["label_components"][0]
        assert all(l_i == label_components for l_i in batch["label_components"]), (
            "label_components is not the same for all instances in the batch."
        )

        # Lift inputs needed by GRPO trainer to top-level
        tokenized = batch.get("tokenized_data", {})
        if "input_ids" in tokenized:
            batch["input_ids"] = tokenized["input_ids"]
        if "position_ids" in tokenized:
            batch["position_ids"] = tokenized["position_ids"]
        if "attention_mask" in tokenized:
            batch["attention_mask"] = tokenized["attention_mask"]
        if "labels_mask" in tokenized:
            lmask = tokenized["labels_mask"]
            if lmask.dtype == torch.bool:
                batch["labels_mask"] = lmask
            else:
                batch["labels_mask"] = lmask.bool()

        # Build logprob_masks expected by trainer.compute_logprobs
        if "labels_mask" in batch:
            lmask = batch["labels_mask"]
            if lmask.dtype == torch.bool:
                batch["logprob_masks"] = lmask
            else:
                batch["logprob_masks"] = lmask.bool()
        else:
            if "input_ids" in tokenized:
                B, T = tokenized["input_ids"].shape
                mask = torch.ones(B, T, dtype=torch.bool, device=tokenized["input_ids"].device)
                mask[:, 0] = False
                batch["logprob_masks"] = mask
            else:
                raise KeyError("tokenized_data.input_ids is required to construct logprob_masks")

        return batch
