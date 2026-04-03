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

"""Qwen-specific processor implementation."""

from functools import partial
from typing import Any, Callable

import torch
from transformers import AutoProcessor

from alpamayo_r1.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN
from alpamayo_r1.chat_template.conversation import build_conversation
from alpamayo_r1.utils.get_label_mask import get_label_mask, get_role_eos_mask


def sort_images_by_camera_ids(
    image_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    relative_timestamps: torch.Tensor | None = None,
    return_camera_ids: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Sort camera images (and optionally timestamps) by their camera ids.

    Args:
        image_frames (torch.Tensor): shape (num_chunk, num_frames, 3, width, height)
        camera_indices (torch.Tensor): shape (num_chunk)
        relative_timestamps (torch.Tensor | None): shape (num_chunk, ...).
            If provided, sorted alongside images and appended to the return tuple.
        return_camera_ids (bool): If True, also return the sorted camera indices

    Returns:
        When only image_frames is requested: sorted image_frames tensor.
        Otherwise a tuple of (sorted_image_frames, [sorted_camera_indices],
            [sorted_relative_timestamps]) depending on which optional outputs
            are enabled.
    """
    sorted_indices = torch.argsort(camera_indices, stable=True)
    sorted_image_frames = image_frames[sorted_indices]

    result: list[torch.Tensor] = [sorted_image_frames]
    if return_camera_ids:
        result.append(camera_indices[sorted_indices])
    if relative_timestamps is not None:
        result.append(relative_timestamps[sorted_indices])

    if len(result) == 1:
        return result[0]
    return tuple(result)


def basic_collation_fn(batch, unstackable_keys: list[str] = []):
    """Collate function that does torch.stack on the keys that are tensors, and for the other keys
    returns a list.
    """
    stackable = {k: isinstance(v, torch.Tensor) for k, v in batch[0].items()}

    # set custom unstackable keys to False
    for k in unstackable_keys:
        if k in stackable:
            stackable[k] = False

    # stackable keys
    out = {
        k: (torch.stack([row[k] for row in batch]) if stackable[k] else [row[k] for row in batch])
        for k in stackable
    }

    return out


class QwenProcessor:
    """Processor implementation for Qwen VLM models."""

    def __init__(
        self,
        vlm_name_or_path: str,
        traj_vocab_size: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
    ) -> None:
        """Initialize the processor.

        Args:
            vlm_name_or_path: Path to the VLM model.
            traj_vocab_size: Size of the trajectory vocabulary.
            min_pixels: Minimum number of pixels for image resizing.
            max_pixels: Maximum number of pixels for image resizing.
            include_camera_ids: Whether to include camera IDs as text before images.
        """
        self.vlm_name_or_path = vlm_name_or_path
        self.traj_vocab_size = traj_vocab_size
        self._processor = None
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums

    def build_processor(self) -> AutoProcessor:
        """Build the Qwen processor."""
        processor_kwargs = {}
        if self._min_pixels is not None:
            processor_kwargs["min_pixels"] = self._min_pixels
        if self._max_pixels is not None:
            processor_kwargs["max_pixels"] = self._max_pixels

        processor = AutoProcessor.from_pretrained(self.vlm_name_or_path, **processor_kwargs)
        tokenizer = processor.tokenizer

        # Add traj tokens to the tokenizer
        if self.traj_vocab_size is not None:
            discrete_tokens = [f"<i{v}>" for v in range(self.traj_vocab_size)]
            num_new_tokens = tokenizer.add_tokens(discrete_tokens)
            assert len(discrete_tokens) == num_new_tokens
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
            tokenizer.traj_token_end_idx = tokenizer.convert_tokens_to_ids(
                f"<i{self.traj_vocab_size - 1}>"
            )

        # Add all special tokens to the tokenizer
        special_tokens = list(SPECIAL_TOKENS.values())
        tokenizer.add_tokens(special_tokens, special_tokens=True)

        # Add mapping from traj token names to ids
        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
        }

        return processor

    def get_preprocess_data_fn(
        self,
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Get the preprocess data function for the Qwen VLM model."""
        processor = self.processor  # This will call build_processor() if needed
        return partial(
            self._preprocess_data,
            processor=processor,
            num_tokens_per_history_traj=num_tokens_per_history_traj,
            num_tokens_per_future_traj=num_tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            label_components=label_components,
            generation_mode=generation_mode,
        )

    def collate_fn(self, data: list[dict[str, Any]], padding_side: str = "left") -> dict[str, Any]:
        """Collate function for Qwen data."""
        # raw image_frames may not have the same size, so we don't stack it
        batched_data: dict[str, Any] = basic_collation_fn(data, unstackable_keys=["image_frames"])

        # stack items with the same dimension in tokenized_data
        tokenized_data = {}
        for k in batched_data["tokenized_data"][0].keys():
            if k not in ["text"]:
                tokenized_data[k] = torch.cat([row[k] for row in batched_data["tokenized_data"]])

        # tokenize the text (with padding) and update the tokenized_data
        batch_text = [instance["text"] for instance in batched_data["tokenized_data"]]
        processed_inputs = self.processor.tokenizer(
            batch_text, return_tensors="pt", padding_side=padding_side, padding=True
        )
        tokenized_data.update(processed_inputs)
        batched_data["tokenized_data"] = tokenized_data

        # assert label_components is the same for all instances in the batch
        label_components = batched_data["label_components"][0]
        assert all(l_i == label_components for l_i in batched_data["label_components"]), (
            "label_components is not the same for all instances in the batch."
        )

        # assert generation_mode is the same for all instances in the batch
        generation_mode = batched_data["generation_mode"][0]
        assert all(g_i == generation_mode for g_i in batched_data["generation_mode"]), (
            "generation_mode is not the same for all instances in the batch."
        )

        # generate mask of labels for loss computation
        if not generation_mode:
            batched_data["labels_mask"] = get_label_mask(
                input_ids=tokenized_data["input_ids"],
                tokenizer=self.processor.tokenizer,
                label_components=label_components,
            )

            # set the mask of assistant's eos token as True
            eos_mask_assistant = get_role_eos_mask(
                input_ids=tokenized_data["input_ids"],
                tokenizer=self.processor.tokenizer,
                bos_token="<|im_start|>",
                eos_token="<|im_end|>",
                role="assistant",
            )
            batched_data["labels_mask"] |= eos_mask_assistant
        else:
            # in generation mode, we set all labels to be False
            batched_data["labels_mask"] = torch.zeros_like(
                tokenized_data["input_ids"],
                dtype=torch.bool,
                device=tokenized_data["input_ids"].device,
            )

        return batched_data

    def _preprocess_data(
        self,
        data: dict[str, Any],
        processor: AutoProcessor,
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
    ) -> dict[str, Any]:
        """Preprocess data for the Qwen VLM model."""
        # 1. generate image inputs
        images, camera_ids, sorted_ts = sort_images_by_camera_ids(
            data["image_frames"],
            data["camera_indices"],
            relative_timestamps=data["relative_timestamps"],
            return_camera_ids=True,
        )
        data["relative_timestamps"] = sorted_ts

        data["image_frames"] = images
        data["camera_indices"] = camera_ids
        data["label_components"] = label_components
        data["generation_mode"] = generation_mode

        # 2. build the chat template
        messages = build_conversation(
            data=data,
            num_tokens_per_history_traj=num_tokens_per_history_traj,
            num_tokens_per_future_traj=num_tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            generation_mode=generation_mode,
            include_camera_ids=self.include_camera_ids,
            camera_ids=camera_ids,
            include_frame_nums=self.include_frame_nums,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=generation_mode,
        )

        # 3. convert to float for bicubic interpolation, normalize to [0, 1]
        # Handle both uint8 (0-255) and float (0-1) inputs
        if images.dtype == torch.uint8:
            images_for_vlm = images.float() / 255.0
        else:
            images_for_vlm = images.float()
        image_inputs = processor.image_processor(
            images=images_for_vlm.flatten(0, 1), do_rescale=False
        )

        # 4. expand each image to the multiple image tokens
        # TODO: this part can be included in the conversation building step
        index = 0
        merge_length = processor.image_processor.merge_size**2
        while processor.image_token in text:
            text = text.replace(
                processor.image_token,
                "<|placeholder|>" * (image_inputs["image_grid_thw"][index].prod() // merge_length),
                1,
            )
            index += 1
        text = text.replace("<|placeholder|>", processor.image_token)
        # print(index, len(image_inputs["image_grid_thw"]))
        assert index == len(image_inputs["image_grid_thw"])

        tokenized_data = {"text": text, **image_inputs}
        return tokenized_data

    @property
    def processor(self) -> Any:
        """Get the vlm processor with expanded vocabulary for single data."""
        if self._processor is None:
            self._processor = self.build_processor()
        return self._processor


# For backward compatibility with existing config files
def build_processor(
    vlm_name_or_path: str,
    traj_vocab_size: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    include_camera_ids: bool = False,
    include_frame_nums: bool = False,
) -> AutoProcessor:
    """Build the processor for the Qwen VLM."""
    qwen_proc = QwenProcessor(
        vlm_name_or_path=vlm_name_or_path,
        traj_vocab_size=traj_vocab_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
    )
    return qwen_proc.build_processor()


def get_preprocess_data_fn_from_model_config(
    components_order: list[str] = None,
    components_prompt: list[str] = None,
    label_components: list[str] = None,
    generation_mode: bool = False,
    include_camera_ids: bool = False,
    include_frame_nums: bool = False,
    model_config: Any = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Get the preprocess data function for the Qwen VLM model."""
    qwen_proc = QwenProcessor(
        vlm_name_or_path=model_config.vlm_name_or_path,
        traj_vocab_size=model_config.traj_vocab_size,
        min_pixels=model_config.min_pixels,
        max_pixels=model_config.max_pixels,
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
    )
    return qwen_proc.get_preprocess_data_fn(
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        num_tokens_per_future_traj=model_config.tokens_per_future_traj,
        components_order=components_order,
        components_prompt=components_prompt,
        label_components=label_components,
        generation_mode=generation_mode,
        **kwargs,
    )


def collate_fn_from_model_config(
    data: list[dict[str, Any]],
    model_config=None,
    padding_side: str = "left",
    include_camera_ids: bool = False,
    include_frame_nums: bool = False,
) -> dict[str, Any]:
    """Wrapper for the origin collate_fn to instantiate from the model config."""
    qwen_proc = QwenProcessor(
        vlm_name_or_path=model_config.vlm_name_or_path,
        traj_vocab_size=model_config.traj_vocab_size,
        min_pixels=model_config.min_pixels,
        max_pixels=model_config.max_pixels,
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
    )
    return qwen_proc.collate_fn(data, padding_side=padding_side)
