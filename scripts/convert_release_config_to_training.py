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

#!/usr/bin/env python3
r"""Convert nvidia/Alpamayo-R1-10B or nvidia/Alpamayo-1.5-10B into a RL training compatible format.

This script downloads the HuggingFace releases, converts the config,
symlinks model weights, and saves the full processor (expanded tokenizer
+ preprocessor configs) -- all in a single command.

Supports both alpamayo1 and alpamayo1.5 release models.

Steps performed:
1. Download the alpamayo model (model weights + config)
2. Download ``nvidia/Cosmos-Reason2-8B`` (tokenizer + preprocessor files only)
3. Symlink safetensors and index from the alpamayo snapshot
4. Convert ``config.json`` (remap _target_ paths, architectures, model_type,
   vlm_name_or_path)
5. Load the model via ``ReasoningVLA.from_pretrained`` and save the full
   processor (tokenizer with trajectory + special tokens, preprocessor configs)

Usage:
    # Convert alpamayo1.5 (default)
    python scripts/convert_release_config_to_training.py

    # Convert alpamayoR1
    python scripts/convert_release_config_to_training.py \
        --alpamayo-model nvidia/Alpamayo-R1-10B

    python scripts/convert_release_config_to_training.py \
        --output-dir /my/custom/path

    python scripts/convert_release_config_to_training.py \
        --vlm-name-or-path /local/Cosmos-Reason2-8B
"""

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

# ============================================================================
# Hydra _target_ remapping table  (release -> training)
# ============================================================================
TARGET_REMAP = {
    # v1 (Alpamayo-R1-10B): _target_ paths already use alpamayo_r1.* -- no remap needed.
    # --- v1.5 (Alpamayo-1.5-10B): remap alpamayo1_5.* -> alpamayo_r1.* ---
    "alpamayo1_5.models.action_in_proj.": "alpamayo_r1.models.action_in_proj.",
    "alpamayo1_5.models.delta_tokenizer.": "alpamayo_r1.models.delta_tokenizer.",
    "alpamayo1_5.action_space.": "alpamayo_r1.action_space.",
    "alpamayo1_5.diffusion.": "alpamayo_r1.diffusion.",
}

DEFAULT_VLM_NAME_OR_PATH = "nvidia/Cosmos-Reason2-8B"
DEFAULT_ALPAMAYO_MODEL = "nvidia/Alpamayo-1.5-10B"


def _default_output_dir(model: str) -> str:
    """Derive a default output directory from the model name."""
    tag = model.rsplit("/", 1)[-1]
    return f"/tmp/{tag}_training"


# ============================================================================
# Config conversion helpers (pure functions, no side effects)
# ============================================================================


def remap_target(target: str) -> str:
    """Remap a Hydra _target_ string from release -> training namespace."""
    for old_prefix, new_prefix in TARGET_REMAP.items():
        if target.startswith(old_prefix):
            return new_prefix + target[len(old_prefix) :]
    return target


def remap_targets_recursive(obj: object) -> None:
    """Walk a nested dict/list and remap all _target_ values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "_target_" and isinstance(value, str):
                obj[key] = remap_target(value)
            else:
                remap_targets_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            remap_targets_recursive(item)


def convert_config(
    config: dict,
    vlm_name_or_path: str = DEFAULT_VLM_NAME_OR_PATH,
) -> tuple[dict, list[str]]:
    """Convert a release config dict to training format.

    Args:
        config: The original release config dict.
        vlm_name_or_path: Value to set for vlm_name_or_path.

    Returns:
        Tuple of (converted config dict, list of change descriptions).
    """
    out = copy.deepcopy(config)
    changes: list[str] = []

    # --- 1. Update model_type ---
    old_model_type = out.get("model_type")
    if old_model_type != "alpamayo_reasoning_vla":
        out["model_type"] = "alpamayo_reasoning_vla"
        changes.append(f"model_type: {old_model_type!r} -> 'alpamayo_reasoning_vla'")

    # --- 2. Update architectures ---
    old_arch = out.get("architectures")
    if old_arch != ["ReasoningVLA"]:
        out["architectures"] = ["ReasoningVLA"]
        changes.append(f"architectures: {old_arch} -> ['ReasoningVLA']")

    # --- 3. Remap all _target_ paths ---
    old_targets = _collect_targets(config)
    remap_targets_recursive(out)
    new_targets = _collect_targets(out)
    for path, old_val in old_targets.items():
        new_val = new_targets.get(path, old_val)
        if old_val != new_val:
            changes.append(f"_target_ remap: {path}: {old_val!r} -> {new_val!r}")

    # --- 4. Set vlm_name_or_path ---
    old_vlm = out.get("vlm_name_or_path")
    out["vlm_name_or_path"] = vlm_name_or_path
    if old_vlm != vlm_name_or_path:
        changes.append(f"vlm_name_or_path: {old_vlm!r} -> {vlm_name_or_path!r}")
    else:
        changes.append(f"vlm_name_or_path: kept as {old_vlm!r}")

    return out, changes


def _collect_targets(obj: object, prefix: str = "") -> dict[str, str]:
    """Recursively collect all _target_ values with their JSON paths."""
    targets: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == "_target_" and isinstance(value, str):
                targets[prefix] = value
            else:
                targets.update(_collect_targets(value, path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            targets.update(_collect_targets(item, f"{prefix}[{i}]"))
    return targets


# ============================================================================
# Pipeline steps
# ============================================================================


def download_models(alpamayo_model: str, vlm_model: str) -> tuple[Path, Path]:
    """Download HF models and return their local snapshot paths."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {alpamayo_model} ...")
    alpamayo_path = Path(snapshot_download(alpamayo_model))
    print(f"  -> {alpamayo_path}")

    print(f"Downloading {vlm_model} (tokenizer + preprocessor only) ...")
    vlm_path = Path(
        snapshot_download(
            vlm_model,
            allow_patterns=[
                "*.json",
                "*.txt",
                "merges.txt",
                "vocab.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
            ],
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx"],
        )
    )
    print(f"  -> {vlm_path}")

    return alpamayo_path, vlm_path


def setup_output_dir(
    output_dir: Path,
    alpamayo_path: Path,
    vlm_path: Path,
    vlm_name_or_path: str,
) -> list[str]:
    """Create the output directory with symlinked weights and converted config.

    Returns a list of actions performed (for logging).
    """
    actions: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Symlink safetensors and index ---
    for src_file in sorted(alpamayo_path.iterdir()):
        if (
            src_file.name.endswith(".safetensors")
            or src_file.name == "model.safetensors.index.json"
        ):
            real_src = src_file.resolve()
            dst = output_dir / src_file.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(real_src, dst)
            actions.append(f"symlink: {src_file.name} -> {real_src}")

    # --- Copy generation_config.json from VLM ---
    gen_cfg = vlm_path / "generation_config.json"
    if gen_cfg.exists():
        shutil.copy2(gen_cfg, output_dir / "generation_config.json")
        actions.append("copy: generation_config.json from VLM")

    # --- Convert config.json ---
    with open(alpamayo_path / "config.json") as f:
        original_config = json.load(f)

    converted, changes = convert_config(original_config, vlm_name_or_path)
    with open(output_dir / "config.json", "w") as f:
        json.dump(converted, f, indent=2)
        f.write("\n")
    actions.append(f"config.json converted ({len(changes)} changes)")
    actions.extend(f"  config: {c}" for c in changes)

    return actions


def setup_training_sys_path() -> None:
    """Add training code directories to sys.path based on this script's location."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent  # alpamayo/

    paths_to_add = [
        project_dir / "src",
        project_dir / "finetune",
        project_dir / "finetune" / "rl" / "models",
    ]
    for p in paths_to_add:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def load_model_and_save_processor(output_dir: Path) -> None:
    """Load the converted checkpoint and save the full processor.

    Saves the expanded tokenizer (with trajectory + special tokens) as well as
    preprocessor_config.json and video_preprocessor_config.json.
    """
    setup_training_sys_path()

    from reasoning_vla.base_model import RLWrapperReasoningVLA

    print(f"Loading model from {output_dir} ...")
    model = RLWrapperReasoningVLA.from_pretrained(output_dir)
    print("Model loaded successfully.")

    print(f"Saving processor to {output_dir} ...")
    model.processor.save_pretrained(output_dir)
    print("Processor saved.")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert nvidia/Alpamayo-R1-10B or nvidia/Alpamayo-1.5-10B into a "
            "training-ready checkpoint for ReasoningVLA."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the training checkpoint (default: /tmp/<model>_training)",
    )
    parser.add_argument(
        "--vlm-name-or-path",
        type=str,
        default=DEFAULT_VLM_NAME_OR_PATH,
        help=f"VLM model for tokenizer/preprocessor (default: {DEFAULT_VLM_NAME_OR_PATH})",
    )
    parser.add_argument(
        "--alpamayo-model",
        type=str,
        default=DEFAULT_ALPAMAYO_MODEL,
        help=f"Alpamayo HF model ID (default: {DEFAULT_ALPAMAYO_MODEL})",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(_default_output_dir(args.alpamayo_model))
    output_dir: Path = args.output_dir.resolve()

    print(f"\n{'=' * 70}")
    print("Convert release checkpoint -> training checkpoint")
    print(f"{'=' * 70}")
    print(f"  Alpamayo model : {args.alpamayo_model}")
    print(f"  VLM model      : {args.vlm_name_or_path}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'=' * 70}\n")

    # Step 1-2: Download models
    alpamayo_path, vlm_path = download_models(args.alpamayo_model, args.vlm_name_or_path)

    # Step 3-4: Symlink weights, copy generation config, convert config
    actions = setup_output_dir(output_dir, alpamayo_path, vlm_path, args.vlm_name_or_path)
    print("\nActions:")
    for action in actions:
        print(f"  {action}")

    # Step 5: Load model and save processor
    print()
    load_model_and_save_processor(output_dir)

    # Summary
    print(f"\n{'=' * 70}")
    print("Done! Training checkpoint written to:")
    print(f"  {output_dir}")
    print(f"{'=' * 70}")
    print("\nContents:")
    for f in sorted(output_dir.iterdir()):
        suffix = " -> " + str(f.resolve()) if f.is_symlink() else ""
        print(f"  {f.name}{suffix}")


if __name__ == "__main__":
    main()
