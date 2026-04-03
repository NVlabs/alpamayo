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

import os
import time
from typing import List, Optional

import torch
import vllm
from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.policy.config import Config, RolloutConfig
from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import apply_fp8_linear_patch
from cosmos_rl.utils import util
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from vllm import SamplingParams
from vllm.entrypoints.llm import LLM

from alpamayo_r1.models.base_model import SPECIAL_TOKENS


def vllm_version_check(rollout_config: RolloutConfig):
    """Raise if vLLM version is too old for the requested parallelism."""
    vllm_version = vllm.__version__
    if vllm_version < "0.9.0" and rollout_config.parallelism.pp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not supported for vLLM < 0.9.0, current version is %s"
            % vllm_version
        )


@RolloutRegistry.register("reasoning_vla_vllm_rollout")
class ReasoningVlaVllmRollout(RolloutBase):
    """vLLM-backed rollout engine for ReasoningVLA models."""

    def __init__(
        self,
        config: Config,
        parallel_dims: ParallelDims,
        device: torch.device,
        model_hf_config: Optional[AutoConfig] = None,
        **kwargs,
    ):
        """Rollout with vLLM as the backend.

        Args:
            config: Cosmos Config.
            parallel_dims: Parallel dimensions for the rollout engine.
            device: The device on which the rollout engine will run.
            model_hf_config: Hugging Face config used to initialize the generating model in vLLM.
        """
        super().__init__(config, parallel_dims, device, **kwargs)

        policy_config = self.config.policy
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation

        vllm_version_check(self.rollout_config)

        model_path = policy_config.model_name_or_path

        self.model_config = util.retry(AutoConfig.from_pretrained)(
            model_path, trust_remote_code=True
        )

        self.tokenizer = util.retry(AutoTokenizer.from_pretrained)(
            self.config.policy.model_name_or_path
        )

        self.pad_token_id = self.tokenizer.pad_token_id

        hf_config_path = self.config.policy.model_name_or_path
        try:
            generation_config = util.retry(GenerationConfig.from_pretrained)(hf_config_path)
            self.eos_token_ids = generation_config.eos_token_id
            if isinstance(self.eos_token_ids, int):
                self.eos_token_ids = [self.eos_token_ids]
        except Exception as e:
            logger.warning(
                "[Rollout] Failed to load generation config from %s: %s; "
                "falling back to default eos_token_id.",
                hf_config_path,
                e,
            )
            self.eos_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]

        self._engine_initialized = False
        self.rollout_engine = None
        self._model_param_map = None  # key: compatible name, value: param
        self.global_rank = int(os.environ.get("RANK", "0"))

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        """Create and configure the vLLM ``LLM`` engine."""
        logger.info(f"[Rollout] Initializing engine for rank: {self.global_rank}")
        self.val_sampling_params = SamplingParams(
            n=self.config.validation.n_generation,
            logprobs=0,
            top_p=self.config.validation.top_p
            if self.config.validation.top_p is not None
            else self.config.rollout.sampling_config.top_p,
            top_k=self.config.validation.top_k
            if self.config.validation.top_k is not None
            else self.config.rollout.sampling_config.top_k,
            temperature=self.config.validation.temperature
            if self.config.validation.temperature is not None
            else self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.validation.repetition_penalty
            if self.config.validation.repetition_penalty is not None
            else self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.validation.max_response_length
            if self.config.validation.max_response_length is not None
            else self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )
        self.sampling_params = SamplingParams(
            n=self.config.rollout.n_generation,
            logprobs=0,
            top_p=self.config.rollout.sampling_config.top_p,
            top_k=self.config.rollout.sampling_config.top_k,
            temperature=self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )
        # Keep special tokens in decoded output text so reward functions can parse markers like
        # `<|cot_end|>` / `<|traj_future_start|>` from `to_be_evaluated`.
        for sp in (self.val_sampling_params, self.sampling_params):
            if hasattr(sp, "skip_special_tokens"):
                sp.skip_special_tokens = False

        # Resolve traj end token id from the tokenizer (avoid hard-coded ids).
        try:
            self._traj_future_end_token_id = self.tokenizer.convert_tokens_to_ids(
                SPECIAL_TOKENS["traj_future_end"]
            )
        except Exception:
            self._traj_future_end_token_id = None

        def _reasoning_vla_vllm_hf_overrides(cfg):
            # Prefer the underlying LLM config for ReasoningVLA as text_config
            base_cfg = cfg.get_llm_config()
            setattr(cfg, "text_config", base_cfg)
            # Make vLLM aware of the custom architecture wrapper. We add both
            # the mixed-case and upper-case variants so they match the entries
            # registered in `ModelRegistry`.
            arches = list(getattr(cfg, "architectures", []) or [])
            for name in ("ReasoningVLA", "REASONING_VLA"):
                if name not in arches:
                    arches.append(name)
            setattr(cfg, "architectures", arches)
            return cfg

        if not self._engine_initialized:
            trust_remote_code = True  # set trust remote code default to True.

            model_path = self.config.policy.model_name_or_path

            rollout_parallelism = self.rollout_config.parallelism

            # disable VLLM_DISABLE_COMPILE_CACHE
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

            tp_size = rollout_parallelism.tp_size
            pp_size = rollout_parallelism.pp_size

            disable_mm_preprocessor_cache = False

            multimodal_type = {"qwen2_5_vl", "qwen3_vl"}

            model_type = self.model_config.model_type
            # Determine underlying LLM (text) model type for composite configs
            try:
                llm_model_type = getattr(self.model_config.get_llm_config(), "model_type", None)
            except Exception:
                llm_model_type = None
            # Disable mm preprocessor cache for Qwen VL backends to avoid profiling path issues
            if (model_type in multimodal_type) or (llm_model_type in multimodal_type):
                disable_mm_preprocessor_cache = True
            assert tp_size * pp_size == rollout_parallelism.world_size, (
                "[Rollout] For tensor/pipeline parallel, tp_size * pp_size must equal world_size. "
                f"Got tp_size={tp_size}, pp_size={pp_size}, "
                f"world_size={rollout_parallelism.world_size}."
            )

            self.quantization = quantization

            policy_config = self.config.policy

            # Ensure Transformers allows executing custom code in HF configs/models for workers
            # to avoid interactive trust prompts when loading local checkpoints with custom code.
            os.environ.setdefault("HF_ALLOW_CODE_EXECUTION", "1")

            # Allow env overrides to quickly test throughput-sensitive knobs without changing code.
            env_backend = os.getenv("COSMOS_VLLM_EXECUTOR_BACKEND")
            # Prefer in-process multiprocessing backend for single-GPU to reduce control overhead
            if env_backend:
                resolved_backend = env_backend
            else:
                resolved_backend = (
                    "uni" if rollout_parallelism.world_size == 1 else "external_launcher"
                )

            env_enforce_eager = os.getenv("COSMOS_VLLM_ENFORCE_EAGER")
            resolved_enforce_eager = (
                (env_enforce_eager.lower() in ["1", "true", "yes"])
                if env_enforce_eager
                else self.rollout_config.enforce_eager
            )

            env_chunked = os.getenv("COSMOS_VLLM_ENABLE_CHUNKED_PREFILL")
            resolved_chunked = (
                (env_chunked.lower() in ["1", "true", "yes"])
                if env_chunked
                else self.rollout_config.enable_chunked_prefill
            )

            env_gpu_util = os.getenv("COSMOS_VLLM_GPU_MEMORY_UTILIZATION")
            resolved_gpu_util = (
                float(env_gpu_util) if env_gpu_util else self.rollout_config.gpu_memory_utilization
            )

            if 2048 >= policy_config.model_max_length:
                default_max_batched = 2048
            else:
                default_max_batched = policy_config.model_max_length
            env_max_batched = os.getenv("COSMOS_VLLM_MAX_NUM_BATCHED_TOKENS")
            resolved_max_batched = int(env_max_batched) if env_max_batched else default_max_batched

            model_type = getattr(self.model_config, "model_type", None)
            if model_type == "alpamayo_reasoning_vla":
                hf_overrides_fn = _reasoning_vla_vllm_hf_overrides
            else:
                raise NotImplementedError(
                    f"Model type {model_type} not supported for vLLM rollout."
                )

            self.rollout_engine = LLM(
                model=model_path,
                hf_overrides=hf_overrides_fn,
                enable_sleep_mode=False,  # enable sleep could corrupt the cuda allocator.
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                skip_mm_profiling=True,
                enable_expert_parallel=False,
                distributed_executor_backend="external_launcher",
                dtype="auto",
                enforce_eager=resolved_enforce_eager,
                gpu_memory_utilization=resolved_gpu_util,
                disable_custom_all_reduce=True,
                disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
                enable_prompt_embeds=False,
                skip_tokenizer_init=False,
                max_model_len=policy_config.model_max_length,
                disable_log_stats=True,
                max_num_batched_tokens=resolved_max_batched,
                enable_chunked_prefill=resolved_chunked,
                enable_prefix_caching=False,
                trust_remote_code=trust_remote_code,
                quantization=self.quantization,
                seed=seed or 42,
                load_format=load_format,
            )
            self._engine_initialized = True
            logger.info("[Rollout] Engine initialized.")
            # initialization done.

            # Log effective vLLM engine config for debugging throughput differences
            try:
                mc = self.rollout_engine.llm_engine.get_model_config()
                logger.info(
                    {
                        "vllm_model_config": {
                            "runner_type": getattr(mc, "runner_type", None),
                            "max_model_len": getattr(mc, "max_model_len", None),
                            "dtype": str(getattr(mc, "dtype", None)),
                            "model": getattr(mc, "model", None),
                            "hf_text_config_type": type(
                                getattr(mc, "hf_text_config", None)
                            ).__name__,
                        },
                        "cosmos_vllm_effective": {
                            "distributed_executor_backend": resolved_backend,
                            "enforce_eager": resolved_enforce_eager,
                            "enable_chunked_prefill": resolved_chunked,
                            "gpu_memory_utilization": resolved_gpu_util,
                            "max_num_batched_tokens": resolved_max_batched,
                            "tp_size": tp_size,
                            "pp_size": pp_size,
                        },
                    }
                )
            except Exception:
                pass

            # patch the vllm model to use rowwise fp8
            if self.quantization == "fp8":
                from vllm.config import set_current_vllm_config

                vllm_config = self.rollout_engine.llm_engine.vllm_config
                with set_current_vllm_config(vllm_config):
                    apply_fp8_linear_patch(self.get_underlying_model())
        logger.info(f"[Rollout] Engine initialized for rank: {self.global_rank}")

    def post_init_hook(self, **kwargs):
        """No-op hook called after engine initialization."""
        pass

    @torch.no_grad()
    def rollout_generation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        *args,
        **kwargs,
    ) -> List[RolloutResult]:
        """Run vLLM generation on a batch of payloads and return rollout results."""
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        # List of payloads.
        # [
        #   payload,
        #   payload,
        #   ...
        # ]
        payloads = [payload.prompt for payload in payloads]

        # Pack the payloads into prompts for vllm.
        t_prep_start = time.perf_counter()
        prompts = [data_packer.get_rollout_input(payload) for payload in payloads]
        prompts = data_packer.rollout_collate_fn(prompts)
        t_prep_end = time.perf_counter()

        # List of completions per prompt.
        # [
        #   [completion_str, completion_str, ...],
        #   [completion_str, completion_str, ...],
        #   ...
        # ]

        response: List[List[str]] = []

        completion_logprobs: List[List[float]] = []

        stream = torch.cuda.current_stream() if stream is None else stream
        try:
            # Use vLLM's own scheduling/streams; avoid wrapping in a custom CUDA stream context
            t_gen_start = time.perf_counter()
            if is_validation:
                sampling_params = self.val_sampling_params
            else:
                sampling_params = self.sampling_params
            if not isinstance(getattr(self, "_traj_future_end_token_id", None), int):
                raise RuntimeError("_traj_future_end_token_id is not set or not an int. ")
            sampling_params.stop_token_ids = [self._traj_future_end_token_id]
            sampling_params.logprobs = 1
            results = self.rollout_engine.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            t_gen_end = time.perf_counter()

            total_generated_tokens = 0
            rollout_results: List[RolloutResult] = []
            for idx, output in enumerate(results):
                response.append([output.outputs[i].text for i in range(len(output.outputs))])
                completion_logprobs.append(
                    [output.outputs[i].cumulative_logprob for i in range(len(output.outputs))]
                )
                for out in output.outputs:
                    token_ids = getattr(out, "token_ids", None)
                    if token_ids is not None:
                        total_generated_tokens += len(token_ids)

            valid_completions: List[List[str]] = []
            valid_logprobs: List[List[float]] = []
            prompt_indices_to_remove: List[int] = []
            if is_validation:
                for i in range(len(response)):
                    rollout_results.append(
                        RolloutResult(
                            prompt=payloads[i],
                            completions=response[i],
                            cumulative_logprob=completion_logprobs[i],
                        )
                    )
            # Remove empty completions for training
            elif response:
                batch_size = len(prompts)
                assert len(response) == batch_size, (
                    f"Error: VLLM returned {len(response)} for {batch_size}"
                )
                for i in range(batch_size):
                    completion = response[i]
                    probs_i = completion_logprobs[i] if completion_logprobs is not None else None

                    skip_output = False
                    total_generation_count = len(completion)
                    empty_generation_count = 0
                    output_texts = []
                    output_probs = [] if probs_i is not None else None

                    for j in range(total_generation_count):
                        output_text = completion[j]
                        if output_text == "":
                            output_text = self.tokenizer.eos_token
                            empty_generation_count += 1
                        output_texts.append(output_text)
                        if output_probs is not None:
                            output_probs.append(probs_i[j])
                    # Skip the output if there is one or zero non-empty completions
                    skip_output = (total_generation_count - empty_generation_count) <= 1
                    if not skip_output:
                        valid_completions.append(output_texts)
                        if output_probs is not None:
                            valid_logprobs.append(output_probs)
                        rollout_results.append(
                            RolloutResult(
                                prompt=payloads[i],
                                completions=response[i],
                                cumulative_logprob=completion_logprobs[i],
                            )
                        )
                    else:
                        prompt_indices_to_remove.append(i)
                        rollout_results.append(RolloutResult(prompt=payloads[i], completions=[]))

            # Lightweight profiling log
            try:
                prep_s = t_prep_end - t_prep_start
                gen_s = t_gen_end - t_gen_start
                toks_per_s = (total_generated_tokens / gen_s) if gen_s > 0 else float("nan")
                logger.info(
                    {
                        "rollout_profile_s": {
                            "data_prepare_s": round(prep_s, 4),
                            "generate_s": round(gen_s, 4),
                            "generated_tokens": total_generated_tokens,
                            "gen_toks_per_s": round(toks_per_s, 2)
                            if isinstance(toks_per_s, float)
                            else toks_per_s,
                        }
                    }
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

        return rollout_results

    def get_underlying_model(self):
        """Get the underlying parallelized model in vLLM internal."""
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    def get_engine(self):
        """Return the initialized vLLM ``LLM`` instance."""
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine

    def is_engine_initialized(self):
        return self._engine_initialized

    def fp8_quantization(self, weight: torch.Tensor):
        """Quantize *weight* to row-wise FP8 and return ``(qweight.T, scale)``."""
        from vllm import _custom_ops as ops

        # quantization of rowwise torch scaled_mm.
        # weight has shape [out_dim, in_dim]
        qweight, weight_scale = ops.scaled_fp8_quant(
            weight, scale=None, use_per_token_if_dynamic=True
        )

        return qweight.t(), weight_scale

    def model_param_map(self, weight_mapper: WeightMapper):
        """Build and cache a mapping from HF-compatible names to vLLM parameters."""
        if self._model_param_map:
            return self._model_param_map
        model = self.get_underlying_model()
        param_map = {}
        for name, param in model.named_parameters():
            compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(name)
            param_map[compatible_name] = param
        self._model_param_map = param_map
        return self._model_param_map
