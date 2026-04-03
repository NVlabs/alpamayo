# Alpamayo RL Post-training using Cosmos RL

This directory contains the RL post-training code for fine-tuning
Alpamayo models (VLM backbone with discrete action-token variant) using GRPO via Cosmos-RL. The code supports both [Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B) and [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B).

<p align="center">
  <img src="assets/alpamayo_rl_framework.png" alt="Alpamayo RL Framework" width="700">
</p>

## Table of contents

1. [Getting started](#getting-started)
   1. [Python environment](#1-python-environment)
   2. [Environment variables](#2-environment-variables)
   3. [Authenticate with HuggingFace](#3-authenticate-with-huggingface)
   4. [Download and prepare the Alpamayo model](#4-download-and-prepare-the-alpamayo-model)
   5. [Download a subset of the Physical AI dataset](#5-download-a-subset-of-the-physical-ai-dataset)
   6. [Launch RL training](#6-launch-rl-training)
   7. [Export the RL checkpoint for inference](#7-export-the-rl-checkpoint-for-inference)
2. [Pipeline overview](#pipeline-overview)
   1. [Architecture](#architecture)
   2. [Key files](#key-files)
   3. [Key parameters](#key-parameters)
   4. [Reward](#reward)
3. [Multi-node large-scale training](#multi-node-large-scale-training)
4. [FAQ](#faq)

## Getting started

This section walks you through a **single-node local verification run** —
from environment setup to launching a short RL training job on a small
dataset. The goal is to verify that the full pipeline (model loading,
rollout generation, reward computation, GRPO training) works end-to-end
before scaling to multi-node cluster training.

> **Hardware requirement:** The local test config requires at least **5 GPUs**, each with at least 80 GB of VRAM.

First, define your working directory (all subsequent commands reference `$YOUR_HOME`):

```bash
export YOUR_HOME="/path/to/your/workspace"
```

### 1. Python environment

```bash
export UV_CACHE_DIR="$YOUR_HOME/.cache/uv"

cd "$YOUR_HOME/alpamayo"
uv venv alpamayo_env
source alpamayo_env/bin/activate
uv sync --active --no-install-package flash-attn   # install all deps except flash-attn
uv sync --active                                   # then build flash-attn (needs torch)
```

### 2. Environment variables

Set the following once per session (or add to `~/.bashrc`):

```bash
# ── Paths ────────────────────────────────────────────────────────
export ALPAMAYO_WORKSPACE="$YOUR_HOME/alpamayo"
export ALPAMAYO_MODEL_DIR="$YOUR_HOME/alpamayo_model_converted_from_hf"
export ALPAMAYO_PAI_LOCAL_DIR="$YOUR_HOME/PAI_mini"
export ALPAMAYO_LOG_DIR="$YOUR_HOME/alpamayo_cosmos_rl_job/logs"

# ── Cache ────────────────────────────────────────────────────────
export HF_HOME="$YOUR_HOME/.cache/huggingface"

# ── Runtime ──────────────────────────────────────────────────────
export WANDB_API_KEY="<your_wandb_api_key>"
```

> **Tip:** If you hit HuggingFace Hub rate limits, set `export HF_HUB_OFFLINE=1`
> and `export TRANSFORMERS_OFFLINE=1` to force all model/tokenizer loads from
> local cache.

| Variable                 | Required    | Purpose                                                                                  |
| ------------------------ | ----------- | ---------------------------------------------------------------------------------------- |
| `ALPAMAYO_WORKSPACE`     | yes         | Root of the `alpamayo` checkout                                                          |
| `ALPAMAYO_MODEL_DIR`     | yes         | Pre-trained Alpamayo model directory (output of step 4)                                  |
| `ALPAMAYO_PAI_LOCAL_DIR` | yes         | PAI dataset root (output of step 5); read by entry scripts at runtime                    |
| `ALPAMAYO_LOG_DIR`       | yes         | Directory for Cosmos-RL logs                                                             |
| `UV_CACHE_DIR`           | recommended | uv cache location (set in step 1, before `uv venv`)                                      |
| `HF_HOME`                | recommended | HuggingFace cache location                                                               |
| `HF_HUB_OFFLINE`         | optional    | Set to `1` to skip HuggingFace Hub calls (useful for rate limits or air-gapped clusters) |
| `TRANSFORMERS_OFFLINE`   | optional    | Set to `1` alongside `HF_HUB_OFFLINE`                                                    |
| `WANDB_API_KEY`          | recommended | Weights & Biases API key; omit if using `[logging].logger = ["console"]`                 |

### 3. Authenticate with HuggingFace

The model and dataset require access to gated resources. Request access here:

- [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo 1 Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [Alpamayo 1.5 Model Weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Then authenticate:

```bash
hf auth login
```

Get your token at: https://huggingface.co/settings/tokens

### 4. Download and prepare the Alpamayo model

Convert the HuggingFace release model into a training-ready checkpoint.
By default, the script downloads
[Alpamayo 1.5 Model Weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B). Use `--alpamayo-model` to switch to
[Alpamayo 1 Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B) if needed.

```bash
cd "$ALPAMAYO_WORKSPACE"

python scripts/convert_release_config_to_training.py \
  --output-dir "$ALPAMAYO_MODEL_DIR"
```

### 5. Download a subset of the Physical AI dataset

```bash
python scripts/download_pai.py \
  --chunk-ids 3116 \
  --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --output-dir "$ALPAMAYO_PAI_LOCAL_DIR"
```

Then curate a mini subset of 16 driving clips for local RL training:

```bash
python scripts/curate_pai_samples.py \
  --clip-index-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet" \
  --chunk 3116 \
  --num-samples 16 \
  --output-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index_mini.parquet"
```

### 6. Launch RL training

Update the TOML config before launching. For local testing use
`finetune/rl/toml/alpamayo_rvla_rl_local_test.toml`.

Key fields to set:

1. `[train].output_dir`: where checkpoints and training artifacts are
   written (e.g., `$YOUR_HOME/alpamayo_cosmos_rl_job/outputs`).
2. `[policy].model_name_or_path`: set to `$ALPAMAYO_MODEL_DIR`.
3. `[policy.parallelism].dp_shard_size`: `4` for local test (1 node),
   `8` for cluster (multi-node).
4. (Optional) `[logging].logger = ["console", "wandb"]`.

**Local test (single node):** activate the installed environment, then
run from the alpamayo directory:

```bash
cd "$ALPAMAYO_WORKSPACE"
cosmos-rl \
  --config finetune/rl/toml/alpamayo_rvla_rl_local_test.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py
```

- `--policy 1 --rollout 1`: launch 1 policy replica and 1 rollout replica.
  This overrides `n_init_replicas` in the TOML config.

- Training logs are written to `$ALPAMAYO_LOG_DIR/logs_<YYYYMMDD-HHMMSS>/`
  with one file per process:

  | Log file          | Process              | What it contains                                                                                 |
  | ----------------- | -------------------- | ------------------------------------------------------------------------------------------------ |
  | `controller.log`  | Cosmos-RL controller | Rollout dispatch, reward stats per step, buffer status (`pending rollouts`), weight sync events  |
  | `policy_<i>.log`  | Policy replica *i*   | Model loading, training loss, gradient norms, checkpoint saving, per-rank data distribution      |
  | `rollout_<i>.log` | Rollout replica *i*  | vLLM engine startup, generation throughput, weight receive events, reward computation per sample |

With the default settings, you should see training reward increase and
trajectory L2 error decrease (reward improved from -0.28 to -0.21, and trajectory L2 decreased from 1.66 to 1.34). The local test finishes within ~10 minutes on a single 8×GPU (H100) node:

<p align="center">
  <img src="assets/local_training_reward_curves.png" alt="Local training reward curves" width="700">
</p>

### 7. Export the RL checkpoint for inference

Cosmos-RL saves checkpoints as per-rank PyTorch files (`model_rank_<r>.pth`)
under `<output_dir>/checkpoints/step_<N>/policy/`. These files contain
DTensor shards and cannot be loaded directly with
`ReasoningVLA.from_pretrained()`.

To convert a policy checkpoint into a standard HuggingFace checkpoint
directory:

```bash
cd "$ALPAMAYO_WORKSPACE"

python scripts/convert_cosmos_rl_checkpoint.py \
  --cosmos-policy-ckpt "$YOUR_HOME/alpamayo_cosmos_rl_job/outputs/checkpoints/step_<N>/policy" \
  --base-hf-ckpt "$ALPAMAYO_MODEL_DIR" \
  --output-dir "$YOUR_HOME/alpamayo_cosmos_rl_job/exported_model"
```

- `--cosmos-policy-ckpt`: path to the Cosmos-RL policy checkpoint
  directory (contains `model_rank_*.pth` files).
- `--base-hf-ckpt`: the training-ready checkpoint directory produced by
  step 4 (`$ALPAMAYO_MODEL_DIR`). Config, tokenizer, and processor files
  are copied from here.
- `--output-dir`: where to write the exported HF checkpoint.

The exported checkpoint can then be loaded for inference:

```python
from reasoning_vla.base_model import RLWrapperReasoningVLA

model = RLWrapperReasoningVLA.from_pretrained(
    "$YOUR_HOME/alpamayo_cosmos_rl_job/exported_model"
)
```

For a complete end-to-end example (loading data, running inference,
visualizing reasoning and trajectories and chain-of-thought), see
[`notebooks/rl_checkpoint_inference.ipynb`](notebooks/rl_checkpoint_inference.ipynb).

## Pipeline overview

### Architecture

Alpamayo RL post-training is built on
[Cosmos-RL](https://github.com/NVIDIA/Cosmos-RL), a scalable
reinforcement learning framework for Physical AI workloads.

Each job consists of:

- one or more **policy replicas**, which train the model, and
- one or more **rollout replicas**, which run inference to generate rollout samples.

These components are coordinated by a central **cosmos-controller**,
which dispatches rollouts, collects rewards, manages the training buffer,
and periodically synchronizes the latest policy weights to the rollout replicas.
This design enables asynchronous RL training at scale, while keeping rollout
generation and policy optimization loosely coupled.

We use
[GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300)
as the RL algorithm.

### Key files

| Path                                                             | Purpose                                                                                                                                                                             |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `toml/alpamayo_rvla_rl_local_test.toml`                          | Cosmos-RL TOML config (local test) -- controls replica counts, parallelism, optimizer, rollout, reward weights, logging, and checkpointing. See Cosmos-RL docs for the full schema. |
| `models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py` | Entry script passed to `cosmos-rl` -- registers model, rollout, trainer, and reward                                                                                                 |
| `hydra_configs/alpamayo1_rvla_rl_pai.yaml`                       | Hydra config for PAI dataset and preprocessing (see also `alpamayo1_5_rvla_rl_pai.yaml` for Alpamayo 1.5)                                                                           |
| `launcher.py`                                                    | Shared launch logic that initializes state and calls the Cosmos worker                                                                                                              |
| `rewards/aggregated_reward.py`                                   | Demo reward implementation (see [Reward](#reward) below)                                                                                                                            |
| `../../scripts/convert_cosmos_rl_checkpoint.py`                  | Converts a Cosmos-RL policy checkpoint into a HuggingFace checkpoint directory (see [step 7](#7-export-the-rl-checkpoint-for-inference))                                            |

### Key parameters

| Parameter (TOML path)                 | Default (local test) | Meaning                                                                                                                                   |
| ------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `policy.parallelism.n_init_replicas`  | 1                    | Number of policy replicas. Each replica is an independent FSDP training worker. More replicas = larger global batch.                      |
| `policy.parallelism.dp_shard_size`    | 4                    | GPUs per policy replica (FSDP sharding degree). `n_init_replicas × dp_shard_size` = total policy GPUs.                                    |
| `rollout.parallelism.n_init_replicas` | 1                    | Number of rollout replicas. Each replica runs a vLLM engine that generates completions. Scale these to match policy consumption speed.    |
| `train.train_batch_per_replica`       | 48                   | Training samples consumed per policy replica per step. **Global batch / step** = `policy.n_init_replicas × train_batch_per_replica`.      |
| `rollout.batch_size`                  | 2                    | Prompts sent to a rollout replica in one batch.                                                                                           |
| `rollout.n_generation`                | 12                   | Completions generated per prompt (the "group" in GRPO). Each prompt produces `n_generation` candidate rollouts that are ranked by reward. |
| `train.sync_weight_interval`          | 2                    | Sync latest policy weights to rollout replicas every *N* training steps. Lower = fresher rollouts but more communication overhead.        |

**Dataloading acceleration.** Physical AI training samples are large.
The default Cosmos-RL pipeline independently loads and preprocesses the
same samples per GPU rank on a node, wasting both I/O bandwidth and CPU
time. The node-level prefetch server (`prefetch/server.py`) fetches and
preprocesses samples ahead of time, then shares the results with all
local ranks via shared memory. This can reduce per-step policy iteration
time significantly (e.g. 44 s → 5 s).

| Parameter (TOML path)                  | Default (local test) | Meaning                                                                                                                                                              |
| -------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `custom.alpamayo.prefetch.capacity`    | 16                   | Cache size (number of samples). Set to `train_batch_per_replica × replicas_per_node`. `<= 0` disables prefetch (falls back to synchronous per-rank dataset loading). |
| `custom.alpamayo.prefetch.num_workers` | 5                    | Worker threads that fetch and preprocess samples in the background.                                                                                                  |

### Reward

The included reward function
([`rewards/aggregated_reward.py`](rewards/aggregated_reward.py))
is a **demo implementation**. It grades each rollout sample on two axes
and combines them into a single scalar:

| Component   | How it is computed                                                                 | Weight key in TOML |
| ----------- | ---------------------------------------------------------------------------------- | ------------------ |
| **ADE**     | Average Displacement Error (L2) between predicted and ground-truth trajectory (XY) | `traj_l2_weight`   |
| **Comfort** | Fraction of timesteps within comfort bounds (acceleration, jerk, yaw rate, etc.)   | `comfort_weight`   |

The reward uses a gated structure: if ADE exceeds a threshold (default 3.0 m),
the reward is clamped to −1. Otherwise it is a weighted combination of
the normalized ADE penalty and the comfort score. Weights are configured
under `[custom.alpamayo.reward]` in the TOML.

> **Note:** This reward is provided as a starting point and only scores
> the trajectory portion of the rollout. Alpamayo 1.5 was RL
> post-trained with reasoning reward applied to the
> chain-of-causation portion of the generation. You can implement
> your own reward -- including one that grades both trajectory and
> reasoning -- by following the same Cosmos-RL reward interface and
> registering it in the entry script. See the FAQ entry
> *"Can I use RL to post-train the reasoning (chain-of-thought)
> generations?"* below for details on extracting the reasoning text
> from a rollout.

## Multi-node large-scale training

The local test above runs on a single node. To scale to multi-node
cluster training, two parameters in the TOML **must** be changed:

1. **`policy.parallelism.dp_shard_size`** — set to 8. This controls FSDP
   sharding: `n_policy_replicas × dp_shard_size` = total policy GPUs.

2. **`train.train_policy.data_dispatch_as_rank_in_mesh`** — set to
   `true`. This enables rank-based data dispatch so each policy replica
   consumes a stable, non-overlapping shard of the dataset. Without this,
   multiple replicas may train on duplicate samples. This flag is also
   required for running the data preloading.

It is also recommended to tune the following parameters for your
specific setting. The values below are what we used to post-train
Alpamayo 1.5:

| Parameter                             | Local test | Cluster training |
| ------------------------------------- | ---------- | ---------------- |
| `policy.parallelism.n_init_replicas`  | 1          | 64               |
| `rollout.parallelism.n_init_replicas` | 1          | 128              |
| `train.train_batch_per_replica`       | 48         | 40               |
| `train.optm_lr`                       | 2e-6       | 2e-6             |
| `train.sync_weight_interval`          | 2          | 5                |
| `rollout.batch_size`                  | 2          | 6                |
| `rollout.n_generation`                | 12         | 12               |
| `custom.alpamayo.prefetch.capacity`   | 16         | 128              |

This gives a global batch of 64 × 40 = **2560** samples per training
step, with 512 policy GPUs and 128 rollout GPUs (640 GPUs total, 80
nodes). For SLURM launch instructions, see the
[Cosmos-RL multi-node documentation](https://nvidia-cosmos.github.io/cosmos-rl/multinodes/overview.html).

> **Tip:** Start with a moderate scale (e.g., 4 policy replicas,
> 8 rollout replicas) and monitor `pending rollouts` in
> `controller.log` before scaling up. See the FAQ entry *"How to
> balance policy replicas and rollout replicas?"* for tuning guidance.

## FAQ

<details>
<summary><strong>What exactly is being RL post-trained?</strong></summary>

This code RL post-trains the **VLM backbone** of the released Alpamayo
models (ReasoningVLA). In this pathway, the VLM autoregressively generates
text and discrete trajectory tokens. The action expert
head (flow-matching-based continuous actions) is **not** trained
by this RL pipeline. RL post-training for the **action expert pathway** will come in a future release.

</details>

<details>
<summary><strong>Can I post-train Alpamayo 1.5?</strong></summary>

Yes. Both [Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B) and
[Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) are supported.

You need to change two things:

1. Point `ALPAMAYO_MODEL_DIR` to the converted checkpoint of the model you want to use.
2. In the entry script, set `hydra_config_name` to match your model:
   - Alpamayo 1.5: `"alpamayo1_5_rvla_rl_pai"`
   - Alpamayo 1: `"alpamayo1_rvla_rl_pai"`

</details>

<details>
<summary><strong>What can I do with this code and how to use the RL checkpoint?</strong></summary>

- **Improve driving behavior** — define reward functions that target
  trajectory accuracy, comfort, safety, or other driving metrics.
- **Improve reasoning and scene understanding** — add rewards that grade
  the text portion of the model's output, steering
  the model toward better situational awareness and decision-making.
- **Train on your own driving data** — prepare a dataset in the PAI
  format, point the config to it, and run RL on your own scenarios.
- **Using the RL checkpoint** — the exported checkpoint contains **only
  the VLM backbone weights** (since RL only trains the VLM backbone). It
  can be loaded into both Alpamayo 1 and Alpamayo 1.5 (from the
  Alpamayo 1.5 directory with minor target renaming in the checkpoint's
  model config) with non-strict weight loading for training the action
  expert model with SFT. Note that the action expert weights will be
  randomly initialized.

</details>

<details>
<summary><strong>How do I replace the reward function?</strong></summary>

Implement your own reward following the Cosmos-RL reward interface and
register it in the entry script
(`finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py`).
See [`finetune/rl/rewards/aggregated_reward.py`](rewards/aggregated_reward.py)
for the expected signature and return format.

</details>

<details>
<summary><strong>Can I use RL to post-train the reasoning (chain-of-thought) generations?</strong></summary>

Yes. The model generates reasoning text before `<|cot_end|>` and trajectory
tokens after `<|traj_future_start|>`. Both are part of the single rollout
completion string (`to_be_evaluated`) passed to the reward function. The
current default reward (`aggregated_reward.py`) only scores the trajectory
portion (ADE + comfort), but you can extend it to also grade the reasoning
trace.

To extract the reasoning text from a rollout completion:

```python
reasoning_text = to_be_evaluated.split("<|cot_end|>")[0]
```

You can then score it with a custom reasoning reward (e.g., an LLM-based
grader, rule-based checks, or a learned reward model). We will soon release
reasoning labels and a corresponding reasoning reward function.

</details>

<details>
<summary><strong>What is the recommended number of GPUs?</strong></summary>

It depends on the dataset size. A larger global batch size
(`policy.parallelism.n_init_replicas` × `train.train_batch_per_replica`)
generally gives better RL performance.

As a rough guide:

| Scale                  | Policy                                              | Rollout                           | `train_batch_per_replica` | `rollout.batch_size` × `n_generation` | Global batch / step |
| ---------------------- | --------------------------------------------------- | --------------------------------- | ------------------------- | ------------------------------------- | ------------------- |
| Local test (1 node)    | 4 GPUs, 1 replica, `dp_shard_size=4`                | 1 GPU, 1 replica                  | 48                        | 2 × 12 = 24                           | 48                  |
| Large scale (80 nodes) | 64 nodes (512 GPUs): 64 replicas, `dp_shard_size=8` | 16 nodes (128 GPUs): 128 replicas | 40                        | 6 × 12 = 72                           | 2560                |

- **Global batch / step** = `n_init_replicas` × `train_batch_per_replica`.
- **Policy GPUs** determine training speed and global batch size. The model
  is sharded across GPUs via FSDP (`dp_shard_size`).
- **Rollout GPUs** determine data generation throughput. Scale rollout
  replicas to match policy consumption speed.
- **Keep rollout ≈ policy speed**: if rollout is too fast, data becomes
  stale; if too slow, policy idles. Monitor `pending rollouts` in the
  controller log.

</details>

<details>
<summary><strong>What is the recommended workflow for new reward, data, and model?</strong></summary>

1. **Overfit on 1 sample.** Curate a single training sample and run RL
   locally. Verify the reward increases. This confirms that
   the reward function, data pipeline, and model are wired correctly.

2. **Overfit on a small set (~16–32 samples) on one node.** Check that
   the reward improves across multiple epochs.
   Use this stage to tune reward weights, learning rate, and
   `n_generation`. Watch for rollout/policy speed imbalance (see
   the FAQ entry *"How to balance policy replicas and rollout replicas?"*
   below).

3. **Scale to multi-node.** Increase policy and rollout replicas. Monitor
   `pending rollouts` and `weight_version` gap to ensure the system is
   balanced. Start with a moderate global batch size (e.g., 320) and scale
   up if reward variance is too high.

4. **Iterate on the reward function.** RL will optimize whatever the
   reward measures. If model behavior is not improving as expected,
   revisit the reward design before scaling further.

</details>

<details>
<summary><strong>How to balance policy replicas and rollout replicas?</strong></summary>

Rollout replicas generate data asynchronously while policy replicas consume
it for training. The two sides must run at roughly the same throughput.

**Rollout too fast (most common):** completed rollouts pile up in the
controller buffer. By the time the policy trains on them, its weights have
moved far beyond the weights that generated those rollouts (large
`weight_version` gap → off-policy data → degraded training quality). In the
extreme case, rollout workers finish all epochs while training is only
halfway done.

**Rollout too slow:** the policy idles waiting for data; GPU utilization
drops.

**How to diagnose** — check these metrics in the controller log:

1. **`pending rollouts` grows monotonically** (rollout too fast): the buffer
   never drains.
2. **Rollout ends early**: `[Controller] All rollouts have ended` appears
   while training is far from `total_steps`.
3. **`pending rollouts` frequently drops to zero** (rollout too slow): the
   policy waits for the next rollout batch.

**Example — rollout too fast:**

```text
# controller.log — buffer grows every step, never drains
Stat: samples=  24  pending=   24          ← start
Stat: samples= 600  pending=  264          ← growing
Stat: samples=1200  pending=  552          ← still growing
Stat: samples=2400  pending=  984          ← rollout outpacing policy
[Controller] All rollouts have ended … 1104 remaining rollouts
# training is at step 37/60 — 23 more steps will use stale rollouts
```

**Example — well balanced:**

```text
# controller.log — buffer stays small, oscillates
Stat: samples=  48  pending=   24
Stat: samples= 600  pending=   48          ← buffer stays low
Stat: samples=1200  pending=   72
Stat: samples=2400  pending=   48          ← not accumulating
```

**Tuning knobs (from fastest to try):**

| Knob                                          | Effect                                                | When to use                                 |
| --------------------------------------------- | ----------------------------------------------------- | ------------------------------------------- |
| Enable prefetch (`prefetch.capacity > 0`)     | Reduces policy iteration time (e.g. 44 s → 12 s)      | Always recommended; biggest single win      |
| Reduce `rollout.batch_size` or `n_generation` | Slows rollout throughput                              | When rollout is much faster than policy     |
| Add rollout replicas                          | Speeds up rollout throughput                          | When policy idles waiting for data          |
| Add policy replicas (`n_init_replicas`)       | Speeds up policy (more parallel training)             | When rollout outpaces policy at large scale |
| Increase `dp_shard_size`                      | Speeds up per-step training via more data parallelism | When each step is too slow                  |
| Reduce `epoch` or set `max_num_steps`         | Fewer total rollouts to generate                      | When rollout finishes far before training   |

**Target state:** `pending rollouts` stays roughly stable. In a healthy
large-scale job (64 policy replicas, 128 rollout replicas, global
batch = 2560), the buffer typically holds 4× the global batch size. The `weight_version` gap within each
training batch should stay within a few multiples of
`sync_weight_interval`.

</details>
