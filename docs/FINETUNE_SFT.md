# Alpamayo R1 SFT

**The following examples were validated on 8× H100 GPUs with 80 GB each.**

This guide explains how to run supervised fine-tuning (SFT) for the Alpamayo 1 model on the Physical AI AV dataset. The SFT scripts live under [finetune/sft](../finetune/sft/).

## Features

- [x] Stage 1: fine-tune the base VLM
- [x] Stage 2: fine-tune the expert trajectory diffusion model
- [x] Data loader support
  - PAI: [physical_ai_av](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

## Prepare dataset and checkpoint

### Download the PAI dataset (script)

You can download the full dataset, a subset by chunk, or individual components on demand.

Use the following command to download a representative slice for Alpamayo 1 (here: chunks 0–10, four cameras, egomotion):

Set your Hugging Face token first:

`export HF_TOKEN=<your Hugging Face token>`

```
python scripts/download_pai.py --chunk-ids 0-10 --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov --calibration camera_intrinsics sensor_extrinsics --labels egomotion --output-dir /path/to/pai_dataset
```

> `--chunk-ids` accepts a single id (`0`), multiple ids (`0 1`), or a range (`0-3`, which yields chunks 0, 1, and 2). Omit it (or pass `None`) to download the full dataset (on the order of ~97 TB).

### Download the checkpoint

Download the pretrained Alpamayo 1 checkpoint from [Hugging Face](https://huggingface.co/nvidia/Alpamayo-R1-10B) into a **local directory** (Stage 1 loads weights from disk, not only a Hub model id). Example:

```
huggingface-cli download nvidia/Alpamayo-R1-10B --local-dir <path/to/model>
```

Point training at that directory: set `checkpoint_path` in [ar1_base.yaml](../finetune/sft/configs/models/ar1_base.yaml), or pass `model.checkpoint_path=<path>` on the command line when you launch Stage 1 (see below).

## Run Stage 1 fine-tuning

> Alpamayo 1 uses Hydra, so you can extend or override configuration in a structured way.

> **Weights & Biases:** To log runs to W&B, ensure the `wandb` default is enabled in [sft_base.yaml](../finetune/sft/configs/sft_base.yaml) (basically uncomment the `wandb` line on top), fill in `team` and `project` in [wandb/default.yaml](../finetune/sft/configs/wandb/default.yaml) (from [wandb.ai](https://wandb.ai)), set `report_to: wandb` under `trainer` if it is not already, and have your W&B API key available when training starts.

### Data loader

After downloading PAI, set `local_dir` to the dataset root (e.g. `/path/to/pai_dataset`) and `chunk_ids` to a range string such as `"0-10"` in your Hydra config or overrides.

### Start training

Training uses a two-stage pipeline for convergence and stability.

1. **Stage 1:** Fine-tune the VLM (`base_model`) to emit discrete trajectory tokens.
2. **Stage 2:** Freeze the Stage-1 VLM and train the action expert (trajectory diffusion) for continuous trajectories.

### Hyperparameters

You can adjust settings such as `dataloader_num_workers` or the learning rate in the config as needed.

#### Stage 1

> Stage 1 fine-tunes the full VLM; enable DeepSpeed (as in the bundled config) for memory-efficient multi-GPU training.

Stage 1 trains [base_model.py](../src/alpamayo_r1/models/base_model.py) for discrete tokens.

```
torchrun --nproc_per_node 8 -m finetune.sft.train_hf --config-path pkg://finetune/sft/configs --config-name sft_stage1
```

You can override the checkpoint location, for example:

`model.checkpoint_path=<path/to/Alpamayo-R1-10B>`

This must be the same directory you created with `huggingface-cli download nvidia/Alpamayo-R1-10B …`.

Example log lines:

```
{'loss': 1.668, 'grad_norm': 0.9367678165435791, 'learning_rate': 1.2500000000000003e-08, 'epoch': 0.02}
{'loss': 1.7079, 'grad_norm': 1.1734423637390137, 'learning_rate': 2.5000000000000005e-08, 'epoch': 0.03}
{'loss': 1.641, 'grad_norm': 0.8667469620704651, 'learning_rate': 3.7500000000000005e-08, 'epoch': 0.05}
{'loss': 1.6859, 'grad_norm': 0.8352743983268738, 'learning_rate': 5.000000000000001e-08, 'epoch': 0.06}
{'loss': 1.6968, 'grad_norm': 1.1325007677078247, 'learning_rate': 6.250000000000001e-08, 'epoch': 0.08}
```

#### Stage 2

Stage 2 adds the trajectory diffusion expert and keeps the Stage-1 VLM frozen.

```
torchrun --nproc_per_node 8 -m finetune.sft.train_hf --config-path pkg://finetune/sft/configs --config-name sft_stage2 model.pretrained_model_name_or_path=/path/to/Alpamayo-R1-10B model.stage1_vlm_checkpoint_path=/path/to/stage1/output/checkpoint-xxxx
```

> model.pretrained_model_name_or_path must be the same local folder you used for Stage 1 (full base checkpoint on disk). model.stage1_vlm_checkpoint_path is your Stage 1 Trainer output, e.g. output_stage1/checkpoint-3500 (a directory containing model.safetensors.index.json and shards).

You should see a loss curve similar to:

![loss.png](loss.png)

### Evaluation

This command evaluates the Stage-2 checkpoint against the `val_dataset` in the config:

```
torchrun --nproc_per_node 8 -m finetune.sft.evaluate_hf --config-path pkg://finetune/sft/configs --config-name sft_stage2 evaluate.eval_ckpt=/path/to/stage2/output/ckpt-xxx
```

With the defaults described above, `val/metric/min_ade` should fall below 1. Example metrics:

```
val/metric/ade              2.0072
val/metric/ade/by_t=3.0     0.3970
val/metric/corner_distance  0.6632
val/metric/min_ade          0.6270
val/metric/min_ade/by_t=0.5 0.0079
val/metric/min_ade/by_t=1.0 0.0261
val/metric/min_ade/by_t=3.0 0.2008
val/metric/min_ade/by_t=5.0 0.4351
```

## Note on example metrics and loss curves

The numbers and plots in this guide are provided **for validation and comparison only**. The release model has already been trained on this data, so running the same fine-tuning recipe will **not** show a large drop in loss relative to a from-scratch scenario. We include these references so you can confirm that your setup matches a **typical** fine-tuning run (logging shape, metric magnitudes, and overall behavior), not to reproduce large pretraining-style loss decreases.
