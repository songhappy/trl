# TRL SFT with Different Parallelism Strategies

This document describes how to run TRL Supervised Fine-Tuning (SFT) with LoRA using different parallelism strategies on Intel XPU (Aurora). All configurations use **FSDP2** as the base distributed framework and vary the parallelism dimensions via Accelerate config files.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Common Configuration](#common-configuration)
4. [Parallelism Strategies](#parallelism-strategies)
   - [1. FSDP Only (Data Parallel)](#1-fsdp-only-data-parallel)
   - [2. Tensor Parallelism (TP)](#2-tensor-parallelism-tp)
   - [3. Context Parallelism (CP)](#3-context-parallelism-cp)
   - [4. 3D Parallelism (DP + TP + CP)](#4-3d-parallelism-dp--tp--cp)
5. [Parallelism Dimension Summary](#parallelism-dimension-summary)

---

## Overview

The four primary parallelism scripts (`run_trl_sn_a_lora_dp.sh`, `run_trl_sn_a_lora_tp.sh`, `run_trl_sn_a_lora_cp.sh`, `run_trl_sn_a_lora_3d.sh`) are **self-contained PBS scripts**. Each script handles environment setup, xpu-smi monitoring, training launch via `accelerate launch`, and post-training analysis — all inline.

| Strategy | Script | Accelerate Config | 
|---|---|---|---|
| FSDP only (DP) | `run_trl_sn_a_lora_dp.sh` | `fsdp2_lora_dp.yaml` | 
| Tensor Parallelism | `run_trl_sn_a_lora_tp.sh` | `fsdp2_lora_tp.yaml` | 
| Context Parallelism | `run_trl_sn_a_lora_cp.sh` | `fsdp2_lora_cp.yaml` | 
| 3D Parallelism | `run_trl_sn_a_lora_3d.sh` | `fsdp2_lora_3d.yaml` | 

---

## Architecture

Each parallelism script follows the same structure:

```
Standalone script (e.g. run_trl_sn_a_lora_dp.sh)
  │
  ├── PBS headers (queue, walltime, nodes, account)
  ├── Environment setup (conda activate trl2, modules, env vars)
  ├── xpu-smi monitoring (GPU memory/utilization dump, background process)
  ├── accelerate launch → trl.scripts.sft_tokens
  ├── Post-training: plot_trl_loss.py  (loss curves)
  └── Post-training: memo_perf_report.py  (performance summary)
```

All four scripts share identical environment setup, training hyperparameters, and post-training steps; they differ only in:
- `RUN_NAME` — used for log file and metrics file naming
- Accelerate config file — controls parallelism dimensions
- `OUTPUT_DIR` — checkpoint output directory

---

## Common Configuration

The following settings are shared across all four parallelism scripts.

### Environment

- **Conda environment:** `trl2`
- **Training module:** `trl.scripts.sft_tokens` (launched via `accelerate launch`)

### Environment Variables

| Variable | Value | Description |
|---|---|---|
| `FI_PROVIDER` | `tcp` | Fabric interface provider (single-node) |
| `FI_TCP_IFACE` | `lo` | Network interface for communication |
| `CCL_ATL_TRANSPORT` | `ofi` | Intel CCL transport layer |
| `CCL_ATL_OFI_PROVIDER` | `tcp` | OFI provider for CCL |
| `TORCH_CPP_LOG_LEVEL` | `WARNING` | PyTorch C++ logging level |
| `TORCH_DISTRIBUTED_DEBUG` | `OFF` | Distributed debug mode |
| `TORCH_DISTRIBUTED_CHECKPOINT_THREAD_COUNT` | `1` | Avoid multithreaded writes on network FS |
| `HF_HUB_OFFLINE` | `1` | Force offline mode (no downloads) |
| `TRANSFORMERS_OFFLINE` | `1` | Force offline mode |
| `HF_DATASETS_OFFLINE` | `1` | Force offline mode |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Model | `Meta-Llama-3.1-8B-Instruct` |
| Max Length | `2048` |
| Epochs | `20` |
| Per-device Batch Size | `2` |
| Gradient Accumulation Steps | `8` |
| Learning Rate | `2.0e-4` |
| Packing | Enabled |
| Pad to Multiple Of | `16` |
| Gradient Checkpointing | Enabled |
| Dataset | `trl-lib/Capybara` |
| LoRA rank (`lora_r`) | `32` |
| LoRA alpha (`lora_alpha`) | `16` |
| Mixed Precision | `bf16` |
| Eval Strategy | `steps` (every 10 steps) |
| Logging Steps | `10` |
| Save Strategy | `steps` (every 500 steps) |
| Report To | `none` |

### Sample Training Command

```bash
accelerate launch --config_file="${SCRIPT_DIR}/../accelerate_configs/fsdp2_lora_dp.yaml" \
    -m trl.scripts.sft_tokens \
    --model_name_or_path "${MODEL_PATH}" \
    --max_length 2048 \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 20 \
    --packing \
    --pad_to_multiple_of 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 10 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --report_to none \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir "${OUTPUT_DIR}"
```

### FSDP2 Base Settings (shared across all configs)

```yaml
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_reshard_after_forward: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
```

---

## Parallelism Strategies

All strategies use **8 processes on 1 node** (`num_processes: 8`, `num_machines: 1`). The parallelism is controlled by four dimensions in the Accelerate config:

- **`dp_replicate_size`** — Data-parallel replication (DDP-style)
- **`dp_shard_size`** — Data-parallel sharding (FSDP-style)
- **`tp_size`** — Tensor parallelism
- **`cp_size`** — Context (sequence) parallelism

The product of all dimensions must equal `num_processes`:
$$dp\_replicate \times dp\_shard \times tp \times cp = num\_processes$$

---

### 1. FSDP Only (Data Parallel)

**Script:** `run_trl_sn_a_lora_dp.sh`
**Config:** `fsdp2_lora_dp.yaml`

All 8 GPUs used for FSDP sharding — the standard data-parallel approach.

```yaml
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 8     # All 8 GPUs for FSDP sharding
  parallelism_config_cp_size: 1
  parallelism_config_tp_size: 1
```

**Dimension breakdown:** $1 \times 8 \times 1 \times 1 = 8$

**When to use:** Default choice. Best throughput for models that fit in memory with sharding alone.

---

### 2. Tensor Parallelism (TP)

**Script:** `run_trl_sn_a_lora_tp.sh`
**Config:** `fsdp2_lora_tp.yaml`

4-way tensor parallelism with 2-way FSDP sharding.

```yaml
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 2     # 2-way FSDP sharding
  parallelism_config_cp_size: 1
  parallelism_config_tp_size: 4           # 4-way tensor parallelism
```

**Dimension breakdown:** $1 \times 2 \times 1 \times 4 = 8$

**When to use:** For very large models where individual layers don't fit on one GPU. Splits layer weights across GPUs.

---

### 3. Context Parallelism (CP)

**Script:** `run_trl_sn_a_lora_cp.sh`
**Config:** `fsdp2_lora_cp.yaml`

All 8 GPUs used for context (sequence) parallelism — no FSDP sharding.

```yaml
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 1     # No FSDP sharding
  parallelism_config_cp_size: 8           # All 8 GPUs for context parallelism
  parallelism_config_tp_size: 1
```

**Dimension breakdown:** $1 \times 1 \times 8 \times 1 = 8$

**When to use:** For very long sequences that don't fit in a single GPU's memory. Splits the sequence dimension across GPUs.

---

### 4. 3D Parallelism (DP + TP + CP)

**Script:** `run_trl_sn_a_lora_3d.sh`
**Config:** `fsdp2_lora_3d.yaml`

Combines all three parallelism dimensions: 2-way FSDP sharding, 2-way tensor parallelism, and 2-way context parallelism.

```yaml
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 2     # 2-way FSDP sharding
  parallelism_config_cp_size: 2           # 2-way context parallelism
  parallelism_config_tp_size: 2           # 2-way tensor parallelism
```

**Dimension breakdown:** $1 \times 2 \times 2 \times 2 = 8$

**When to use:** For the largest models with long sequences. Combines benefits of all strategies at the cost of more complex communication patterns.

---

## Parallelism Dimension Summary

| Config | dp_replicate | dp_shard | tp | cp | Total |
|---|:---:|:---:|:---:|:---:|:---:|
| `fsdp2_lora_dp.yaml` | 1 | **8** | 1 | 1 | 8 |
| `fsdp2_lora_tp.yaml` | 1 | 2 | **4** | 1 | 8 |
| `fsdp2_lora_cp.yaml` | 1 | 1 | 1 | **8** | 8 |
| `fsdp2_lora_3d.yaml` | 1 | **2** | **2** | **2** | 8 |

---