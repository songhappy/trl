GRPO on Intel XPU (Distributed, Script-Only)
============================================

Overview
--------
This guide shows how to run GRPO (Group Relative Policy Optimization)
training on Intel XPU in distributed mode using upstream PyTorch only. 
Everything is script-based; no notebooks required.

Tested configuration:
- launcher: HuggingFace accelerate with MULTI_XPU distributed type
- script: examples/rl/grpo_lora_xpu.py
- model: Qwen/Qwen2.5-7B-Instruct
- dataset: AI-MO/NuminaMath-TIR
- precision: bf16 + LoRA


What is GRPO
------------
GRPO samples a group of completions per prompt, scores them with reward
functions, and computes relative advantages within each group. Completions
that score above the group mean get reinforced; those below get penalized.
No value model is needed — advantage is derived from group statistics.

Key requirement: reward functions must produce varying scores across
completions in a group. If all completions get the same reward (e.g. all
zero), advantages collapse to zero and training produces zero loss.


Why LoRA
--------
LoRA inserts small trainable adapters into the model while freezing base
weights. This reduces memory usage enough to run 7B models on XPU with
bf16 precision, without needing quantization.


Files
-----
- examples/rl/grpo_lora_xpu.py — distributed GRPO training script
- examples/accelerate_configs/multi_xpu.yaml — accelerate config for XPU
- examples/rl/loss.py — parse training logs and plot loss curves


Prerequisites
-------------
1. PyTorch 2.4+ with native torch.xpu support
2. Python packages:
   ```bash
   pip install trl[peft] accelerate datasets math_verify
   ```
3. A local model directory (download on a node with internet):
   ```bash
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$MODEL_PATH')
   "
   ```
4. A local dataset directory (download on a node with internet):
   ```bash
   python -c "
   from datasets import load_dataset
   ds = load_dataset('AI-MO/NuminaMath-TIR')
   ds.save_to_disk('$DATA_PATH')
   "
   ```

Replace `$MODEL_PATH` and `$DATA_PATH` with your own local directories.


How to run (distributed)
-------------------------
Set your paths and launch from the repo root:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
DATA_PATH=/path/to/NuminaMath-TIR
LOG_FILE=grpo_run.log

accelerate launch --config_file examples/accelerate_configs/multi_xpu_grpo.yaml \
    --num_processes 4 \
    examples/rl/grpo_lora_xpu.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --max_steps 500 \
    --logging_steps 10 \
    --max_completion_length 256 \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    2>&1 | tee $LOG_FILE
```

To run in the background with nohup:

```bash
nohup accelerate launch --config_file examples/accelerate_configs/multi_xpu.yaml \
    --num_processes 4 \
    examples/rl/grpo_lora_xpu.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --max_steps 500 \
    --logging_steps 10 \
    --max_completion_length 256 \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    > $LOG_FILE 2>&1 &
```

Monitor and plot:

```bash
tail -f $LOG_FILE
python examples/rl/loss.py --log $LOG_FILE
```


Key differences from the CUDA notebook flow
--------------------------------------------
| CUDA notebook                  | XPU script                        |
|-------------------------------|-----------------------------------|
| bitsandbytes QLoRA (4-bit)    | LoRA + bf16 (no quantization)     |
| paged_adamw_8bit optimizer    | adamw_torch                       |
| liger-kernel                  | removed (CUDA-only)               |
| device_map="auto"             | accelerate handles placement      |
| MULTI_GPU distributed type    | MULTI_XPU distributed type        |
| notebook workflow             | script-only                       |


Migration details: notebook to XPU script
------------------------------------------
The following settings are kept identical to the notebook
(examples/notebooks/grpo_trl_lora_qlora.ipynb):

  Identical settings:
  - SYSTEM_PROMPT text
  - dataset: AI-MO/NuminaMath-TIR, split train[:5%]
  - make_conversation logic and column removal
  - LoRA: r=32, alpha=32
  - target_modules: q/k/v/o/gate/up/down_proj
  - learning_rate: 2e-5
  - max_steps: 500
  - per_device_train_batch_size: 8
  - max_completion_length: 256
  - num_generations: 8
  - logging_steps: 10
  - attn_implementation: sdpa
  - log_completions: False

  Changed for XPU (intentional):
  - optim: paged_adamw_8bit -> adamw_torch
    reason: bitsandbytes not available on XPU
  - use_liger_kernel: True -> removed
    reason: liger-kernel is CUDA-only
  - model dtype: float32 + 4-bit quantization -> bfloat16
    reason: no bitsandbytes quantization on XPU; bf16 native support
  - report_to: "trackio" -> "none"
    reason: offline compute node, no HF Hub access
  - push_to_hub: True -> False
    reason: same as above
  - gradient_checkpointing: not set -> True
    reason: reduces memory on XPU, helpful without quantization
  - reward_funcs: same as notebook
    think_format_reward + reasoning_accuracy_reward


TRL library changes for XPU
----------------------------
On branch 'grpo', three files were modified:

trl/trainer/grpo_trainer.py:
  Added "MULTI_XPU" to distributed_type checks at 3 locations so
  device_map=None is set correctly for multi-XPU training.

trl/generation/vllm_generation.py:
  Replaced torch.cuda.current_device() with XPU-aware version using
  is_torch_xpu_available().


Known warnings (safe to ignore)
-------------------------------
- CCL_WARN about MPI launcher variables:
  expected when not using MPI launcher; XCCL falls back to ATL/OFI
- DDP find_unused_parameters=True warning:
  performance advisory, not a correctness issue
- generation_config + disable_compile deprecation:
  warning only; does not affect training


Planned next steps
------------------
1. Validate convergence on longer runs (500+ steps) with non-zero loss.
2. Add multi-node accelerate config for XPU clusters.
3. Test vLLM XPU backend for faster generation during GRPO training.
4. Profile throughput (tokens/s) and memory vs CUDA baseline.
5. Upstream MULTI_XPU support changes to TRL repository.
