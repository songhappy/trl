GRPO on Intel XPU (Distributed, Script-Only)
============================================

Overview
--------
This guide shows how to run GRPO (Group Relative Policy Optimization)
training on Intel XPU in distributed mode using upstream PyTorch only. 
Everything is script-based; no notebooks required, it's migrated from examples/notebooks/grpo_trl_lora_qlora.ipynb.

Tested configuration:
- launcher: HuggingFace accelerate with MULTI_XPU distributed type
- script: examples/rl/grpo_lora_train.py
- model: Qwen/Qwen2.5-7B-Instruct
- dataset: AI-MO/NuminaMath-TIR
- precision: bf16 + LoRA
- hardware: 8x Intel XPU (single node)


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
- examples/rl/grpo_lora_train.py — distributed GRPO training script
- examples/rl/grpo_infer.py — load fine-tuned model and run inference
- examples/rl/log.py — parse training metrics (JSONL) and plot loss/reward curves
- examples/rl/run_grpo.sh — end-to-end script: train, plot, and infer
- examples/accelerate_configs/multi_xpu_grpo.yaml — accelerate config for XPU


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
The easiest way is to use the all-in-one script from the repo root:

```bash
cd /home/guoqiong/git/trl
nohup bash examples/rl/run_grpo.sh /home/guoqiong/models/Qwen2.5-7B-Instruct /home/guoqiong/data/NuminaMath-TIR 8 &
```

This runs training (500 steps on 8 XPUs), plots metrics, and runs inference.

To run steps manually:

```bash
MODEL_PATH=/home/guoqiong/models/Qwen2.5-7B-Instruct
DATA_PATH=/home/guoqiong/data/NuminaMath-TIR

# Training
accelerate launch --config_file examples/accelerate_configs/multi_xpu_grpo.yaml \
    --num_processes 8 \
    examples/rl/grpo_lora_train.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --output_dir examples/rl/xpu-grpo-output \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 100 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 2 \
    --num_generations 4 \
    --learning_rate 5e-6
```

Monitor progress:

```bash
tail -f examples/rl/xpu-grpo-output/run.log
```


Plot training curves
--------------------
After training (or if training was interrupted and checkpoints exist):

```bash
cd examples/rl
python log.py --log xpu-grpo-output/training_metrics.jsonl --output xpu-grpo-output/curves.png
```

This generates a PNG with loss and reasoning accuracy reward curves.


Inference
---------
After training completes, use `grpo_infer.py` to load the LoRA
adapter and run inference on XPU. Use the latest checkpoint if
training was interrupted.

Run on dataset test samples:

```bash
cd examples/rl
python grpo_infer.py \
    --base_model /home/guoqiong/models/Qwen2.5-7B-Instruct \
    --adapter xpu-grpo-output/checkpoint-300 \
    --dataset_name /home/guoqiong/data/NuminaMath-TIR \
    --num_samples 5
```

Ask a single question:

```bash
python grpo_infer.py \
    --base_model /home/guoqiong/models/Qwen2.5-7B-Instruct \
    --adapter xpu-grpo-output/checkpoint-300 \
    --question "What is the sum of all prime numbers less than 20?"
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


Planned next steps
------------------
1. Validate convergence and reward on longer runs (500+ steps) with non-zero loss.
2. Add multi-node accelerate config for XPU clusters.
3. Test vLLM XPU backend for faster generation during GRPO training.
4. Profile throughput (tokens/s) and memory vs CUDA baseline.
5. Upstream MULTI_XPU support changes to TRL repository.
