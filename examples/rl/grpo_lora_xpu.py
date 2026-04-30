#!/usr/bin/env python
# GRPO with LoRA on Intel XPU — distributed version
#
# Usage (single device):
#   python examples/rl/grpo_lora_xpu.py --model_name_or_path <model_path> --dataset_name <dataset_path>
#
# Usage (multi-XPU distributed):
#   accelerate launch --config_file examples/accelerate_configs/multi_xpu.yaml \
#       examples/rl/grpo_lora_xpu.py --model_name_or_path <model_path> --dataset_name <dataset_path>

import argparse

from datasets import load_dataset
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer
from trl.rewards import think_format_reward, reasoning_accuracy_reward


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed strictly within <think> and </think> tags. "
    "After closing </think>, the assistant MUST provide the final answer in plain text."
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="GRPO LoRA training on XPU")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to local model or HuggingFace model ID")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Path to local dataset (saved with save_to_disk) or HuggingFace dataset ID")
    parser.add_argument("--dataset_split", type=str, default="train[:5%]")
    parser.add_argument("--from_disk", action="store_true", default=True,
                        help="Load dataset from local disk (saved with save_to_disk)")
    parser.add_argument("--no_from_disk", action="store_false", dest="from_disk",
                        help="Load dataset from HuggingFace Hub instead of local disk")
    parser.add_argument("--output_dir", type=str, default="./xpu-grpo-output")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    if args.from_disk:
        from datasets import load_from_disk, DatasetDict
        full_dataset = load_from_disk(args.dataset_name)
        # Apply split selection (e.g. "train[:5%]")
        split_name = args.dataset_split.split("[")[0] if "[" in args.dataset_split else args.dataset_split
        subset = args.dataset_split.split("[")[1].rstrip("]") if "[" in args.dataset_split else None
        dataset = full_dataset[split_name]
        if subset:
            # Parse percentage or count
            if "%" in subset:
                pct = int(subset.replace(":", "").replace("%", ""))
                n = int(len(dataset) * pct / 100)
            else:
                n = int(subset.replace(":", ""))
            dataset = dataset.select(range(n))
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns(["messages", "problem"])

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Training config — pass model as string so accelerate handles device placement
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        report_to="none",
        log_completions=False,
        push_to_hub=False,
        model_init_kwargs={
            "attn_implementation": "sdpa",
            "dtype": "bfloat16",
        },
    )

    # Create trainer — model passed as string for distributed compatibility
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=[think_format_reward, reasoning_accuracy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
