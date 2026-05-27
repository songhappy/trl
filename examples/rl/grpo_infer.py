#!/usr/bin/env python
# Load a GRPO fine-tuned LoRA model and run inference on XPU
#
# Usage:
#   python examples/rl/grpo_infer.py \
#       --base_model <base_model_path> \
#       --adapter <output_dir_from_training> \
#       --dataset_name <dataset_path>

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from grpo_lora_train import SYSTEM_PROMPT, make_conversation


def run_inference(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference with GRPO fine-tuned LoRA model on XPU")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model (same model used for training)")
    parser.add_argument("--adapter", type=str, default="./xpu-grpo-output",
                        help="Path to LoRA adapter (the output_dir from training)")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Path to dataset for test examples (optional)")
    parser.add_argument("--from_disk", action="store_true", default=True,
                        help="Load dataset from local disk (saved with save_to_disk)")
    parser.add_argument("--no_from_disk", action="store_false", dest="from_disk")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of test examples to run")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to ask (overrides dataset)")
    args = parser.parse_args()

    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load fine-tuned adapter
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    # --- Run inference ---
    if args.question:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.question},
        ]
        print(f"\nQuestion: {args.question}")
        print("-" * 60)
        output = run_inference(model, tokenizer, messages, args.max_new_tokens)
        print(output)
        return

    # Dataset mode
    if args.dataset_name is None:
        print("Error: provide --dataset_name or --question")
        return

    if args.from_disk:
        from datasets import load_from_disk
        full_dataset = load_from_disk(args.dataset_name)
        test_dataset = full_dataset["test"]
    else:
        from datasets import load_dataset
        test_dataset = load_dataset(args.dataset_name, split="test")

    test_dataset = test_dataset.map(make_conversation)
    n = min(args.num_samples, len(test_dataset))

    for i in range(n):
        messages = test_dataset[i]["prompt"]
        question = messages[-1]["content"]
        print(f"\n{'='*60}")
        print(f"Example {i+1}/{n}")
        print(f"Question: {question}")

        print(f"\n--- Fine-tuned model ---")
        output = run_inference(model, tokenizer, messages, args.max_new_tokens)
        print(output)


if __name__ == "__main__":
    main()
