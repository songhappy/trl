#!/usr/bin/env python
"""Parse GRPO training metrics (JSONL) and plot loss and reward curves.

Usage:
  python examples/rl/log.py --log ./xpu-grpo-output/training_metrics.jsonl
  python examples/rl/log.py --log ./xpu-grpo-output/training_metrics.jsonl --output ./xpu-grpo-output/curves.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_jsonl(log_path):
    steps, losses = [], []
    reward_steps, rewards = [], []
    reward_std_steps, reward_stds = [], []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            step = entry.get("step")
            if step is None:
                continue
            if "loss" in entry:
                steps.append(step)
                losses.append(entry["loss"])
            if "rewards/reasoning_accuracy_reward/mean" in entry:
                reward_steps.append(step)
                rewards.append(entry["rewards/reasoning_accuracy_reward/mean"])
            if "rewards/reasoning_accuracy_reward/std" in entry:
                reward_std_steps.append(step)
                reward_stds.append(entry["rewards/reasoning_accuracy_reward/std"])

    return steps, losses, reward_steps, rewards, reward_std_steps, reward_stds


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training loss and reward curves")
    parser.add_argument("--log", type=str, required=True, help="Path to training_metrics.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: same dir as log)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.log.rsplit(".", 1)[0] + "_curves.png"

    steps, losses, reward_steps, rewards, reward_std_steps, reward_stds = parse_jsonl(args.log)

    if not steps and not reward_steps:
        print(f"No loss or reward entries found in {args.log}")
        return

    has_loss = len(steps) > 0
    has_reward = len(reward_steps) > 0
    num_plots = has_loss + has_reward

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), squeeze=False)
    ax_idx = 0

    if has_loss:
        ax = axes[ax_idx, 0]
        ax.plot(steps, losses, marker="o", markersize=3, color="tab:blue")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("GRPO Training Loss")
        ax.grid(True)
        ax_idx += 1

    if has_reward:
        ax = axes[ax_idx, 0]
        ax.plot(reward_steps, rewards, marker="o", markersize=3, color="tab:green", label="reasoning_accuracy")
        if reward_stds and len(reward_stds) == len(rewards):
            ax.fill_between(
                reward_std_steps,
                [r - s for r, s in zip(rewards, reward_stds)],
                [r + s for r, s in zip(rewards, reward_stds)],
                alpha=0.2, color="tab:green", label="reasoning_accuracy +/- std",
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Reasoning Accuracy Reward")
        ax.set_title("GRPO Reasoning Accuracy Reward")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

    if has_loss:
        print(f"Loss: {len(steps)} points, {losses[0]:.4f} -> {losses[-1]:.4f}")
    if has_reward:
        print(f"Reward: {len(reward_steps)} points, {rewards[0]:.4f} -> {rewards[-1]:.4f}")
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
