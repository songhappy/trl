#!/usr/bin/env python
"""Parse GRPO training log and plot loss curve.

Usage:
  python examples/rl/loss.py --log /home/guoqiong/data/grpo_run.log
  python examples/rl/loss.py --log /home/guoqiong/data/grpo_run.log --output /home/guoqiong/data/grpo_loss_curve.png
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(log_path):
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"'loss':\s*([\d.]+).*'step':\s*(\d+)", line)
            if m:
                losses.append(float(m.group(1)))
                steps.append(int(m.group(2)))
    return steps, losses


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training loss curve")
    parser.add_argument("--log", type=str, required=True, help="Path to training log file")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: same dir as log)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.log.rsplit(".", 1)[0] + "_loss.png"

    steps, losses = parse_log(args.log)

    if not steps:
        print(f"No loss entries found in {args.log}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker="o", markersize=4)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("GRPO Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)

    print(f"Steps: {len(steps)}")
    print(f"Loss: {losses[0]:.4f} (start) -> {losses[-1]:.4f} (end)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
