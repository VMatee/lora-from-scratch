"""Train LoRA on a reproducible synthetic low-rank adaptation task."""
import argparse
import torch
from torch import nn

from .layers import LoRALinear


def run(steps: int = 160, seed: int = 7) -> tuple[float, float]:
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    dimensions, rank = 12, 2
    x = torch.randn(256, dimensions)
    base = nn.Linear(dimensions, dimensions)
    teacher_a = torch.randn(rank, dimensions) * 0.3
    teacher_b = torch.randn(dimensions, rank) * 0.3
    with torch.no_grad():
        target = base(x) + (x @ teacher_a.T) @ teacher_b.T
    model = LoRALinear(base, rank=rank, alpha=rank)
    optimizer = torch.optim.Adam([model.A, model.B], lr=0.08)
    initial = torch.mean((model(x) - target) ** 2).item()
    for _ in range(steps):
        loss = torch.mean((model(x) - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = torch.mean((model(x) - target) ** 2).item()
    all_params = sum(p.numel() for p in model.parameters())
    print(f"MSE before: {initial:.6f}; after: {final:.6f}")
    print(f"Trainable: {model.adapter_parameter_count()} / {all_params} parameters")
    return initial, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.steps, args.seed)


if __name__ == "__main__":
    main()
