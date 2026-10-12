"""An end-to-end adapter injection and lightweight checkpoint example."""
import torch
from torch import nn

from .inject import (
    inject_lora, freeze_non_adapter_parameters,
    adapter_parameter_count, iter_adapters,
)


def main() -> None:
    torch.manual_seed(9)
    base = nn.Sequential(
        nn.Linear(12, 12), nn.Tanh(), nn.Linear(12, 6)
    )
    inputs = torch.randn(256, 12)
    teacher = nn.Sequential(nn.Linear(12, 12), nn.Tanh(), nn.Linear(12, 6))
    with torch.no_grad():
        expected = teacher(inputs).detach()
    targets = inject_lora(base, rank=2, alpha=2)
    freeze_non_adapter_parameters(base)
    optimizer = torch.optim.Adam((p for p in base.parameters() if p.requires_grad), lr=0.07)
    with torch.no_grad():
        initial = float((base(inputs) - expected).square().mean())
    for _ in range(100):
        loss = (base(inputs) - expected).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = float((base(inputs) - expected).square().mean())
    print("Injected modules:", targets)
    print("Trainable adapter parameters:", adapter_parameter_count(base))
    print(f"Initial MSE: {initial:.5f}; final MSE: {final:.5f}")
    # Merge/unmerge is only performed after optimizer updates are complete.
    with torch.no_grad():
        reference = base(inputs)
        for _, adapter in iter_adapters(base):
            adapter.merge()
        difference = float((base(inputs) - reference).abs().max())
        print(f"Max difference after merging: {difference:.8f}")


if __name__ == "__main__":
    main()
