"""LoRA injection without PEFT or high-level model-adapter libraries.

Wrap selected torch.nn.Linear modules in the hand-written LoRALinear layer,
freeze non-adapter weights, and export just the trainable low-rank matrices.
"""
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .layers import LoRALinear


def _linear_modules(model: nn.Module):
    """Yield parent, attribute name, dotted module path and linear layer."""
    def visit(parent: nn.Module, prefix: str):
        for attr, child in list(parent.named_children()):
            name = f"{prefix}.{attr}" if prefix else attr
            if isinstance(child, nn.Linear):
                yield parent, attr, name, child
            elif not isinstance(child, LoRALinear):
                yield from visit(child, name)
    yield from visit(model, "")


def inject_lora(
    model: nn.Module,
    rank: int = 2,
    alpha: float = 4.0,
    target_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Replace nested linear modules in place; fail before partial replacement.

    Exact dotted module names can be used to select only attention or MLP
    projections. Passing target_names=None selects every nn.Linear child.
    """
    candidates = list(_linear_modules(model))
    targets = set(target_names) if target_names is not None else {p[2] for p in candidates}
    known = {name for _, _, name, _ in candidates}
    unknown = targets - known
    if unknown:
        raise ValueError(f"Unknown or already-adapted target modules: {sorted(unknown)}")
    selected = [(parent, attr, name, layer) for parent, attr, name, layer in candidates if name in targets]
    if not selected:
        raise ValueError("No linear layers selected")
    if rank < 1 or any(rank > min(layer.in_features, layer.out_features) for _, _, _, layer in selected):
        raise ValueError("LoRA rank exceeds one of the selected linear layers")
    # Construct all adapters before installing any of them.
    adapters = [LoRALinear(layer, rank=rank, alpha=alpha) for _, _, _, layer in selected]
    for (parent, attr, _, _), adapter in zip(selected, adapters):
        setattr(parent, attr, adapter)
    return tuple(name for _, _, name, _ in selected)


def iter_adapters(model: nn.Module):
    """Yield dotted path and adapter for each LoRALinear in a model."""
    for name, child in model.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def freeze_non_adapter_parameters(model: nn.Module) -> None:
    """Make only A and B trainable, keeping base layers and norms frozen."""
    for param in model.parameters():
        param.requires_grad_(False)
    for _, adapter in iter_adapters(model):
        adapter.A.requires_grad_(True)
        adapter.B.requires_grad_(True)


def adapter_parameter_count(model: nn.Module) -> int:
    return sum(layer.adapter_parameter_count() for _, layer in iter_adapters(model))


def adapter_state(model: nn.Module) -> dict[str, Any]:
    """Collect tensor-only weights suitable for a lightweight checkpoint."""
    return {
        name: {
            "rank": layer.rank,
            "scaling": layer.scaling,
            "A": layer.A.detach().cpu().clone(),
            "B": layer.B.detach().cpu().clone(),
        }
        for name, layer in iter_adapters(model)
    }


@torch.no_grad()
def load_adapter_state(model: nn.Module, state: dict[str, Any]) -> None:
    """Restore adapter weights by exact name, rank, scaling and shape."""
    existing = dict(iter_adapters(model))
    if set(existing) != set(state):
        raise ValueError("Adapter module names do not match checkpoint")
    for name, layer in existing.items():
        spec = state[name]
        if (spec["rank"] != layer.rank or abs(spec["scaling"] - layer.scaling) > 1e-12
                or tuple(spec["A"].shape) != tuple(layer.A.shape)
                or tuple(spec["B"].shape) != tuple(layer.B.shape)):
            raise ValueError(f"Incompatible adapter weights for {name}")
        if layer.merged:
            raise ValueError("Unmerge adapters before loading trainable weights")
    for name, layer in existing.items():
        spec = state[name]
        layer.A.copy_(spec["A"].to(device=layer.A.device, dtype=layer.A.dtype))
        layer.B.copy_(spec["B"].to(device=layer.B.device, dtype=layer.B.dtype))
