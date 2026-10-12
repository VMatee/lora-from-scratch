import pytest
import torch
from torch import nn
from lora_lab import LoRALinear
from lora_lab.demo import run


def test_initial_layer_identical_and_base_frozen():
    torch.manual_seed(2)
    layer = nn.Linear(8, 5)
    x = torch.randn(4, 8)
    original = layer(x).detach().clone()
    adapter = LoRALinear(layer, rank=2, alpha=2)
    assert torch.allclose(adapter(x), original, atol=1e-6)
    assert not any(p.requires_grad for p in adapter.base.parameters())
    assert adapter.A.requires_grad and adapter.B.requires_grad
    assert adapter.adapter_parameter_count() == 2 * (8 + 5)


def test_adapter_receives_gradients_and_merging_is_equivalent():
    torch.manual_seed(3)
    model = LoRALinear(nn.Linear(7, 4), rank=2)
    x = torch.randn(3, 7)
    loss = model(x).square().mean()
    loss.backward()
    assert model.B.grad is not None
    with torch.no_grad():
        model.B.add_(0.1)
    unmerged = model(x).detach().clone()
    original_weights = model.base.weight.detach().clone()
    model.merge()
    model.merge()  # idempotent
    assert torch.allclose(model(x), unmerged, atol=2e-6)
    model.unmerge()
    model.unmerge()
    assert torch.allclose(model.base.weight, original_weights, atol=2e-6)
    assert torch.allclose(model(x), unmerged, atol=2e-6)


def test_synthetic_task_improves():
    before, after = run(steps=120, seed=4)
    assert after < before * 0.3


def test_invalid_rank_alpha():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 2), rank=3)
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 2), rank=1, alpha=-1)
