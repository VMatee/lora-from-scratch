"""Numerical and structural tests for nested low-rank model injection."""
import pytest
import torch
from torch import nn
from lora_lab.inject import (
    inject_lora, freeze_non_adapter_parameters, adapter_parameter_count,
    adapter_state, load_adapter_state, iter_adapters,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


def test_nested_injection_preserves_initial_forward_output_and_only_adapters_train():
    torch.manual_seed(11)
    model = ToyModel()
    data = torch.randn(5, 8)
    original = model(data).detach().clone()
    selected = inject_lora(model, rank=2, alpha=2)
    assert selected == ("encoder.0", "encoder.2", "head")
    freeze_non_adapter_parameters(model)
    assert torch.allclose(model(data), original, atol=1e-7)
    assert adapter_parameter_count(model) == 2 * (8+6) + 2 * (6+4) + 2 * (4+2)
    assert all(p.requires_grad for name, layer in iter_adapters(model) for p in (layer.A, layer.B))
    assert sum(p.requires_grad for p in model.parameters()) == len(selected) * 2


def test_low_rank_matmul_and_merging_agree_with_dense_reference():
    torch.manual_seed(13)
    model = ToyModel()
    inject_lora(model, rank=2)
    x = torch.randn(4, 8)
    layer = model.encoder[0]
    with torch.no_grad():
        layer.A.normal_(0, 0.2)
        layer.B.normal_(0, 0.2)
        expected = layer.base(x) + layer.scaling * ((x @ layer.A.T) @ layer.B.T)
        assert torch.allclose(layer(x), expected, atol=1e-6)
        before = model(x)
        for _, adapter in iter_adapters(model):
            adapter.merge()
        assert torch.allclose(model(x), before, atol=1e-6)
        for _, adapter in iter_adapters(model):
            adapter.unmerge()
        assert torch.allclose(model(x), before, atol=1e-6)


def test_checkpoint_roundtrip_and_target_validation():
    torch.manual_seed(17)
    source = ToyModel()
    target = ToyModel()
    inject_lora(source, rank=2)
    inject_lora(target, rank=2)
    weights = adapter_state(source)
    load_adapter_state(target, weights)
    assert torch.equal(target.encoder[0].A, source.encoder[0].A)
    assert torch.equal(target.encoder[0].B, source.encoder[0].B)
    with pytest.raises(ValueError):
        inject_lora(ToyModel(), rank=2, target_names=["does.not.exist"])
    with pytest.raises(ValueError):
        inject_lora(ToyModel(), rank=100)
    with pytest.raises(ValueError):
        load_adapter_state(target, {"not-a-layer": {}})
