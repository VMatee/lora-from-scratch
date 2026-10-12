"""Transparent low-rank adapter for a frozen dense layer."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Applies W x + b + (alpha / rank) B A x to the last dimension."""

    def __init__(self, base: nn.Linear, rank: int = 2, alpha: float = 4.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must fit inside base matrix dimensions")
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("alpha must be positive and finite")
        # Own the base module. Consumers should not modify the original during training.
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.merged:
            return y
        update = F.linear(F.linear(x, self.A), self.B)
        return y + self.scaling * update

    @torch.no_grad()
    def merge(self) -> None:
        """Merge adapter into W once, for inference with no active optimizer."""
        if not self.merged:
            self.base.weight.add_(self.scaling * (self.B @ self.A))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Restore original W for future adapter training/evaluation."""
        if self.merged:
            self.base.weight.sub_(self.scaling * (self.B @ self.A))
            self.merged = False

    def adapter_parameter_count(self) -> int:
        return self.A.numel() + self.B.numel()
