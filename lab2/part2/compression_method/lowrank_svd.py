"""Truncated-SVD Linear replacement. rank_fraction is relative to min(in,out)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankLinear(nn.Module):
    _is_lowrank_linear = True
    def __init__(self, linear, rank):
        super().__init__()
        w = linear.weight.detach().float()
        # Full reduced SVD; cast factors back to original dtype for storage.
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        root = s[:rank].clamp_min(0).sqrt()
        left = u[:, :rank] * root.unsqueeze(0)
        right = root.unsqueeze(1) * vh[:rank, :]
        self.register_buffer('left', left.to(linear.weight.dtype), persistent=True)
        self.register_buffer('right', right.to(linear.weight.dtype), persistent=True)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if linear.bias is None: self.register_parameter('bias', None)
        else: self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
    def forward(self, x):
        y = F.linear(x, self.right)
        return F.linear(y, self.left, None if self.bias is None else self.bias.to(x.dtype))

def apply_svd(model, rank_fraction=0.5, min_rank=1):
    names = [n for n,m in model.named_modules() if n and isinstance(m, nn.Linear)]
    replaced = 0
    for name in names:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        rank = max(min_rank, int(min(old.in_features, old.out_features) * rank_fraction))
        rank = min(rank, min(old.in_features, old.out_features))
        setattr(parent, child, LowRankLinear(old, rank))
        replaced += 1
    return replaced
