"""Unstructured magnitude pruning stored as a packed 1-bit mask + retained FP16 values."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BitmaskPrunedLinear(nn.Module):
    _is_bitmask_pruned_linear = True
    def __init__(self, linear, prune_fraction=0.20):
        super().__init__()
        w = linear.weight.detach()
        flat = w.abs().float().reshape(-1)
        n = flat.numel()
        k = int(round(n * prune_fraction))
        mask = torch.ones(n, dtype=torch.bool, device=w.device)
        if k:
            idx = torch.topk(flat, k, largest=False, sorted=False).indices
            mask[idx] = False
        kept = w.reshape(-1)[mask]
        # Pack mask bits, least-significant bit first.
        pad = (-n) % 8
        bits = mask.to(torch.uint8)
        if pad: bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8, device=w.device)])
        packed = torch.zeros(bits.numel() // 8, dtype=torch.uint8, device=w.device)
        for b in range(8): packed |= bits[b::8] << b
        self.register_buffer('mask_bits', packed, persistent=True)
        self.register_buffer('values', kept.contiguous(), persistent=True)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.pruned_count, self.total_count = k, n
        if linear.bias is None: self.register_parameter('bias', None)
        else: self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
    def dequant_weight(self, dtype):
        n = self.in_features * self.out_features
        p = self.mask_bits
        bits = torch.stack([((p >> b) & 1) for b in range(8)], dim=1).reshape(-1)[:n].bool()
        dense = torch.zeros(n, dtype=self.values.dtype, device=self.values.device)
        dense[bits] = self.values
        return dense.reshape(self.out_features, self.in_features).to(dtype)
    def forward(self, x):
        return F.linear(x, self.dequant_weight(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

def apply_pruning(model, prune_fraction=0.20):
    names = [n for n,m in model.named_modules() if n and isinstance(m, nn.Linear)]
    for name in names:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, BitmaskPrunedLinear(getattr(parent, child), prune_fraction))
    return len(names)
