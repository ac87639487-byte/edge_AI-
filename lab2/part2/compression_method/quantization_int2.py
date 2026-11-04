"""Packed unsigned INT2 weight-only quantization with per-output-channel affine parameters."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PackedInt2Linear(nn.Module):
    _is_packed_int2_linear = True
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w = linear.weight.detach().float()
        lo = w.amin(dim=1, keepdim=True)
        hi = w.amax(dim=1, keepdim=True)
        scale = ((hi - lo) / 3).clamp_min(1e-8)
        zero = torch.round(-lo / scale).clamp(0, 3).to(torch.uint8)
        q = torch.round(w / scale + zero.float()).clamp(0, 3).to(torch.uint8).reshape(-1)
        n = q.numel()
        pad = (-n) % 4
        if pad: q = torch.cat([q, torch.zeros(pad, dtype=torch.uint8, device=q.device)])
        packed = q[0::4] | (q[1::4] << 2) | (q[2::4] << 4) | (q[3::4] << 6)
        self.register_buffer('qweight', packed.contiguous(), persistent=True)
        self.register_buffer('scales', scale.to(linear.weight.dtype), persistent=True)
        self.register_buffer('zeros', zero, persistent=True)
        if linear.bias is None: self.register_parameter('bias', None)
        else: self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
    def dequant(self, dtype):
        p = self.qweight
        q = torch.stack((p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3), dim=1).reshape(-1)[:self.in_features*self.out_features]
        q = q.reshape(self.out_features, self.in_features).float()
        w = (q - self.zeros.float()) * self.scales.float()
        return w.to(dtype)
    def forward(self, x):
        return F.linear(x, self.dequant(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

def quantize_int2(model):
    names = [n for n,m in model.named_modules() if n and isinstance(m, nn.Linear)]
    for name in names:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, PackedInt2Linear(getattr(parent, child)))
    return len(names)
