"""Packed signed 3-bit, per-output-channel weight-only Linear quantization."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PackedInt3Linear(nn.Module):
    _is_packed_quant_linear = True

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        device = linear.weight.device
        dtype = linear.weight.dtype

        w = linear.weight.detach().float()
        # One scale per output channel. Signed 3-bit values are in [-4, 3].
        max_abs = w.abs().amax(dim=1, keepdim=True)
        scales = torch.where(max_abs > 0, max_abs / 3.0, torch.ones_like(max_abs))
        q = torch.round(w / scales).clamp(-4, 3).to(torch.int16)
        q_unsigned = (q & 0x7).reshape(-1).to(torch.int32)

        # Pack a continuous little-endian 3-bit stream into uint8 bytes.
        n = q_unsigned.numel()
        bit_pos = torch.arange(n, device=device, dtype=torch.int64) * 3
        byte_idx = bit_pos // 8
        shift = bit_pos % 8
        packed_len = math.ceil(n * 3 / 8)
        packed = torch.zeros(packed_len + 1, dtype=torch.int32, device=device)
        for bit in range(3):
            positions = bit_pos + bit
            dst_byte = positions // 8
            dst_shift = positions % 8
            packed.scatter_add_(0, dst_byte, (((q_unsigned >> bit) & 1) << dst_shift).to(torch.int32))
        self.register_buffer("qweight", packed[:packed_len].to(torch.uint8), persistent=True)
        self.register_buffer("scales", scales.to(dtype=dtype), persistent=True)

        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def _dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        n = self.in_features * self.out_features
        device = self.qweight.device
        bit_pos = torch.arange(n, device=device, dtype=torch.int64) * 3
        byte_idx = bit_pos // 8
        shift = bit_pos % 8
        raw = self.qweight.to(torch.int32)
        # A second byte is needed when a 3-bit value straddles a byte boundary.
        padded = torch.cat((raw, torch.zeros(1, dtype=torch.int32, device=device)))
        value = ((padded[byte_idx] >> shift) |
                 (padded[byte_idx + 1] << (8 - shift))) & 0x7
        signed = torch.where(value >= 4, value - 8, value)
        q = signed.reshape(self.out_features, self.in_features).to(torch.float32)
        w = q * self.scales.float()
        return w.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize(x.dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def quantize_model_linear_layers(model: nn.Module) -> int:
    """Replace every nn.Linear in-place. Returns the number of replaced layers."""
    targets = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            targets.append(module_name)

    for name in targets:
        if not name:
            raise RuntimeError("The model itself is nn.Linear; unsupported root replacement.")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child_name)
        setattr(parent, child_name, PackedInt3Linear(old))
    return len(targets)
