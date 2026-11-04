"""Packed signed 3-bit, per-output-channel weight-only Linear quantization."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def _rows_per_chunk(in_features: int) -> int:
    # 每塊元素數必須是 8 的倍數，3-bit 位元流才會在 byte 邊界對齊
    step = 1 if in_features % 8 == 0 else 8
    return max(step, (_CHUNK_ELEMS // in_features) // step * step)


def _pack3(q_unsigned: torch.Tensor) -> torch.Tensor:
    n = q_unsigned.numel()
    device = q_unsigned.device
    bit_pos = torch.arange(n, device=device, dtype=torch.int64) * 3
    packed_len = math.ceil(n * 3 / 8)
    packed = torch.zeros(packed_len + 1, dtype=torch.int32, device=device)
    for bit in range(3):
        positions = bit_pos + bit
        dst_byte = positions // 8
        dst_shift = positions % 8
        packed.scatter_add_(0, dst_byte, (((q_unsigned >> bit) & 1) << dst_shift).to(torch.int32))
    return packed[:packed_len].to(torch.uint8)



class PackedInt3Linear(nn.Module):
    _is_packed_quant_linear = True

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        dtype = linear.weight.dtype
        w_all = linear.weight.detach()
        rows = _rows_per_chunk(self.in_features)

        q_chunks, s_chunks = [], []
        for r0 in range(0, self.out_features, rows):
            w = w_all[r0:r0 + rows].float()
            max_abs = w.abs().amax(dim=1, keepdim=True)
            scales = torch.where(max_abs > 0, max_abs / 3.0, torch.ones_like(max_abs))
            q = torch.round(w / scales).clamp(-4, 3).to(torch.int16)
            q_unsigned = (q & 0x7).reshape(-1).to(torch.int32)
            q_chunks.append(_pack3(q_unsigned))
            s_chunks.append(scales.to(dtype=dtype))
            del w, q, q_unsigned

        self.register_buffer("qweight", torch.cat(q_chunks), persistent=True)
        self.register_buffer("scales", torch.cat(s_chunks), persistent=True)

        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def _dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        out_f, in_f = self.out_features, self.in_features
        device = self.qweight.device
        weight = torch.empty(out_f, in_f, dtype=dtype, device=device)
        rows = _rows_per_chunk(in_f)
        for r0 in range(0, out_f, rows):
            r1 = min(r0 + rows, out_f)
            n = (r1 - r0) * in_f
            b0 = (r0 * in_f * 3) // 8
            b1 = math.ceil(r1 * in_f * 3 / 8)
            raw = self.qweight[b0:b1].to(torch.int32)
            padded = torch.cat((raw, torch.zeros(1, dtype=torch.int32, device=device)))
            bit_pos = torch.arange(n, device=device, dtype=torch.int64) * 3
            byte_idx = bit_pos // 8
            shift = bit_pos % 8
            value = ((padded[byte_idx] >> shift) |
                     (padded[byte_idx + 1] << (8 - shift))) & 0x7
            signed = torch.where(value >= 4, value - 8, value)
            q = signed.reshape(r1 - r0, in_f).to(torch.float32)
            weight[r0:r1] = (q * self.scales[r0:r1].float()).to(dtype=dtype)
        return weight

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
