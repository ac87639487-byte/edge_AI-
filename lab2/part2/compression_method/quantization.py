"""Packed unsigned 3-bit, group-wise asymmetric weight-only Linear quantization."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 32
_CHUNK_ELEMS = 1 << 24                      # 每塊約 1600 萬元素，控制暫存峰值
_SHRINK = torch.linspace(1.0, 0.6, 21).tolist()   # 裁切範圍搜尋的收縮比例


def _rows_per_chunk(in_features: int) -> int:
    step = 1 if in_features % 8 == 0 else 8   # 每塊元素數需為 8 的倍數，3-bit 才在 byte 邊界對齊
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
        in_f, out_f = self.in_features, self.out_features
        g = GROUP_SIZE if in_f % GROUP_SIZE == 0 else in_f
        self.group_size = g
        ng = in_f // g
        dtype = linear.weight.dtype
        w_all = linear.weight.detach()
        rows = _rows_per_chunk(in_f)

        q_chunks, s_chunks, z_chunks = [], [], []
        for r0 in range(0, out_f, rows):
            w = w_all[r0:r0 + rows].float().reshape(-1, ng, g)
            xmin = w.amin(-1, keepdim=True).clamp(max=0)
            xmax = w.amax(-1, keepdim=True).clamp(min=0)
            best_err = torch.full_like(xmin, float("inf"))
            best_s = torch.ones_like(xmin)
            best_z = torch.zeros_like(xmin)
            for p in _SHRINK:
                mn, mx = xmin * p, xmax * p
                s = ((mx - mn) / 7).clamp_min(1e-5).to(dtype).float()   # 以實際儲存精度評估誤差
                z = torch.round(-mn / s).clamp(0, 7)
                qv = torch.clamp(torch.round(w / s) + z, 0, 7)
                err = ((qv - z) * s - w).pow(2).sum(-1, keepdim=True)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, s, best_s)
                best_z = torch.where(better, z, best_z)
            q = torch.clamp(torch.round(w / best_s) + best_z, 0, 7)
            q_chunks.append(_pack3(q.reshape(-1).to(torch.int32)))
            s_chunks.append(best_s.squeeze(-1).to(dtype))
            z_chunks.append(best_z.squeeze(-1).to(torch.uint8))
            del w, q, xmin, xmax, best_err, best_s, best_z

        self.register_buffer("qweight", torch.cat(q_chunks), persistent=True)
        self.register_buffer("scales", torch.cat(s_chunks), persistent=True)   # (out, in/g)
        self.register_buffer("zeros", torch.cat(z_chunks), persistent=True)    # (out, in/g)

        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def _dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        out_f, in_f, g = self.out_features, self.in_features, self.group_size
        ng = in_f // g
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
            q = ((padded[byte_idx] >> shift) | (padded[byte_idx + 1] << (8 - shift))) & 0x7
            q = q.reshape(r1 - r0, ng, g).to(torch.float32)
            z = self.zeros[r0:r1].float().unsqueeze(-1)
            s = self.scales[r0:r1].float().unsqueeze(-1)
            weight[r0:r1] = ((q - z) * s).reshape(r1 - r0, in_f).to(dtype)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize(x.dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def quantize_model_linear_layers(model: nn.Module) -> int:
    """Replace selected nn.Linear in-place, keeping compression rate > 75%."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    else:
        num_layers = 0

    # 1. 只跳過最前面的第 0 層（不要跳過最後一層，也不要跳過 lm_head）
    skip_layer_indices = {0} if num_layers > 0 else set()

    targets = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 注意：lm_head 不跳過，必須量化，否則壓縮率會暴跌
            is_sensitive = any(f"layers.{idx}." in module_name for idx in skip_layer_indices)

            if not is_sensitive:
                targets.append(module_name)

    # 執行模組抽換
    for name in targets:
        if not name:
            raise RuntimeError("The model itself is nn.Linear; unsupported root replacement.")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child_name)
        setattr(parent, child_name, PackedInt3Linear(old))

    return len(targets)
