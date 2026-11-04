"""Packed signed 3-bit, per-output-channel weight-only Linear quantization."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_CHUNK_ELEMS = 16 * 1024 * 1024


def _pack_8_to_3_bytes(q_unsigned: torch.Tensor) -> torch.Tensor:
    """將每 8 個 3-bit (0~7) 整數向量化打包成 3 個 uint8 (24 bits)

    無須 bit-by-bit 迴圈，純張量並行運算。
    """
    # 形狀 reshape 為 (-1, 8)
    q = q_unsigned.view(-1, 8).to(torch.int32)
    b0 = (q[:, 0]) | (q[:, 1] << 3) | ((q[:, 2] & 0x03) << 6)
    b1 = ((q[:, 2] >> 2) & 0x01) | (q[:, 3] << 1) | (q[:, 4] << 4) | ((q[:, 5] & 0x01) << 7)
    b2 = ((q[:, 5] >> 1) & 0x03) | (q[:, 6] << 2) | (q[:, 7] << 5)
    return torch.stack([b0, b1, b2], dim=1).reshape(-1).to(torch.uint8)


def _unpack_3_bytes_to_8(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    """將每 3 個 uint8 向量化還原為 8 個 3-bit 數值"""
    p = packed.view(-1, 3).to(torch.int32)
    b0, b1, b2 = p[:, 0], p[:, 1], p[:, 2]

    v0 = b0 & 0x07
    v1 = (b0 >> 3) & 0x07
    v2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    v3 = (b1 >> 1) & 0x07
    v4 = (b1 >> 4) & 0x07
    v5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    v6 = (b2 >> 2) & 0x07
    v7 = (b2 >> 5) & 0x07

    unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).reshape(-1)
    return unpacked[:n_elements]


class PackedInt3Linear(nn.Module):
    _is_packed_quant_linear = True

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        dtype = linear.weight.dtype
        w = linear.weight.detach().float()

        # 1. 對稱量化 (Symmetric 3-bit, 範圍 -4 ~ 3)
        max_abs = w.abs().amax(dim=1, keepdim=True)
        # 用 3.5 或是 3.0 作為 scale 分母
        scales = torch.where(
            max_abs > 0, max_abs / 3.0, torch.ones_like(max_abs)
        )
        q = torch.round(w / scales).clamp(-4, 3).to(torch.int16)

        # 轉成 0~7 的 unsigned 3-bit 表示
        q_unsigned = (q & 0x07).reshape(-1)

        # 2. 向量化打包
        packed = _pack_8_to_3_bytes(q_unsigned)

        self.register_buffer("qweight", packed, persistent=True)
        self.register_buffer("scales", scales.to(dtype=dtype), persistent=True)

        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                linear.bias.detach().clone(), requires_grad=False
            )

        # 宣告動態快取
        self._cached_weight = None

    def _dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        n_elements = self.out_features * self.in_features
        # 向量化直接解碼
        value = _unpack_3_bytes_to_8(self.qweight, n_elements)

        # 轉回 signed: 4~7 對應 -4~-1
        signed = torch.where(value >= 4, value - 8, value).to(torch.float32)
        q = signed.reshape(self.out_features, self.in_features)

        # 乘回 scales (out_features, 1)
        weight = (q * self.scales.float()).to(dtype=dtype)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 推論模式下快取解碼權重，避免 35,000 次重複解包
        if not self.training:
            if (
                self._cached_weight is None
                or self._cached_weight.dtype != x.dtype
            ):
                self._cached_weight = self._dequantize(x.dtype)
            weight = self._cached_weight
        else:
            weight = self._dequantize(x.dtype)

        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def quantize_model_linear_layers(model: nn.Module) -> int:
    """Replace every nn.Linear in-place.

    Returns the number of replaced layers.
    """
    targets = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            targets.append(module_name)

    for name in targets:
        if not name:
            raise RuntimeError(
                "The model itself is nn.Linear; unsupported root replacement."
            )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child_name)
        setattr(parent, child_name, PackedInt3Linear(old))
    return len(targets)
