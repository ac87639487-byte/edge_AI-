"""Storage accounting for target Linear layers. Rates are fractions of bits saved."""
import torch.nn as nn

def _bits(t):
    return t.numel() * t.element_size() * 8

def record_base(model):
    bits = 0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            bits += _bits(layer.weight)
            if layer.bias is not None: bits += _bits(layer.bias)
    model._base_linear_bits = bits
    return bits

def _compressed_layer_bits(layer):
    if getattr(layer, '_is_packed_quant_linear', False):
        bits = _bits(layer.qweight) + _bits(layer.scales)
        if layer.bias is not None: bits += _bits(layer.bias)
        return bits
    if getattr(layer, '_is_packed_int2_linear', False):
        bits = _bits(layer.qweight) + _bits(layer.scales) + _bits(layer.zeros)
        if layer.bias is not None: bits += _bits(layer.bias)
        return bits
    if getattr(layer, '_is_lowrank_linear', False):
        bits = _bits(layer.left) + _bits(layer.right)
        if layer.bias is not None: bits += _bits(layer.bias)
        return bits
    if getattr(layer, '_is_bitmask_pruned_linear', False):
        bits = _bits(layer.mask_bits) + _bits(layer.values)
        if layer.bias is not None: bits += _bits(layer.bias)
        return bits
    if isinstance(layer, nn.Linear):
        bits = _bits(layer.weight)
        if layer.bias is not None: bits += _bits(layer.bias)
        return bits
    return 0

def record_compressed(model):
    bits = sum(_compressed_layer_bits(m) for m in model.modules())
    model._compressed_linear_bits = bits
    return bits

def compression_rate(model):
    base = getattr(model, '_base_linear_bits', None)
    compressed = getattr(model, '_compressed_linear_bits', None)
    if not base or compressed is None: return 0.0
    return 1.0 - compressed / base

def get_pruning_fraction(model):
    pruned = total = 0
    for m in model.modules():
        if getattr(m, '_is_bitmask_pruned_linear', False):
            pruned += m.pruned_count
            total += m.total_count
    return pruned / total if total else 0.0
