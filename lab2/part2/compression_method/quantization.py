"""Case 1: mixed 3/4-bit, group-wise asymmetric, weight-only Linear quantization with GPTQ.

The engine (calibration, sensitivity probe, bit plan, GPTQ, packing) lives in quantization_int2.py.
Every block gets 4 bit; the least sensitive blocks are lowered to 3 bit until the storage rate
(packed codes + fp16 scales + uint8 zeros, exactly what common.py counts) reaches TARGET_RATE.
"""
from compression_method.quantization_int2 import PackedLowBitLinear, quantize_mixed

TARGET_RATE = 0.755   # must stay above the 75% requirement with all stored buffers counted


def quantize_model_linear_layers(model) -> int:
    """Replace every nn.Linear in-place. Returns the number of replaced layers."""
    return quantize_mixed(model, low_bits=3, high_bits=4, target_rate=TARGET_RATE)
