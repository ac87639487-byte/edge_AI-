import os

# 減少顯存碎片；必須在 import torch / transformers 之前設定
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import sys

# 1. 修復 dtype 參數相容性（必須在載入 evaluate 前執行）
from transformers.modeling_utils import PreTrainedModel
import transformers

_orig_from_pretrained = PreTrainedModel.from_pretrained

@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
    return _orig_from_pretrained.__func__(cls, *args, **kwargs)

PreTrainedModel.from_pretrained = _patched_from_pretrained

# 同步修復 Qwen3ForCausalLM 的建構函式 __init__，防止殘留的 dtype 傳入報錯
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    _orig_qwen3_init = Qwen3ForCausalLM.__init__
    def _patched_qwen3_init(self, config, *args, **kwargs):
        kwargs.pop("dtype", None)
        _orig_qwen3_init(self, config, *args, **kwargs)
    Qwen3ForCausalLM.__init__ = _patched_qwen3_init
except Exception:
    pass

# 2. 設定路徑
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

import torch
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model
from common import record_base, record_compressed, compression_rate
from compression_method.quantization import quantize_model_linear_layers


def main():
    parser = argparse.ArgumentParser(description="Case 1: packed 3-bit Linear weight-only quantization")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.dtype)
    base_bits = record_base(model)
    n_layers = quantize_model_linear_layers(model)
    compressed_bits = record_compressed(model)

    # 量化完釋放暫存與快取，再進入評估
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Linear layers quantized: {n_layers}")
    print(f"Original Linear storage: {base_bits / 8 / 1024**2:.2f} MiB")
    print(f"Compressed Linear storage: {compressed_bits / 8 / 1024**2:.2f} MiB")
    print(f"Compression rate: {compression_rate(model):.2%}")
    ppl = eval_ppl_wikitext2(model, tokenizer)
    print(f"case1_ppl: {ppl:.4f}")
    print(f"case1_comp_rate: {compression_rate(model):.2%}")


if __name__ == "__main__":
    main()
