import os
import sys

# 1. 設置顯存碎片整理機制（必須在 import torch 之前執行）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 2. 修復 Qwen3 與 Transformers dtype 參數相容性（避免 TypeError）
from transformers.modeling_utils import PreTrainedModel
_orig_from_pretrained = PreTrainedModel.from_pretrained

@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
    return _orig_from_pretrained.__func__(cls, *args, **kwargs)

PreTrainedModel.from_pretrained = _patched_from_pretrained

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    _orig_qwen3_init = Qwen3ForCausalLM.__init__
    def _patched_qwen3_init(self, config, *args, **kwargs):
        kwargs.pop("dtype", None)
        _orig_qwen3_init(self, config, *args, **kwargs)
    Qwen3ForCausalLM.__init__ = _patched_qwen3_init
except Exception:
    pass

# 3. 模組搜尋路徑設定
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

import gc
import argparse
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
    
    # 執行量化
    n_layers = quantize_model_linear_layers(model)
    compressed_bits = record_compressed(model)

    # 量化完成後立即清理垃圾回收與 CUDA 顯存快取，為後續 evaluate.py 騰出 4~5 GB 空間
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
