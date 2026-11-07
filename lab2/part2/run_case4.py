import os

# 1. 顯存碎片整理（必須在 import torch / transformers 之前設定）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys

# 2. 修復 Qwen3 與 Transformers 的 dtype 參數相容性（與 run_case1.py 相同）
from transformers.modeling_utils import PreTrainedModel
import transformers

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

# 3. 設定路徑
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

import torch
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model
from common import record_base, record_compressed, compression_rate, get_pruning_fraction
from compression_method.prune_bitmask import PRUNE_FRACTION, TARGET_RATE, apply_pruning


def main():
    p = argparse.ArgumentParser(description="Case 4: SparseGPT-style pruning + 2/3-bit GPTQ, bitmask storage")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--prune-fraction", type=float, default=PRUNE_FRACTION)
    p.add_argument("--target-rate", type=float, default=TARGET_RATE)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not 0 < a.prune_fraction < 1:
        p.error("--prune-fraction must be in (0,1)")
    torch.manual_seed(a.seed)
    model, tok = load_model(a.model, a.dtype)
    base = record_base(model)
    n = apply_pruning(model, a.prune_fraction, target=a.target_rate)
    comp = record_compressed(model)
    print(f"Linear layers pruned: {n}")
    print(f"Pruning rate: {get_pruning_fraction(model):.2%}")
    print(f"Original Linear storage: {base / 8 / 1024**2:.2f} MiB")
    print(f"Compressed Linear storage: {comp / 8 / 1024**2:.2f} MiB")
    print(f"case4_comp_rate: {compression_rate(model):.2%}")
    ppl = eval_ppl_wikitext2(model, tok)
    print(f"case4_ppl: {ppl:.4f}")


if __name__ == "__main__":
    main()
