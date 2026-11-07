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
from common import record_base, record_compressed, compression_rate
from compression_method.lowrank_svd import OVERALL_TARGET, SVD_SAVING, apply_svd, svd_param_saving


def main():
    p = argparse.ArgumentParser(description="Case 3: whitened truncated-SVD + int8 factors")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--svd-saving", type=float, default=SVD_SAVING,
                   help="minimum fraction of each Linear's weights saved by the rank truncation alone")
    p.add_argument("--overall-target", type=float, default=OVERALL_TARGET,
                   help="overall compression rate to land on (fp16 factors are used to hit it from above)")
    p.add_argument("--rank-fraction", type=float, default=None,
                   help="optional: rank = fraction * min(in, out) instead of --svd-saving")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.rank_fraction is not None and not 0 < a.rank_fraction <= 1:
        p.error("--rank-fraction must be in (0,1]")
    torch.manual_seed(a.seed)
    model, tok = load_model(a.model, a.dtype)
    base = record_base(model)
    n = apply_svd(model, a.rank_fraction, a.svd_saving, target=a.overall_target)
    comp = record_compressed(model)
    saving, worst = svd_param_saving(model)
    print(f"Linear layers decomposed: {n}")
    print(f"SVD compression ratio (rank truncation only): {saving:.2%} (smallest per-layer: {worst:.2%})")
    print(f"Original Linear storage: {base / 8 / 1024**2:.2f} MiB")
    print(f"Compressed Linear storage: {comp / 8 / 1024**2:.2f} MiB")
    print(f"case3_comp_rate: {compression_rate(model):.2%}")
    ppl = eval_ppl_wikitext2(model, tok)
    print(f"case3_ppl: {ppl:.4f}")


if __name__ == "__main__":
    main()
