import argparse
import os
import sys

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

    print(f"Linear layers quantized: {n_layers}")
    print(f"Original Linear storage: {base_bits / 8 / 1024**2:.2f} MiB")
    print(f"Compressed Linear storage: {compressed_bits / 8 / 1024**2:.2f} MiB")
    print(f"Compression rate: {compression_rate(model):.2%}")
    ppl = eval_ppl_wikitext2(model, tokenizer)
    print(f"case1_ppl: {ppl:.4f}")
    print(f"case1_comp_rate: {compression_rate(model):.2%}")


if __name__ == "__main__":
    main()
