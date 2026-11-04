import argparse, os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model
from compression_rate import record_base, record_compressed, compression_rate
from compression.quantization_int2 import quantize_int2

def main():
    p=argparse.ArgumentParser(description='Case 2: packed INT2 weight-only quantization')
    p.add_argument('--model',default=DEFAULT_MODEL); p.add_argument('--dtype',default='float16',choices=['float16','bfloat16']); p.add_argument('--seed',type=int,default=0)
    a=p.parse_args(); torch.manual_seed(a.seed)
    model,tok=load_model(a.model,a.dtype); base=record_base(model); n=quantize_int2(model); comp=record_compressed(model)
    print(f'Linear layers quantized: {n}'); print(f'Original Linear storage: {base/8/1024**2:.2f} MiB'); print(f'Compressed Linear storage: {comp/8/1024**2:.2f} MiB'); print(f'case2_comp_rate: {compression_rate(model):.2%}')
    ppl=eval_ppl_wikitext2(model,tok); print(f'case2_ppl: {ppl:.4f}')
if __name__=='__main__': main()
