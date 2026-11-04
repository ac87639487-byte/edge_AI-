import argparse, os, sys, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); sys.path.insert(0,os.path.dirname(ROOT))
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model
from common import record_base,record_compressed,compression_rate
from compression.lowrank_svd import apply_svd

def main():
 p=argparse.ArgumentParser(description='Case 3: truncated-SVD Linear compression')
 p.add_argument('--model',default=DEFAULT_MODEL); p.add_argument('--dtype',default='float16',choices=['float16','bfloat16']); p.add_argument('--rank-fraction',type=float,default=0.5); p.add_argument('--seed',type=int,default=0)
 a=p.parse_args()
 if not 0<a.rank_fraction<=1: p.error('--rank-fraction must be in (0,1]')
 torch.manual_seed(a.seed); model,tok=load_model(a.model,a.dtype); base=record_base(model); n=apply_svd(model,a.rank_fraction); comp=record_compressed(model)
 print(f'Linear layers decomposed: {n}'); print(f'SVD rank fraction: {a.rank_fraction:.3f}'); print(f'Original Linear storage: {base/8/1024**2:.2f} MiB'); print(f'Compressed Linear storage: {comp/8/1024**2:.2f} MiB'); print(f'case3_comp_rate: {compression_rate(model):.2%}')
 ppl=eval_ppl_wikitext2(model,tok); print(f'case3_ppl: {ppl:.4f}')
if __name__=='__main__': main()
