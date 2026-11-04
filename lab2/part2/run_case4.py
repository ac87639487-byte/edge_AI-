import argparse, os, sys, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); sys.path.insert(0,os.path.dirname(ROOT))
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model
from common import record_base,record_compressed,compression_rate,get_pruning_fraction
from compression_method.prune_bitmask import apply_pruning

def main():
 p=argparse.ArgumentParser(description='Case 4: magnitude pruning with packed bitmask storage')
 p.add_argument('--model',default=DEFAULT_MODEL); p.add_argument('--dtype',default='float16',choices=['float16','bfloat16']); p.add_argument('--prune-fraction',type=float,default=0.20); p.add_argument('--seed',type=int,default=0)
 a=p.parse_args()
 if not 0<=a.prune_fraction<1: p.error('--prune-fraction must be in [0,1)')
 torch.manual_seed(a.seed); model,tok=load_model(a.model,a.dtype); base=record_base(model); n=apply_pruning(model,a.prune_fraction); comp=record_compressed(model)
 print(f'Linear layers pruned: {n}'); print(f'Actual pruning fraction: {get_pruning_fraction(model):.2%}'); print(f'Original Linear storage: {base/8/1024**2:.2f} MiB'); print(f'Compressed Linear storage: {comp/8/1024**2:.2f} MiB'); print(f'case4_comp_rate: {compression_rate(model):.2%}')
 ppl=eval_ppl_wikitext2(model,tok); print(f'case4_ppl: {ppl:.4f}')
if __name__=='__main__': main()
