"""Mixed-precision (2/3/4-bit), group-wise asymmetric, weight-only Linear quantization with GPTQ.
(Case 2 uses 2/3 bit via quantize_int2; Case 1 uses 3/4 bit via quantize_mixed.)

Pipeline (entry point: quantize_int2(model)):
  1. Calibration data: random windows from the WikiText-2 *train* split (never the test split).
  2. Sensitivity probe: fake-quantize each block (attn / mlp of every layer, lm_head) to 2 bit with RTN,
     measure the calibration-loss increase.
  3. Bit plan: greedily give 2 bit to the least sensitive blocks (per weight) until the storage
     accounting of common.py reaches TARGET_RATE; all other blocks get 3 bit.
  4. GPTQ, layer by layer (true-sequential), then replace every nn.Linear with PackedLowBitLinear.
Storage per Linear = packed codes + fp16 scale + uint8 zero per group (all counted by common.py).
"""
import gc
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 128
TARGET_RATE = 0.812          # common.py compression rate to reach (assignment asks for >= 0.81)
N_SAMPLES = 64               # calibration windows
SEQLEN = 2048
PROBE_SEGS = 2               # windows used by the sensitivity probe
_CHUNK_ELEMS = 1 << 24       # bounds temporary tensors when packing / unpacking
_SHRINK = torch.linspace(1.0, 0.5, 26).tolist()   # clipping-range search per group


def _rows_per_chunk(in_features):
    return max(1, _CHUNK_ELEMS // in_features)


# ----------------------------------------------------------------------------- packing
def _pack_codes(codes, bits):
    """uint8 codes (rows, in) with values < 2**bits -> flat uint8 stream.
    Little-endian bit stream: 8 consecutive values occupy exactly `bits` bytes."""
    rows, in_f = codes.shape
    assert in_f % 8 == 0
    acc = torch.int64 if bits > 3 else torch.int32
    shifts = torch.arange(8, device=codes.device, dtype=acc) * bits
    out = []
    step = _rows_per_chunk(in_f)
    for r0 in range(0, rows, step):
        v = codes[r0:r0 + step].reshape(-1, 8).to(acc)
        x = (v << shifts).sum(1, dtype=acc)
        out.append(torch.stack([(x >> (8 * k)) & 0xFF for k in range(bits)], 1).to(torch.uint8).reshape(-1))
    return torch.cat(out)


class PackedLowBitLinear(nn.Module):
    # common.py counts qweight + scales + zeros for this flag (every stored buffer is accounted for).
    _is_packed_int2_linear = True

    def __init__(self, in_features, out_features, bits, group_size, qweight, scales, zeros, bias=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("scales", scales, persistent=True)   # fp16, (out, in/group)
        self.register_buffer("zeros", zeros, persistent=True)     # uint8, (out, in/group)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def _dequantize(self, dtype):
        out_f, in_f, g, bits = self.out_features, self.in_features, self.group_size, self.bits
        ng = in_f // g
        dev = self.qweight.device
        mask = (1 << bits) - 1
        acc = torch.int64 if bits > 3 else torch.int32
        shifts = torch.arange(8, device=dev, dtype=acc) * bits
        weight = torch.empty(out_f, in_f, dtype=dtype, device=dev)
        step = _rows_per_chunk(in_f)
        for r0 in range(0, out_f, step):
            r1 = min(r0 + step, out_f)
            raw = self.qweight[r0 * in_f * bits // 8: r1 * in_f * bits // 8].reshape(-1, bits).to(acc)
            x = raw[:, 0]
            for k in range(1, bits):
                x = x | (raw[:, k] << (8 * k))
            q = ((x.unsqueeze(1) >> shifts) & mask).reshape(r1 - r0, ng, g).to(torch.float32)
            z = self.zeros[r0:r1].to(torch.float32).unsqueeze(-1)
            s = self.scales[r0:r1].to(torch.float32).unsqueeze(-1)
            weight[r0:r1] = ((q - z) * s).reshape(r1 - r0, in_f).to(dtype)
        return weight

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self._dequantize(x.dtype), bias)


PackedInt2Linear = PackedLowBitLinear   # old name


# ----------------------------------------------------------------------------- GPTQ
def _group_params(Wg, maxq):
    """Per-row asymmetric scale / zero for a (rows, group) slice, with a clipping-range search."""
    xmin = Wg.amin(1, keepdim=True).clamp(max=0)
    xmax = Wg.amax(1, keepdim=True).clamp(min=0)
    best_err = torch.full_like(xmin, float("inf"))
    best_s = torch.ones_like(xmin)
    best_z = torch.zeros_like(xmin)
    for p in _SHRINK:
        mn, mx = xmin * p, xmax * p
        s = ((mx - mn) / maxq).clamp_min(1e-5).half().float()      # scale is stored as fp16
        z = torch.round(-mn / s).clamp(0, maxq)
        q = torch.clamp(torch.round(Wg / s) + z, 0, maxq)
        err = ((q - z) * s - Wg).pow(2).sum(1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
        best_z = torch.where(better, z, best_z)
    return best_s, best_z


@torch.no_grad()
def _gptq(W, H, bits, gs, qdtype, want_fake=True, blocksize=128, percdamp=0.01):
    """GPTQ. W: float32 (out, in), modified in place. H: float32 (in, in).
    Returns codes (uint8), scales (fp16), zeros (uint8) and the dequantized weight (qdtype) if want_fake."""
    maxq = 2 ** bits - 1
    out_f, in_f = W.shape
    blocksize = max(blocksize, gs)
    assert in_f % gs == 0 and blocksize % gs == 0
    dev = W.device

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    idx = torch.arange(in_f, device=dev)
    damp = percdamp * torch.mean(torch.diag(H))
    Hinv = None
    for _ in range(8):
        Hd = H.clone()
        Hd[idx, idx] += damp
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except RuntimeError:
            damp = damp * 10
    if Hinv is None:
        raise RuntimeError("GPTQ: Hessian is not positive definite even after damping")
    del Hd

    ng = in_f // gs
    codes = torch.empty(out_f, in_f, dtype=torch.uint8, device=dev)
    scales = torch.empty(out_f, ng, dtype=torch.float16, device=dev)
    zeros = torch.empty(out_f, ng, dtype=torch.uint8, device=dev)
    Q = torch.empty(out_f, in_f, dtype=qdtype, device=dev) if want_fake else None

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % gs == 0:
                s, z = _group_params(W1[:, i:i + gs], maxq)
                scales[:, col // gs] = s.squeeze(1).half()
                zeros[:, col // gs] = z.squeeze(1).to(torch.uint8)
            w = W1[:, i]
            q = torch.clamp(torch.round(w.unsqueeze(1) / s) + z, 0, maxq)
            deq = ((q - z) * s).squeeze(1)
            codes[:, col] = q.squeeze(1).to(torch.uint8)
            if want_fake:
                Q[:, col] = deq.to(qdtype)
            err1 = (w - deq) / Hinv1[i, i]
            W1[:, i:].addr_(err1, Hinv1[i, i:], alpha=-1)
            Err1[:, i] = err1
        if i2 < in_f:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes, scales, zeros, Q


class _Hess:
    def __init__(self, in_f, dev):
        self.H = torch.zeros(in_f, in_f, dtype=torch.float32, device=dev)
        self.n = 0

    def add(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        self.H.addmm_(x.t(), x)
        self.n += x.shape[0]

    def get(self):
        return self.H / max(self.n, 1)


# ----------------------------------------------------------------------------- helpers
def _get(root, path):
    for p in path.split("."):
        root = getattr(root, p)
    return root


def _set(root, path, new):
    parent, _, child = path.rpartition(".")
    setattr(_get(root, parent) if parent else root, child, new)


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _calib_segments(name, n, seqlen, seed=0):
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")      # train split only
    except (PermissionError, OSError, ConnectionError, ValueError):
        cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        pat = os.path.join(cache, "wikitext/wikitext-2-raw-v1/*/*/wikitext-train.arrow")
        ds = Dataset.from_file(sorted(glob.glob(pat))[0])
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seqlen - 1, (n,), generator=g).tolist()
    return torch.stack([ids[s:s + seqlen] for s in starts])


@torch.no_grad()
def _rtn_fake(w, bits, gs=GROUP_SIZE, chunk_rows=4096):
    maxq = 2 ** bits - 1
    out = torch.empty_like(w)
    for r0 in range(0, w.shape[0], chunk_rows):
        x = w[r0:r0 + chunk_rows].float().reshape(-1, gs)
        mn = x.amin(1, keepdim=True).clamp(max=0)
        mx = x.amax(1, keepdim=True).clamp(min=0)
        s = ((mx - mn) / maxq).clamp_min(1e-5)
        z = torch.round(-mn / s).clamp(0, maxq)
        q = (torch.round(x / s) + z).clamp(0, maxq)
        out[r0:r0 + chunk_rows] = ((q - z) * s).reshape(-1, w.shape[1]).to(w.dtype)
    return out


@torch.no_grad()
def _calib_loss(model, segs, dev):
    tot = 0.0
    for b in segs:
        ids = b.unsqueeze(0).to(dev)
        logits = model(ids).logits.float()
        tot += F.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
        del logits
    return tot / len(segs)


_ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
_MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


@torch.no_grad()
def _plan_bits(model, segs, dev, low_bits=2, high_bits=3, target=None):
    target = TARGET_RATE if target is None else target
    layers = model.model.layers
    blocks = {}
    for i, layer in enumerate(layers):
        blocks[f"{i}.attn"] = [_get(layer, p) for p in _ATTN]
        blocks[f"{i}.mlp"] = [_get(layer, p) for p in _MLP]
    blocks["head"] = [model.lm_head]
    n_params = {k: sum(m.weight.numel() for m in v) for k, v in blocks.items()}

    base = _calib_loss(model, segs, dev)
    dloss = {}
    for key, mods in blocks.items():
        saved = [m.weight for m in mods]
        for m in mods:
            m.weight = nn.Parameter(_rtn_fake(m.weight.data, low_bits), requires_grad=False)
        dloss[key] = _calib_loss(model, segs, dev) - base
        for m, w in zip(mods, saved):
            m.weight = w
    _empty_cache()

    total = sum(n_params.values())

    def bits_of(n, b):                       # packed codes + fp16 scale + uint8 zero per group
        return n * b + (n // GROUP_SIZE) * 24

    plan = {k: high_bits for k in blocks}

    def rate():
        return 1.0 - sum(bits_of(n_params[k], plan[k]) for k in plan) / (16.0 * total)

    for key in sorted(blocks, key=lambda k: dloss[k] / n_params[k]):
        if rate() >= target:
            break
        plan[key] = low_bits
    n2 = sum(n_params[k] for k in plan if plan[k] == low_bits)
    print(f"[gptq] baseline calib loss {base:.4f}; {sum(v == low_bits for v in plan.values())} blocks at "
          f"{low_bits} bit ({n2 / total:.1%} of weights), the rest at {high_bits} bit; "
          f"estimated compression rate {rate():.2%}")
    print(f"[gptq] {low_bits}-bit blocks:", ", ".join(k for k in plan if plan[k] == low_bits))
    return plan


# ----------------------------------------------------------------------------- main pipeline
class _Stop(Exception):
    pass


@torch.no_grad()
def _run_gptq(model, segs, plan, dev):
    layers = model.model.layers
    n, L = segs.shape
    dtype = next(model.parameters()).dtype
    H_dim = model.config.hidden_size
    inps = torch.zeros(n, L, H_dim, dtype=dtype, device=dev)
    outs = torch.zeros_like(inps)

    # capture the input of layer 0 and the keyword arguments the decoder layers are called with
    store, cnt = {}, [0]

    def pre(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        inps[cnt[0]].copy_(hs[0])
        if "kw" not in store:
            store["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise _Stop

    handle = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
    for j in range(n):
        cnt[0] = j
        try:
            model(segs[j:j + 1].to(dev))
        except _Stop:
            pass
    handle.remove()
    kw = store["kw"]

    steps = [("attn", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
             ("attn", ["self_attn.o_proj"]),
             ("mlp", ["mlp.gate_proj", "mlp.up_proj"]),
             ("mlp", ["mlp.down_proj"])]

    for li, layer in enumerate(layers):
        stored = {}
        for kind, paths in steps:
            bits = plan[f"{li}.{kind}"]
            mods = [_get(layer, p) for p in paths]
            hess = _Hess(mods[0].in_features, dev)
            hook = mods[0].register_forward_hook(lambda m, a, o, hs=hess: hs.add(a[0]))
            for j in range(n):
                layer(inps[j:j + 1], **kw)
            hook.remove()
            W = torch.cat([m.weight.data for m in mods], 0).float()
            codes, scales, zeros, Q = _gptq(W, hess.get(), bits, GROUP_SIZE, dtype)
            del W, hess
            r0 = 0
            for p, m in zip(paths, mods):
                r1 = r0 + m.out_features
                m.weight.data.copy_(Q[r0:r1])          # later steps see the quantized weights
                stored[p] = (codes[r0:r1], scales[r0:r1], zeros[r0:r1], bits)
                r0 = r1
            del Q
        for j in range(n):                              # outputs of the quantized layer feed the next one
            o = layer(inps[j:j + 1], **kw)
            outs[j].copy_((o[0] if isinstance(o, tuple) else o)[0])
        for p, (codes, scales, zeros, bits) in stored.items():
            old = _get(layer, p)
            new = PackedLowBitLinear(old.in_features, old.out_features, bits, GROUP_SIZE,
                                     _pack_codes(codes, bits), scales.clone(), zeros.clone(), old.bias)
            _set(layer, p, new)
        del stored
        inps, outs = outs, inps
        _empty_cache()
        print(f"[gptq] layer {li + 1}/{len(layers)} done", flush=True)

    # lm_head: its input is the final norm output
    head = model.lm_head
    hess = _Hess(head.in_features, dev)
    for j in range(n):
        hess.add(model.model.norm(inps[j:j + 1]))
    W = head.weight.data.float()
    codes, scales, zeros, _ = _gptq(W, hess.get(), plan["head"], GROUP_SIZE, dtype, want_fake=False)
    del W, hess
    model.lm_head = PackedLowBitLinear(head.in_features, head.out_features, plan["head"], GROUP_SIZE,
                                       _pack_codes(codes, plan["head"]), scales, zeros, head.bias)
    del head, codes
    del inps, outs
    _empty_cache()


@torch.no_grad()
def quantize_mixed(model, low_bits, high_bits, target_rate, segs=None):
    """GPTQ with a per-block bit plan: every block gets `high_bits`, the least sensitive blocks are lowered
    to `low_bits` until the compression rate of common.py reaches target_rate. Returns #layers replaced."""
    dev = next(model.parameters()).device
    if segs is None:
        name = (getattr(model.config, "_name_or_path", None) or getattr(model.config, "name_or_path", None)
                or "Qwen/Qwen3-4B")
        segs = _calib_segments(name, N_SAMPLES, SEQLEN)
    plan = _plan_bits(model, segs[:PROBE_SEGS], dev, low_bits, high_bits, target_rate)
    _run_gptq(model, segs, plan, dev)
    return sum(1 for m in model.modules() if isinstance(m, PackedLowBitLinear))


@torch.no_grad()
def quantize_int2(model, segs=None):
    """Case 2: mixed 2/3-bit."""
    return quantize_mixed(model, 2, 3, TARGET_RATE, segs)
