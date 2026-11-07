"""Case 4: unstructured pruning (SparseGPT-style) + 2/3-bit GPTQ quantization of the kept weights.

Storage of one Linear (everything is counted by common.py):
  * mask_bits : 1 bit per original weight (packed, LSB first, 1 = kept)       [counted: _is_bitmask_pruned_linear]
  * values    : the kept weights as packed b-bit codes in row-major order     [counted: _is_bitmask_pruned_linear]
  * params    : fp16 scale + uint8 zero per (row, group of GROUP original columns) [counted: _is_packed_int2_linear]
Pipeline: calibration windows (WikiText-2 *train*) -> sensitivity probe -> bit plan (2 or 3 bit per block so that
the overall compression reaches TARGET_RATE) -> layer-by-layer SparseGPT+GPTQ with the OBS error compensation.
Every Linear is pruned by (about) PRUNE_FRACTION: per block of GROUP columns the weights with the smallest
saliency w^2 / [H^-1]_jj^2 are removed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from compression_method.quantization_int2 import (
    N_SAMPLES, SEQLEN, _CHUNK_ELEMS, _Hess, _Stop, _calib_loss, _calib_segments, _empty_cache, _get,
    _group_params, _pack_codes, _rtn_fake, _set)

PRUNE_FRACTION = 0.5     # fraction of weights removed from every Linear (assignment: >= 20%)
GROUP = 256              # columns per quantization group; also the block used for mask selection
TARGET_RATE = 0.851      # overall compression (common.py) to reach; the assignment requires >= 85%
PROBE_SEGS = 2

_ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
_MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
_STEPS = [("attn", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
          ("attn", ["self_attn.o_proj"]),
          ("mlp", ["mlp.gate_proj", "mlp.up_proj"]),
          ("mlp", ["mlp.down_proj"])]


# ----------------------------------------------------------------------------- storage
class PrunedQuantParams(nn.Module):
    """Scales and zeros of a pruned layer, counted by common.py through the _is_packed_int2_linear branch."""
    _is_packed_int2_linear = True

    def __init__(self, scales, zeros):
        super().__init__()
        self.register_buffer("qweight", torch.zeros(0, dtype=torch.uint8, device=scales.device), persistent=True)
        self.register_buffer("scales", scales.contiguous(), persistent=True)   # fp16 (out, in/GROUP)
        self.register_buffer("zeros", zeros.contiguous(), persistent=True)     # uint8 (out, in/GROUP)
        self.register_parameter("bias", None)


def _pack_mask(keep):
    """bool (rows, in), in % 8 == 0 -> uint8 stream, 8 weights per byte, LSB first."""
    rows, in_f = keep.shape
    shifts = torch.arange(8, device=keep.device, dtype=torch.int32)
    out = []
    step = max(1, _CHUNK_ELEMS // in_f)
    for r0 in range(0, rows, step):
        v = keep[r0:r0 + step].reshape(-1, 8).to(torch.int32)
        out.append((v << shifts).sum(1, dtype=torch.int32).to(torch.uint8))
    return torch.cat(out)


class PrunedQuantLinear(nn.Module):
    # common.py counts mask_bits + values; get_pruning_fraction() uses pruned_count / total_count
    _is_bitmask_pruned_linear = True

    def __init__(self, in_features, out_features, bits, group_size, mask_bits, values, params, pruned_count,
                 bias=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        self.pruned_count, self.total_count = int(pruned_count), in_features * out_features
        self.register_buffer("mask_bits", mask_bits, persistent=True)
        self.register_buffer("values", values, persistent=True)
        self.params = params
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def _dequantize(self, dtype):
        out_f, in_f, g, bits = self.out_features, self.in_features, self.group_size, self.bits
        ng = in_f // g
        dev = self.mask_bits.device
        acc = torch.int64 if bits > 3 else torch.int32
        bshift = torch.arange(8, device=dev, dtype=torch.int32)
        cshift = torch.arange(8, device=dev, dtype=acc) * bits
        cmask = (1 << bits) - 1
        weight = torch.empty(out_f, in_f, dtype=dtype, device=dev)
        step = max(1, _CHUNK_ELEMS // in_f)
        offset = 0                                   # kept weights of the previous row chunks
        for r0 in range(0, out_f, step):
            r1 = min(r0 + step, out_f)
            pk = self.mask_bits[r0 * in_f // 8: r1 * in_f // 8].to(torch.int32)
            keep = ((pk.unsqueeze(1) >> bshift) & 1).reshape(r1 - r0, in_f).bool()
            pos = torch.cumsum(keep.reshape(-1), 0)  # 1-based index of each kept weight inside this chunk
            cnt = int(pos[-1])
            g0, g1 = offset // 8, (offset + cnt + 7) // 8
            raw = self.values[g0 * bits: g1 * bits].reshape(-1, bits).to(acc)
            x = raw[:, 0]
            for k in range(1, bits):
                x = x | (raw[:, k] << (8 * k))
            codes = ((x.unsqueeze(1) >> cshift) & cmask).reshape(-1)[offset - g0 * 8: offset - g0 * 8 + cnt]
            dense = codes[(pos - 1).clamp_min(0)].reshape(r1 - r0, ng, g).to(torch.float32)
            z = self.params.zeros[r0:r1].to(torch.float32).unsqueeze(-1)
            s = self.params.scales[r0:r1].to(torch.float32).unsqueeze(-1)
            w = ((dense - z) * s).reshape(r1 - r0, in_f).masked_fill(~keep, 0.0)
            weight[r0:r1] = w.to(dtype)
            offset += cnt
        return weight

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self._dequantize(x.dtype), bias)


def _make_module(W_shape, bits, codes, scales, zeros, keep, bias):
    out_f, in_f = W_shape
    stream = codes[keep]                                           # kept codes, row-major
    pad = (-stream.numel()) % 8
    if pad:
        stream = torch.cat([stream, torch.zeros(pad, dtype=torch.uint8, device=stream.device)])
    values = _pack_codes(stream.reshape(-1, 8), bits)
    return PrunedQuantLinear(in_f, out_f, bits, GROUP, _pack_mask(keep), values, PrunedQuantParams(scales, zeros),
                             int((~keep).sum()), bias)


# ----------------------------------------------------------------------------- SparseGPT + GPTQ
@torch.no_grad()
def _sparse_gptq(W, H, bits, gs, prune, qdtype, want_fake=True, percdamp=0.01):
    """W float32 (out, in), modified in place; H float32 (in, in).
    Per block of gs columns: prune the `prune` fraction of weights with the smallest saliency, quantize the kept
    ones with GPTQ and propagate every error with the inverse Hessian.
    Returns codes (uint8), scales (fp16), zeros (uint8), keep mask (bool) and the dequantized weight."""
    maxq = 2 ** bits - 1
    out_f, in_f = W.shape
    assert in_f % gs == 0
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
        raise RuntimeError("SparseGPT: Hessian is not positive definite even after damping")
    del Hd

    ng = in_f // gs
    codes = torch.empty(out_f, in_f, dtype=torch.uint8, device=dev)
    keep = torch.empty(out_f, in_f, dtype=torch.bool, device=dev)
    scales = torch.empty(out_f, ng, dtype=torch.float16, device=dev)
    zeros = torch.empty(out_f, ng, dtype=torch.uint8, device=dev)
    Q = torch.empty(out_f, in_f, dtype=qdtype, device=dev) if want_fake else None

    for i1 in range(0, in_f, gs):
        i2 = i1 + gs
        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]
        k = int(W1.numel() * prune)
        pm = torch.zeros(W1.numel(), dtype=torch.bool, device=dev)
        if k > 0:
            sal = (W1.pow(2) / torch.diag(Hinv1).pow(2).unsqueeze(0)).reshape(-1)
            pm[torch.argsort(sal)[:k]] = True
            del sal
        pm = pm.reshape(out_f, gs)
        s, z = _group_params(W1.masked_fill(pm, 0.0), maxq)
        scales[:, i1 // gs] = s.squeeze(1).half()
        zeros[:, i1 // gs] = z.squeeze(1).to(torch.uint8)
        Err1 = torch.zeros_like(W1)
        for i in range(gs):
            w = W1[:, i]
            q = torch.clamp(torch.round(w.unsqueeze(1) / s) + z, 0, maxq)
            deq = ((q - z) * s).squeeze(1).masked_fill(pm[:, i], 0.0)       # pruned weights become exactly 0
            codes[:, i1 + i] = q.squeeze(1).to(torch.uint8)
            if want_fake:
                Q[:, i1 + i] = deq.to(qdtype)
            err1 = (w - deq) / Hinv1[i, i]
            W1[:, i:].addr_(err1, Hinv1[i, i:], alpha=-1)
            Err1[:, i] = err1
        keep[:, i1:i2] = ~pm
        if i2 < in_f:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes, scales, zeros, keep, Q


# ----------------------------------------------------------------------------- bit plan
@torch.no_grad()
def _fake_pruned(w, prune, bits, chunk_rows=2048):
    """Magnitude pruning per row + RTN quantization (cheap proxy used only for the sensitivity probe)."""
    out = torch.empty_like(w)
    k = int(w.shape[1] * prune)
    for r0 in range(0, w.shape[0], chunk_rows):
        x = w[r0:r0 + chunk_rows]
        if k > 0:
            a = x.abs().float()
            thr = torch.kthvalue(a, k, dim=1, keepdim=True).values
            x = torch.where(a <= thr, torch.zeros_like(x), x)
        out[r0:r0 + chunk_rows] = x
    return _rtn_fake(out, bits, GROUP)


@torch.no_grad()
def _plan_bits(model, segs, dev, prune, target):
    layers = model.model.layers
    blocks = {}
    for i, layer in enumerate(layers):
        blocks[f"{i}.attn"] = [_get(layer, p) for p in _ATTN]
        blocks[f"{i}.mlp"] = [_get(layer, p) for p in _MLP]
    blocks["head"] = [model.lm_head]

    def stored_bits(mods, b):
        tot = 0
        for m in mods:
            o, i = m.weight.shape
            n = o * i
            kept = n - int(o * GROUP * prune) * (i // GROUP)
            tot += n + kept * b + o * (i // GROUP) * 24      # mask + codes + fp16 scale + uint8 zero
        return tot

    n_params = {k: sum(m.weight.numel() for m in v) for k, v in blocks.items()}
    base = _calib_loss(model, segs, dev)
    dloss = {}
    for key, mods in blocks.items():
        saved = [m.weight for m in mods]
        for m in mods:
            m.weight = nn.Parameter(_fake_pruned(m.weight.data, prune, 2), requires_grad=False)
        dloss[key] = _calib_loss(model, segs, dev) - base
        for m, w in zip(mods, saved):
            m.weight = w
    _empty_cache()

    total16 = 16.0 * sum(n_params.values())
    plan = {k: 3 for k in blocks}

    def rate():
        return 1.0 - sum(stored_bits(blocks[k], plan[k]) for k in plan) / total16

    for key in sorted(blocks, key=lambda k: dloss[k] / n_params[k]):
        if rate() >= target:
            break
        plan[key] = 2
    n2 = sum(n_params[k] for k in plan if plan[k] == 2)
    print(f"[prune] baseline calib loss {base:.4f}; prune {prune:.0%}; {sum(v == 2 for v in plan.values())} blocks "
          f"with 2-bit values ({n2 / sum(n_params.values()):.1%} of weights), the rest 3-bit; "
          f"estimated compression rate {rate():.2%}")
    if rate() < target:
        print("[prune] WARNING: target compression cannot be reached with this prune fraction; raise it.")
    return plan


# ----------------------------------------------------------------------------- pipeline
@torch.no_grad()
def apply_pruning(model, prune_fraction=PRUNE_FRACTION, segs=None, target=TARGET_RATE):
    """Prune + quantize every nn.Linear. Returns the number of replaced layers."""
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if prune_fraction < 0.2:
        print("[prune] WARNING: prune fraction below 0.2; the overall 85% compression target is out of reach.")
    if segs is None:
        name = (getattr(model.config, "_name_or_path", None) or getattr(model.config, "name_or_path", None)
                or "Qwen/Qwen3-4B")
        segs = _calib_segments(name, N_SAMPLES, SEQLEN)
    plan = _plan_bits(model, segs[:PROBE_SEGS], dev, prune_fraction, target)

    layers = model.model.layers
    n, L = segs.shape
    inps = torch.zeros(n, L, model.config.hidden_size, dtype=dtype, device=dev)
    outs = torch.zeros_like(inps)
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

    for li, layer in enumerate(layers):
        stored = {}
        for kind, paths in _STEPS:
            bits = plan[f"{li}.{kind}"]
            mods = [_get(layer, p) for p in paths]
            hess = _Hess(mods[0].in_features, dev)
            hook = mods[0].register_forward_hook(lambda m, a, o, hs=hess: hs.add(a[0]))
            for j in range(n):
                layer(inps[j:j + 1], **kw)
            hook.remove()
            H = hess.get()
            del hess
            for p, m in zip(paths, mods):
                codes, scales, zeros, keep, Q = _sparse_gptq(m.weight.data.float(), H.clone(), bits, GROUP,
                                                             prune_fraction, dtype)
                m.weight.data.copy_(Q)                  # later steps see the pruned + quantized weights
                stored[p] = _make_module(tuple(m.weight.shape), bits, codes, scales, zeros, keep, m.bias)
                del codes, scales, zeros, keep, Q
            del H
        for j in range(n):
            o = layer(inps[j:j + 1], **kw)
            outs[j].copy_((o[0] if isinstance(o, tuple) else o)[0])
        for p, new in stored.items():
            _set(layer, p, new)
        del stored
        inps, outs = outs, inps
        _empty_cache()
        print(f"[prune] layer {li + 1}/{len(layers)} done", flush=True)

    head = model.lm_head
    hess = _Hess(head.in_features, dev)
    for j in range(n):
        hess.add(model.model.norm(inps[j:j + 1]))
    codes, scales, zeros, keep, _ = _sparse_gptq(head.weight.data.float(), hess.get(), plan["head"], GROUP,
                                                 prune_fraction, dtype, want_fake=False)
    model.lm_head = _make_module(tuple(head.weight.shape), plan["head"], codes, scales, zeros, keep, head.bias)
    del head, hess, codes, scales, zeros, keep, inps, outs
    _empty_cache()
    return sum(1 for m in model.modules() if isinstance(m, PrunedQuantLinear))
