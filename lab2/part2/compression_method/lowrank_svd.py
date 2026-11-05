"""Case 3: truncation-aware (whitened) truncated SVD of every nn.Linear + 8-bit group quantization of the factors.

W (out x in) ~= left (out x r) @ right (r x in).
  * The rank r of every Linear is chosen so that the SVD alone saves at least SVD_SAVING of that layer's
    weights (r*(in+out) <= (1-SVD_SAVING)*in*out), rounded down to a multiple of GROUP.
  * Plain SVD ignores which input directions matter; here the SVD is taken on W @ L, where L is the Cholesky
    factor of the calibration Gram matrix X^T X, so the truncation minimizes ||(W - W')X|| (SVD-LLM idea).
    Calibration data: random windows of the WikiText-2 *train* split. Layers are processed one after another,
    each seeing the outputs of the already compressed layers before it.
  * Both factors are stored as 8-bit codes with an fp16 scale and a uint8 zero per group of GROUP values.
    Every stored buffer is counted by common.py (the factors are Int8GroupLinear modules).
Engine helpers (calibration data, Hessian accumulation) come from quantization_int2.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from compression_method.quantization_int2 import (
    N_SAMPLES, SEQLEN, _CHUNK_ELEMS, _Hess, _Stop, _calib_segments, _empty_cache, _get, _set)

SVD_SAVING = 0.205      # every Linear saves at least this fraction of its weights through the SVD alone
GROUP = 64              # quantization group size; ranks are multiples of this
PERCDAMP = 1e-4         # relative damping of the Gram matrix before the Cholesky factorization


# ----------------------------------------------------------------------------- 8-bit factor storage
def _quant8(M):
    """float32 (rows, cols), cols % GROUP == 0 -> uint8 codes (rows, cols), fp16 scales, uint8 zeros."""
    rows, cols = M.shape
    x = M.reshape(rows, cols // GROUP, GROUP)
    mn = x.amin(-1, keepdim=True).clamp(max=0)
    mx = x.amax(-1, keepdim=True).clamp(min=0)
    s = ((mx - mn) / 255).clamp_min(6.2e-5).half().float()
    z = torch.round(-mn / s).clamp(0, 255)
    q = (torch.round(x / s) + z).clamp(0, 255)
    return (q.reshape(rows, cols).to(torch.uint8), s.squeeze(-1).half(), z.squeeze(-1).to(torch.uint8))


def _deq8(codes, scales, zeros):
    rows, cols = codes.shape
    q = codes.reshape(rows, cols // GROUP, GROUP).float()
    return ((q - zeros.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)).reshape(rows, cols)


class Int8GroupLinear(nn.Module):
    # common.py counts qweight + scales + zeros for this flag
    _is_packed_int2_linear = True

    def __init__(self, in_features, out_features, qweight, scales, zeros, bias=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", qweight.contiguous(), persistent=True)   # uint8 (out, in)
        self.register_buffer("scales", scales.contiguous(), persistent=True)     # fp16 (out, in/GROUP)
        self.register_buffer("zeros", zeros.contiguous(), persistent=True)       # uint8 (out, in/GROUP)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def _dequantize(self, dtype):
        out_f, in_f = self.out_features, self.in_features
        weight = torch.empty(out_f, in_f, dtype=dtype, device=self.qweight.device)
        step = max(1, _CHUNK_ELEMS // in_f)
        for r0 in range(0, out_f, step):
            r1 = min(r0 + step, out_f)
            weight[r0:r1] = _deq8(self.qweight[r0:r1], self.scales[r0:r1], self.zeros[r0:r1]).to(dtype)
        return weight

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self._dequantize(x.dtype), bias)


class SVDInt8Linear(nn.Module):
    """x -> right (in -> r) -> left (r -> out). The two Int8GroupLinear children are counted by common.py."""
    _is_svd_int8_linear = True

    def __init__(self, right, left):
        super().__init__()
        self.right, self.left = right, left
        self.in_features, self.out_features = right.in_features, left.out_features
        self.rank = right.out_features

    def forward(self, x):
        return self.left(self.right(x))


# ----------------------------------------------------------------------------- whitened SVD
@torch.no_grad()
def _svd_factors(W, H, r, percdamp=PERCDAMP):
    """Best rank-r factorization of W (out, in) for inputs with Gram matrix H (in, in), all float32.
    Returns left (out, r), right (r, in) with the norms of left's columns and right's rows balanced."""
    out_f, in_f = W.shape
    idx = torch.arange(in_f, device=W.device)
    damp = percdamp * torch.mean(torch.diag(H))
    Lc = None
    for _ in range(10):
        Hd = H.clone()
        Hd[idx, idx] += damp
        try:
            Lc = torch.linalg.cholesky(Hd)
            break
        except RuntimeError:
            damp = damp * 10
    if Lc is None:
        raise RuntimeError("SVD: Gram matrix is not positive definite even after damping")
    del Hd
    Wp = W @ Lc                                            # whitened weight; ||(W-W')X|| = ||(Wp-Wp')||
    if in_f <= out_f:
        _, evecs = torch.linalg.eigh(Wp.T @ Wp)
        V = evecs[:, -r:]                                  # top-r right singular vectors (in, r)
        left0 = Wp @ V                                     # (out, r)
        right0 = torch.linalg.solve_triangular(Lc, V.T.contiguous(), upper=False, left=False)   # V^T Lc^-1
    else:
        _, evecs = torch.linalg.eigh(Wp @ Wp.T)
        U = evecs[:, -r:]                                  # top-r left singular vectors (out, r)
        left0 = U
        right0 = U.T @ W                                   # U^T Wp Lc^-1 = U^T W
    ln = left0.norm(dim=0).clamp_min(1e-12)
    rn = right0.norm(dim=1).clamp_min(1e-12)
    s = (rn / ln).sqrt()                                   # equalize column / row norms for quantization
    return left0 * s.unsqueeze(0), right0 / s.unsqueeze(1)


def _rank_for(in_f, out_f, rank_fraction, svd_saving):
    if rank_fraction is not None:
        r = int(min(in_f, out_f) * rank_fraction)
    else:
        r = int((1.0 - svd_saving) * in_f * out_f / (in_f + out_f))
    return max(GROUP, min(r // GROUP * GROUP, min(in_f, out_f) // GROUP * GROUP))


def _compress(W, H, r):
    """-> (right_q, left_q, dense approximation built from the *quantized* factors)."""
    left, right = _svd_factors(W, H, r)
    lq, rq = _quant8(left), _quant8(right)
    dense = _deq8(*lq) @ _deq8(*rq)
    return rq, lq, dense


def _make_module(in_f, out_f, rq, lq, bias):
    right = Int8GroupLinear(in_f, rq[0].shape[0], rq[0], rq[1], rq[2])
    left = Int8GroupLinear(lq[0].shape[1], out_f, lq[0], lq[1], lq[2], bias)
    return SVDInt8Linear(right, left)


# ----------------------------------------------------------------------------- pipeline
@torch.no_grad()
def apply_svd(model, rank_fraction=None, svd_saving=SVD_SAVING, segs=None):
    """Replace every nn.Linear with a whitened low-rank, int8-quantized SVDInt8Linear. Returns the count.
    rank_fraction (optional): rank = rank_fraction * min(in, out) instead of the saving-based rule."""
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if segs is None:
        name = (getattr(model.config, "_name_or_path", None) or getattr(model.config, "name_or_path", None)
                or "Qwen/Qwen3-4B")
        segs = _calib_segments(name, N_SAMPLES, SEQLEN)
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

    steps = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
             ["self_attn.o_proj"],
             ["mlp.gate_proj", "mlp.up_proj"],
             ["mlp.down_proj"]]

    for li, layer in enumerate(layers):
        stored = {}
        for paths in steps:
            mods = [_get(layer, p) for p in paths]
            hess = _Hess(mods[0].in_features, dev)
            hook = mods[0].register_forward_hook(lambda m, a, o, hs=hess: hs.add(a[0]))
            for j in range(n):
                layer(inps[j:j + 1], **kw)
            hook.remove()
            H = hess.get()
            del hess
            for p, m in zip(paths, mods):
                r = _rank_for(m.in_features, m.out_features, rank_fraction, svd_saving)
                rq, lq, dense = _compress(m.weight.data.float(), H, r)
                m.weight.data.copy_(dense.to(dtype))        # later steps see the compressed weights
                stored[p] = (m.in_features, m.out_features, rq, lq, m.bias)
                del dense
            del H
        for j in range(n):
            o = layer(inps[j:j + 1], **kw)
            outs[j].copy_((o[0] if isinstance(o, tuple) else o)[0])
        for p, (in_f, out_f, rq, lq, bias) in stored.items():
            _set(layer, p, _make_module(in_f, out_f, rq, lq, bias))
        del stored
        inps, outs = outs, inps
        _empty_cache()
        print(f"[svd] layer {li + 1}/{len(layers)} done", flush=True)

    head = model.lm_head
    hess = _Hess(head.in_features, dev)
    for j in range(n):
        hess.add(model.model.norm(inps[j:j + 1]))
    r = _rank_for(head.in_features, head.out_features, rank_fraction, svd_saving)
    rq, lq, _ = _compress(head.weight.data.float(), hess.get(), r)
    model.lm_head = _make_module(head.in_features, head.out_features, rq, lq, head.bias)
    del head, hess, rq, lq, inps, outs
    _empty_cache()
    return sum(1 for m in model.modules() if isinstance(m, SVDInt8Linear))


def svd_param_saving(model):
    """Fraction of Linear weights saved by the rank truncation alone (before 8-bit quantization),
    plus the smallest per-layer saving."""
    kept = total = 0
    worst = 1.0
    for m in model.modules():
        if isinstance(m, SVDInt8Linear):
            full = m.in_features * m.out_features
            used = m.rank * (m.in_features + m.out_features)
            kept += used
            total += full
            worst = min(worst, 1.0 - used / full)
    return (1.0 - kept / total if total else 0.0), worst
