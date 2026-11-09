"""Tensor-parallel Idol / TripoSplat denoiser (comfy.ldm.triposplat.model.LatentSeqMMFlowModel) across two GPUs
without P2P (Kaggle 2x T4). Correctness trial, modelled on h3_tensor_parallel.py.

Only the 24 main `blocks` (UnifiedTransformerBlock) run tensor parallel; noise/context refiners, RePo rope generators,
input/cond/camera/output layers and the shared adaLN stay unchanged on GPU0.

Per block (Megatron style, two ranks, one process):
  attn.qkv   column-parallel by heads (8 of 16 heads per rank, rows regrouped [q|k|v]), bias sliced alike
  attn.out   row-parallel (input columns of the rank's heads), fp16 partial sums swapped through pinned host
  mlp.mlp.0  column-parallel (half the FFN rows), GELU per rank
  mlp.mlp.2  row-parallel, partial sums swapped
  replicated: LayerNorm (no affine), modulation (shared t_mod + per-block shift_table), fp32 residual.
RoPE comes from GPU0 per block (input dependent, 16 heads); each rank gets its head slice.
Blocks are overridden per instance (`blk.forward`); the original forward is kept for the TP-CHECK reference.
"""
import contextlib
import gc
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

from .tp_source import open_weights

import comfy.cli_args
import comfy.model_management
import comfy.patcher_extension
import comfy.weight_adapter
import comfy_kitchen.backends.cuda as ck_cuda

try:
    import comfy.ldm.triposplat.model as tsm
    _TSM_ERR = None
except Exception as _e:  # ComfyUI without TripoSplat: supports() is False
    tsm, _TSM_ERR = None, _e

try:
    import comfy.model_prefetch as _prefetch
except Exception:
    _prefetch = None

logger = logging.getLogger("MultiGPU")

WRAPPER_KEY = "idol_tp"
GROUP = 256
DEVICES = (torch.device("cuda:0"), torch.device("cuda:1"))
# XCHG_OFF (env MULTIGPU_TP_XCHG_OFF=1, or set idol_tensor_parallel.XCHG_OFF = True at run time): the row-parallel
# partial sums are NOT exchanged; every rank keeps only its own partial. The output is wrong on purpose.
# Default is False (exchange on). Do not reassign it anywhere else in this module.
XCHG_OFF = os.environ.get("MULTIGPU_TP_XCHG_OFF", "0") == "1"
# IDOL_TP_CHECK=0 disables the block-0 comparison against the original single-GPU block.
CHECK = os.environ.get("IDOL_TP_CHECK", "1") != "0"
LORA_TARGETS = ("attn.qkv", "attn.out", "mlp.mlp.0", "mlp.mlp.2")
# With the exchange on, both replicas add the same fp16 partials in fp32: drift should be ~0.
DRIFT_WARN = 1e-3

_INSTANCES = {}
_SHARD_CACHE = {}


# ------------------------------------------------------------------------------------------------------- helpers

def _record(dev, stream=None):
    ev = torch.cuda.Event()
    ev.record(stream if stream is not None else torch.cuda.current_stream(dev))
    return ev


def _per_rank(fn):
    out = []
    for r, dev in enumerate(DEVICES):
        with torch.cuda.device(dev):
            out.append(fn(r))
    return out


def _rows(t, chunk, part, rank):
    """Rows [rank*part, (rank+1)*part) of every consecutive `chunk`-row group, concatenated."""
    if t.shape[0] % chunk:
        raise ValueError(f"cannot regroup {tuple(t.shape)} in chunks of {chunk}")
    return torch.cat([t[j * chunk + rank * part: j * chunk + (rank + 1) * part] for j in range(t.shape[0] // chunk)])


def _scale_rows(s, rows, chunk, part, rank):
    """Column-parallel slice of a weight scale: tensorwise (one value) stays, per-row follows the weight rows."""
    if s.numel() == 1:
        return s.clone()
    if s.shape[0] != rows:
        raise ValueError(f"weight scale shape {tuple(s.shape)} matches neither tensorwise nor {rows} rows")
    return _rows(s, chunk, part, rank).contiguous()


def _lin(x, w):
    out = ck_cuda.int8_linear(x, w[0], w[1], convrot=True, convrot_groupsize=GROUP)
    return out if out.dtype == torch.float16 else out.to(torch.float16)


def _pause():
    fn = getattr(_prefetch, "pause_malloc_graph", None) if _prefetch is not None else None
    return fn(sync=True) if fn is not None else contextlib.nullcontext()


def _atc():
    c = getattr(tsm, "AttentionTensorContainer", None)
    if c is None:
        from comfy.ldm.modules.attention import AttentionTensorContainer as c
    return c


# ------------------------------------------------------------------------------------------------------ exchange

class _Exchange:
    """Swap row-parallel partial sums between the ranks through pinned host buffers (no P2P on Kaggle T4s).
    Copies run on the compute streams; events make rank r wait until the other rank has read host[r] before
    overwriting it in the next block."""

    def __init__(self):
        self.bufs = {}

    def swap(self, phase, parts):
        if XCHG_OFF:   # module global, read at call time
            return None
        shape, dtype = tuple(parts[0].shape), parts[0].dtype
        if tuple(parts[1].shape) != shape or parts[1].dtype != dtype:
            raise RuntimeError(f"idol TP exchange: rank shapes differ {shape} vs {tuple(parts[1].shape)}")
        key = (phase, shape, dtype)
        buf = self.bufs.get(key)
        if buf is None:
            buf = self.bufs[key] = {"host": [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in DEVICES],
                                    "read": [None, None]}
        sent = []
        for r, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                s = torch.cuda.current_stream(dev)
                if buf["read"][1 - r] is not None:   # the other rank finished reading host[r] (previous block)
                    s.wait_event(buf["read"][1 - r])
                buf["host"][r].copy_(parts[r], non_blocking=True)
                sent.append(_record(dev))
        got = []
        for r, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                s = torch.cuda.current_stream(dev)
                s.wait_event(sent[1 - r])
                dst = torch.empty(shape, dtype=dtype, device=dev)
                dst.copy_(buf["host"][1 - r], non_blocking=True)
                buf["read"][r] = _record(dev)
                got.append(dst)
        return got


# -------------------------------------------------------------------------------------------------------- shards

def _int8_pair(f, layers, lin, shape):
    """(int8 weight, scale) on the CPU for linear `lin`; dense weights are requantized to int8 convrot on GPU0."""
    conf = layers.get(lin)
    if conf:
        if conf.get("format") != "int8_tensorwise" or not conf.get("convrot") or int(conf.get("convrot_groupsize", 0)) != GROUP:
            raise ValueError(f"idol TP needs int8 convrot group {GROUP} or dense weights; {lin} is {conf}")
        q, s = f.get_tensor(lin + ".weight"), f.get_tensor(lin + ".weight_scale")
    else:
        dense = f.get_tensor(lin + ".weight")
        if not dense.is_floating_point() or dense.dim() != 2:
            raise ValueError(f"idol TP: {lin}.weight is {dense.dtype} {tuple(dense.shape)} without int8 metadata")
        with torch.cuda.device(DEVICES[0]):
            d = dense.to(DEVICES[0], torch.float16)
            q, s = ck_cuda.quantize_int8_convrot_weight(d, GROUP)
            q, s = q.cpu(), s.float().cpu().reshape(-1, 1)
            del d
    if q.dtype != torch.int8 or tuple(q.shape) != tuple(shape):
        raise ValueError(f"idol TP: {lin} int8 weight is {q.dtype} {tuple(q.shape)}, expected int8 {tuple(shape)}")
    if s.numel() not in (1, shape[0]):
        raise ValueError(f"idol TP: {lin} weight_scale {tuple(s.shape)} is neither tensorwise nor per-row")
    return q, s.float()


def _head_norm(gamma, rank, dev):
    """Upstream MultiHeadRMSNorm with the rank's head slice of the per-head gamma [heads, head_dim]."""
    heads, dim = gamma.shape
    hr = heads // 2
    g = gamma[rank * hr:(rank + 1) * hr].contiguous()
    m = tsm.MultiHeadRMSNorm(dim, hr, dtype=g.dtype, device=dev)
    p = getattr(m, "gamma", None)
    if not isinstance(p, torch.Tensor) or tuple(p.shape) != (hr, dim):
        raise ValueError(f"idol TP: MultiHeadRMSNorm({dim}, {hr}).gamma is {None if p is None else tuple(p.shape)}, "
                         f"expected {(hr, dim)}")
    with torch.no_grad():
        p.copy_(g.to(dev))
    m.requires_grad_(False)
    m.eval()
    return m


def _load_shards(path, n_blocks):
    """Per-rank shards of the main blocks from `path` (checkpoint path or tp_source.ModelWeights).
    Returns [rank0_blocks, rank1_blocks]; each block is a dict of device tensors / per-rank head norms."""
    cached = _SHARD_CACHE.get(path)
    if cached is not None and all(r is not None and len(r) == n_blocks for r in cached):
        logger.info("[TP] reusing resident Idol shards on both GPUs")
        return cached
    if torch.cuda.device_count() < 2:
        raise RuntimeError("idol tensor parallel needs two CUDA devices")
    if tsm is None:
        raise ImportError(f"comfy.ldm.triposplat.model unavailable: {_TSM_ERR}")
    started = time.perf_counter()
    for dev in DEVICES:
        comfy.model_management.free_memory(1 << 30, dev)
    with open_weights(path) as f:
        meta = f.metadata() or {}
        layers = json.loads(meta.get("_quantization_metadata") or '{"layers": {}}').get("layers", {})
        keys = set(f.keys())
        for name in ("attn.q_norm.gamma", "attn.k_norm.gamma", "shift_table", "attn.qkv.bias", "attn.out.bias",
                     "mlp.mlp.0.bias", "mlp.mlp.2.bias"):
            if f"blocks.0.{name}" not in keys or f"blocks.{n_blocks - 1}.{name}" not in keys:
                raise KeyError(f"idol TP: blocks.*.{name} missing from the weight source")
        shards = [[], []]
        for i in range(n_blocks):
            p = f"blocks.{i}."
            g = lambda k: f.get_tensor(p + k)
            qg, kg = g("attn.q_norm.gamma"), g("attn.k_norm.gamma")
            if qg.dim() != 2 or qg.shape != kg.shape or qg.shape[0] % 2:
                raise ValueError(f"idol TP: q/k norm gamma {tuple(qg.shape)} / {tuple(kg.shape)} is not [even heads, head_dim]")
            heads, hd = qg.shape
            inner = heads * hd
            qkv_b, out_b, fc1_b, fc2_b = g("attn.qkv.bias"), g("attn.out.bias"), g("mlp.mlp.0.bias"), g("mlp.mlp.2.bias")
            ffn = fc1_b.shape[0]
            if inner % (2 * GROUP) or ffn % (2 * GROUP) or qkv_b.shape[0] != 3 * inner or out_b.shape[0] != inner:
                raise ValueError(f"idol TP: block {i} dims inner={inner} ffn={ffn} not splittable on {GROUP}-groups")
            qkv, qkv_s = _int8_pair(f, layers, p + "attn.qkv", (3 * inner, inner))
            out, out_s = _int8_pair(f, layers, p + "attn.out", (inner, inner))
            fc1, fc1_s = _int8_pair(f, layers, p + "mlp.mlp.0", (ffn, inner))
            fc2, fc2_s = _int8_pair(f, layers, p + "mlp.mlp.2", (inner, ffn))
            st = g("shift_table")
            if tuple(st.shape) != (1, 6 * inner):
                raise ValueError(f"idol TP: block {i} shift_table {tuple(st.shape)}, expected (1, {6 * inner})")
            half, hf = inner // 2, ffn // 2
            for rank, dev in enumerate(DEVICES):
                with torch.cuda.device(dev):
                    shards[rank].append({
                        "heads": heads // 2, "head_dim": hd,
                        "qkv": (_rows(qkv, inner, half, rank).contiguous().to(dev),
                                _scale_rows(qkv_s, 3 * inner, inner, half, rank).to(dev)),
                        "qkv_b": _rows(qkv_b, inner, half, rank).to(dev, torch.float16),
                        "out": (out[:, rank * half:(rank + 1) * half].contiguous().to(dev), out_s.clone().to(dev)),
                        "out_b": out_b.to(dev, torch.float32),
                        "fc1": (_rows(fc1, ffn, hf, rank).contiguous().to(dev),
                                _scale_rows(fc1_s, ffn, ffn, hf, rank).to(dev)),
                        "fc1_b": fc1_b[rank * hf:(rank + 1) * hf].contiguous().to(dev, torch.float16),
                        "fc2": (fc2[:, rank * hf:(rank + 1) * hf].contiguous().to(dev), fc2_s.clone().to(dev)),
                        "fc2_b": fc2_b.to(dev, torch.float32),
                        "shift_table": st.to(dev),
                        "qn": _head_norm(qg, rank, dev), "kn": _head_norm(kg, rank, dev),
                    })
    _SHARD_CACHE[path] = shards
    logger.info("[TP] sharded %d blocks", n_blocks)
    logger.info("[MultiGPU TP] Idol: %d blocks ready over %s in %.2f s (%.1f MiB allocated on cuda:1)", n_blocks,
                ", ".join(map(str, DEVICES)), time.perf_counter() - started, torch.cuda.memory_allocated(DEVICES[1]) / 2**20)
    return shards


# ---------------------------------------------------------------------------------------------------------- LoRA

def _lora_factors(entries):
    """(down with strength * alpha / rank folded in, up) of plain LoRA patches on one weight, stacked along the rank."""
    downs, ups = [], []
    for entry in entries:
        strength, adapter, strength_model, offset, function = entry[:5]
        w = adapter.weights if isinstance(adapter, comfy.weight_adapter.LoRAAdapter) else None
        if w is None or strength_model != 1.0 or offset is not None or function is not None or w[3] is not None or w[4] is not None:
            raise ValueError("Idol tensor parallel supports plain LoRA only (no LoCon mid, DoRA, LoHa/LoKr, diffs, "
                             "offsets or model strength != 1)")
        down, up = (v.flatten(1).to(device="cpu", dtype=torch.float32) for v in (w[1], w[0]))
        alpha = float(w[2]) / down.shape[0] if w[2] is not None else 1.0
        downs.append(down * (strength * alpha))
        ups.append(up)
    return torch.cat(downs), torch.cat(ups, dim=1)


def _shard_lora(name, down, up, rank):
    if name == "attn.qkv":
        inner = up.shape[0] // 3
        return down, _rows(up, inner, inner // 2, rank)
    if name == "mlp.mlp.0":
        hf = up.shape[0] // 2
        return down, up[rank * hf:(rank + 1) * hf]
    half = down.shape[1] // 2   # attn.out / mlp.mlp.2: row-parallel, the rank's input columns
    return down[:, rank * half:(rank + 1) * half], up


def _lora(x, pair):
    return F.linear(F.linear(x, pair[0]), pair[1])


# ------------------------------------------------------------------------------------------------------ validate

def _validate(dm):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("idol tensor parallel needs two CUDA devices")
    if tsm is None:
        raise ImportError(f"comfy.ldm.triposplat.model unavailable: {_TSM_ERR}")
    if not callable(getattr(tsm, "attention", None)):
        raise AttributeError("comfy.ldm.triposplat.model.attention not found")
    if not getattr(dm, "share_mod", False):
        raise ValueError("idol TP expects share_mod=True (blocks receive adaLN(t_emb) from the model)")
    blocks = dm.blocks
    if len(blocks) == 0:
        raise ValueError("idol TP: no blocks")
    b0 = blocks[0]
    a0 = b0.attn
    sig0 = (a0.num_heads, a0.head_dim, a0.channels, bool(a0.use_rope), bool(a0.qk_rms_norm), type(b0.mlp.mlp[1]))
    for i, b in enumerate(blocks):
        a = b.attn
        if type(b) is not type(b0) or (a.num_heads, a.head_dim, a.channels, bool(a.use_rope), bool(a.qk_rms_norm),
                                       type(b.mlp.mlp[1])) != sig0:
            raise ValueError(f"idol TP: block {i} differs from block 0")
        if not b.modulation or not b.share_mod or getattr(b, "adaLN_modulation", None) is not None:
            raise ValueError(f"idol TP: block {i} is not a shared-modulation block")
        if getattr(b.norm1, "weight", None) is not None or getattr(b.norm2, "weight", None) is not None:
            raise ValueError(f"idol TP: block {i} LayerNorm has affine weights (not sharded)")
        seq = b.mlp.mlp
        if not isinstance(seq, torch.nn.Sequential) or len(seq) != 3 or any(True for _ in seq[1].parameters()):
            raise ValueError(f"idol TP: block {i} MLP is not Linear-activation-Linear")
    heads, hd, ch = sig0[:3]
    if heads % 2 or heads * hd != ch or (ch // 2) % GROUP:
        raise ValueError(f"idol TP: heads={heads} head_dim={hd} channels={ch} not splittable in two on {GROUP}-groups")
    ffn = getattr(b0.mlp.mlp[0], "out_features", None)
    if ffn is None or (ffn // 2) % GROUP or ffn % 2:
        raise ValueError(f"idol TP: FFN width {ffn} not splittable")
    if not a0.qk_rms_norm or getattr(tsm, "MultiHeadRMSNorm", None) is None:
        raise ValueError("idol TP expects q/k MultiHeadRMSNorm (its gamma gives the per-rank head count)")
    if a0.use_rope and not callable(getattr(tsm, "apply_rope", None)):
        raise AttributeError("comfy.ldm.triposplat.model.apply_rope not found")


# ------------------------------------------------------------------------------------------------------ instance

def _make_forward(tp, i, orig):
    def forward(x, mod=None, rotary_emb=None, transformer_options=None, **kwargs):
        if kwargs:
            raise TypeError(f"idol TP block {i}: unexpected arguments {sorted(kwargs)}")
        if _INSTANCES.get(tp.key) is not tp:   # detached by tp_auto: restore the model, one GPU, logged
            logger.warning("[MultiGPU TP] Idol TP instance detached; restoring original block forwards (one GPU)")
            tp.restore_forwards()
            return orig(x, mod=mod, rotary_emb=rotary_emb, transformer_options=transformer_options)
        return tp.block(i, x, mod, rotary_emb, transformer_options, orig)
    forward._idol_tp = tp
    forward._idol_orig = orig
    return forward


class IdolTensorParallel:
    def __init__(self, path, diffusion_model, patcher, key):
        _validate(diffusion_model)
        self.path = path
        self.key = key
        self.patcher = patcher
        self.blocks = diffusion_model.blocks
        self.shards = None
        self.xchg = _Exchange()
        blk = self.blocks[0]
        self.eps1, self.eps2 = blk.norm1.eps, blk.norm2.eps
        self.heads, self.head_dim = blk.attn.num_heads, blk.attn.head_dim
        self.use_rope = bool(blk.attn.use_rope)
        self.lora, self.lora_entries, self.lora_sig = {}, {}, None
        self.checked_key = None
        self.last_check = None
        self.xchg_warned = None   # last XCHG_OFF state that was announced in the log

    # -------------------------------------------------------------------------------------------- forwards
    def install_forwards(self):
        for i, blk in enumerate(self.blocks):
            cur = blk.__dict__.get("forward")
            orig = cur._idol_orig if cur is not None and hasattr(cur, "_idol_orig") else blk.forward
            blk.forward = _make_forward(self, i, orig)

    def restore_forwards(self):
        for blk in self.blocks:
            cur = blk.__dict__.get("forward")
            if cur is not None and hasattr(cur, "_idol_orig"):
                del blk.forward

    # ------------------------------------------------------------------------------------------ memory
    def release(self, drop_shards=False):
        for d in DEVICES:
            torch.cuda.synchronize(d)
        self.xchg.bufs.clear()
        self.lora = {}
        if drop_shards:
            _SHARD_CACHE.pop(self.path, None)
            self.shards = None
            self.lora_entries, self.lora_sig = {}, None
            logger.warning("[MultiGPU TP] Idol shards released; the next forward re-shards them")
        gc.collect()
        for d in DEVICES:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if host_empty is not None:
            host_empty()

    # -------------------------------------------------------------------------------------------- LoRA
    def set_lora(self, patches):
        entries, bad = {}, []
        for key, value in (patches or {}).items():
            if not isinstance(key, str):
                bad.append(str(key))
                continue
            parts = key.split(".")
            if len(parts) < 4 or parts[0] != "diffusion_model" or parts[1] != "blocks":
                continue   # refiners / embedders / adaLN run on ComfyUI's own patched modules on GPU0
            name = ".".join(parts[3:-1])
            if name in LORA_TARGETS and parts[-1] == "weight":
                entries[(int(parts[2]), name)] = value
            else:
                bad.append(key)
        if bad:
            raise ValueError("Idol tensor parallel cannot apply these block patches: " + str(bad[:5]))
        sig = tuple(sorted((k, tuple((id(p[1]), p[0], p[2], id(p[3]), id(p[4])) for p in v)) for k, v in entries.items()))
        if sig == self.lora_sig:
            return
        by_block = {}
        for (i, name), value in entries.items():
            _lora_factors(value)   # validate now; raises on anything but plain LoRA
            by_block.setdefault(i, {})[name] = value
        self.lora_entries, self.lora, self.lora_sig = by_block, {}, sig
        if entries:
            logger.info("[MultiGPU TP] Idol LoRA: %d block linears as side paths", len(entries))

    def _block_lora(self, i):
        entries = self.lora_entries.get(i)
        if not entries:
            return ({}, {})
        if i not in self.lora:
            pair = ({}, {})
            for name, value in entries.items():
                down, up = _lora_factors(value)
                for r, dev in enumerate(DEVICES):
                    d, u = _shard_lora(name, down, up, r)
                    pair[r][name] = (d.to(dev, torch.float16).contiguous(), u.to(dev, torch.float16).contiguous())
            self.lora[i] = pair
        return self.lora[i]

    # ---------------------------------------------------------------------------------------- per rank
    def _ensure_ready(self, i):
        n = len(self.blocks)
        if self.shards is None or len(self.shards) != 2 or any(r is None or len(r) != n for r in self.shards):
            with _pause():
                self.shards = _load_shards(self.path, n)
        if i == 0 or self.lora_sig is None:
            self.set_lora(getattr(self.patcher, "patches", None) or {})

    def _rope_part(self, rope, r, dev):
        if rope is None:
            raise ValueError("idol TP: block uses rope but rotary_emb is None")
        if torch.is_tensor(rope):
            if rope.dim() < 3:
                raise ValueError(f"idol TP: rope tensor {tuple(rope.shape)} has no head axis at dim 2")
            hr = self.heads // 2
            if rope.shape[2] == self.heads:
                t = rope[:, :, r * hr:(r + 1) * hr]
            elif rope.shape[2] == 1:
                t = rope
            else:
                raise ValueError(f"idol TP: rope tensor {tuple(rope.shape)}: dim 2 is neither {self.heads} heads nor 1")
            return t.to(dev).contiguous()
        if isinstance(rope, (tuple, list)):
            return type(rope)(self._rope_part(t, r, dev) for t in rope)
        raise TypeError(f"idol TP: unsupported rope type {type(rope).__name__}")

    def _attn_partial(self, x, mods, sh, rope, topts, lo, blk):
        B, L, C = x.shape
        hr, hd = sh["heads"], sh["head_dim"]
        h = F.layer_norm(x, (C,), eps=self.eps1)
        h = torch.addcmul(mods[0], h, 1 + mods[1]).to(torch.float16).reshape(B * L, C)
        qkv = _lin(h, sh["qkv"])
        if "attn.qkv" in lo:
            qkv = qkv + _lora(h, lo["attn.qkv"])
        qkv = (qkv + sh["qkv_b"]).view(B, L, 3, hr, hd)
        del h
        q, k, v = qkv.unbind(2)
        if self.use_rope:
            q, k = tsm.apply_rope(q, k, rope)
        q = sh["qn"](q)
        k = sh["kn"](k)
        del qkv
        atc = _atc()
        o = tsm.attention(atc(q), atc(k), atc(v), transformer_options=topts, preferred_attention=blk.attn.comfy_attention)
        if tuple(o.shape) != (B, L, hr, hd):
            raise RuntimeError(f"idol TP: attention returned {tuple(o.shape)}, expected {(B, L, hr, hd)}")
        o = o.reshape(B * L, hr * hd).to(torch.float16).contiguous()
        part = _lin(o, sh["out"])
        if "attn.out" in lo:
            part = part + _lora(o, lo["attn.out"])
        return part.contiguous()

    def _mlp_partial(self, x, mods, sh, lo, blk):
        B, L, C = x.shape
        h = F.layer_norm(x, (C,), eps=self.eps2)
        h = torch.addcmul(mods[3], h, 1 + mods[4]).to(torch.float16).reshape(B * L, C)
        a = _lin(h, sh["fc1"])
        if "mlp.mlp.0" in lo:
            a = a + _lora(h, lo["mlp.mlp.0"])
        a = blk.mlp.mlp[1](a + sh["fc1_b"]).to(torch.float16).contiguous()
        part = _lin(a, sh["fc2"])
        if "mlp.mlp.2" in lo:
            part = part + _lora(a, lo["mlp.mlp.2"])
        return part.contiguous()

    @staticmethod
    def _residual(x, own, got, bias, gate):
        t = own.float()
        if got is not None:
            t = t + got.float()   # fp32 add of the same two fp16 partials: identical on both ranks
        t = t + bias
        x.addcmul_(t.view(x.shape), gate)

    # ------------------------------------------------------------------------------------------- block
    def block(self, i, x, mod, rope, topts, orig):
        with torch.no_grad():
            return self._block(i, x, mod, rope, topts, orig)

    def _block(self, i, x, mod, rope, topts, orig):
        d0, d1 = DEVICES
        if x.device != d0:
            raise RuntimeError(f"idol TP expects the block input on {d0}, got {x.device}")
        if x.dim() != 3 or mod is None:
            raise ValueError(f"idol TP: block {i} needs x [B,L,C] and mod (got {tuple(x.shape)}, mod={mod is not None})")
        self._ensure_ready(i)
        xchg_off = bool(XCHG_OFF)
        if i == 0 and self.xchg_warned != xchg_off:
            self.xchg_warned = xchg_off
            if xchg_off:
                logger.warning("[MultiGPU TP] Idol XCHG_OFF is set: row-parallel partial sums are NOT exchanged; "
                               "output is wrong on purpose (diagnostic/timing only)")
            else:
                logger.info("[MultiGPU TP] Idol row-parallel exchange on")
        blk = self.blocks[i]
        sh = (self.shards[0][i], self.shards[1][i])
        ck = (id(self.shards), xchg_off)
        ref = None
        if CHECK and i == 0 and self.checked_key != ck:
            ref = orig(x.clone(), mod=mod, rotary_emb=rope, transformer_options=topts).float()
        lo = self._block_lora(i)

        modf = (mod + sh[0]["shift_table"].to(mod.device, mod.dtype)).float()
        mods0 = [m.unsqueeze(1) for m in modf.chunk(6, dim=1)]
        x0 = x.to(torch.float32, copy=True)
        with torch.cuda.device(d1):
            mods1 = [m.to(d1) for m in mods0]
            x1 = x.to(d1).to(torch.float32)
            rope1 = self._rope_part(rope, 1, d1) if self.use_rope else None
        rope0 = self._rope_part(rope, 0, d0) if self.use_rope else None
        xs, mods, ropes = (x0, x1), (mods0, mods1), (rope0, rope1)

        own = _per_rank(lambda r: self._attn_partial(xs[r], mods[r], sh[r], ropes[r], topts, lo[r], blk))
        got = self.xchg.swap("attn", own)
        _per_rank(lambda r: self._residual(xs[r], own[r], None if got is None else got[r], sh[r]["out_b"], mods[r][2]))
        del own, got
        own = _per_rank(lambda r: self._mlp_partial(xs[r], mods[r], sh[r], lo[r], blk))
        got = self.xchg.swap("mlp", own)
        _per_rank(lambda r: self._residual(xs[r], own[r], None if got is None else got[r], sh[r]["fc2_b"], mods[r][5]))
        del own, got

        if ref is not None:
            rel = float((x0 - ref).norm() / ref.norm().clamp_min(1e-30))
            drift = float((x0 - x1.to(d0)).abs().max())
            finite = bool(torch.isfinite(x0).all())
            self.checked_key = ck
            self.last_check = {"rel_err": rel, "drift": drift, "finite": finite, "xchg_off": xchg_off}
            logger.info("[TP-CHECK] block0 rel_err=%.6e drift=%.6e finite=%s", rel, drift, finite)
            if not xchg_off and drift > DRIFT_WARN:
                logger.warning("[MultiGPU TP] Idol replica drift %.3e with the exchange on: partial sums were not "
                               "shared (or added twice) between the ranks", drift)
        return x0.to(x.dtype)


# --------------------------------------------------------------------------------------------------- module API

def release_gpu(drop_shards=True):
    """Free exchange buffers and LoRA copies and, with drop_shards, the shards (re-made on the next forward)."""
    for tp in _INSTANCES.values():
        tp.release(drop_shards)


def _dm_wrapper(executor, *args, **kwargs):
    # execute(x, t, context, ref_latents, transformer_options, ...): the model's own block weights are unused by TP
    topts = kwargs.get("transformer_options", args[4] if len(args) > 4 else None)
    if isinstance(topts, dict):
        topts["prefetch_dynamic_vbars"] = False
    return executor(*args, **kwargs)


def _prepare_sampling(tp, executor, *args, **kwargs):
    try:
        mp = getattr(getattr(executor, "class_obj", None), "model_patcher", None)
        if mp is not None:
            tp.patcher = mp
        tp.set_lora(getattr(tp.patcher, "patches", None) or {})
        return executor(*args, **kwargs)
    finally:
        tp.release()


def supports(dm):
    if tsm is None:
        return False
    cls, blk_cls = getattr(tsm, "LatentSeqMMFlowModel", None), getattr(tsm, "UnifiedTransformerBlock", None)
    if cls is None or blk_cls is None or not isinstance(dm, cls):
        return False
    blocks = getattr(dm, "blocks", None)
    return blocks is not None and len(blocks) > 0 and isinstance(blocks[0], blk_cls)


def attach(model, source, key):
    """Run the main blocks of `model` (a ModelPatcher) on both GPUs; shards from `source` (path or ModelWeights)."""
    dm = model.model.diffusion_model
    if not supports(dm):
        raise ValueError("idol_tensor_parallel only supports comfy.ldm.triposplat LatentSeqMMFlowModel")
    comfy.cli_args.args.disable_comfy_compiler = True   # the Comfy compiler records one device's allocations
    for other_key, other in list(_INSTANCES.items()):
        if other_key != key and other.shards is not None:
            other.release(drop_shards=True)
    tp = _INSTANCES.get(key)
    if tp is None:
        tp = IdolTensorParallel(source, dm, model, key)
        _INSTANCES[key] = tp
    else:
        _validate(dm)
        tp.restore_forwards()
        tp.blocks = dm.blocks
        tp.patcher = model
    tp.install_forwards()
    dm._mgpu_tp = tp
    if not model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY):
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _dm_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY,
                                   lambda executor, *a, **k: _prepare_sampling(tp, executor, *a, **k))
    return tp


def detach(model_or_dm):
    """Restore the original block forwards and drop the instance of this diffusion model."""
    dm = getattr(getattr(model_or_dm, "model", None), "diffusion_model", model_or_dm)
    for k, tp in list(_INSTANCES.items()):
        if tp.blocks is getattr(dm, "blocks", None):
            tp.restore_forwards()
            tp.release(drop_shards=True)
            _INSTANCES.pop(k, None)
    if getattr(dm, "_mgpu_tp", None) is not None:
        dm._mgpu_tp = None
