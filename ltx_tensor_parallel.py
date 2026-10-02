"""Tensor-parallel LTX 2.x audio-video DiT block across two GPUs without P2P (Kaggle 2x T4).

Megatron-style split of every BasicAVTransformerBlock linear: each attention by heads (to_q / to_k / to_v /
to_gate_logits by output rows, to_out by input columns, bias added once), each feed-forward by hidden width
(net.0.proj by output rows, net.2 by input columns). Both GPUs keep a replica of the video and audio residual
streams and compute every AdaLN / gate / norm on it; row-parallel partial sums and the all-head q/k RMSNorm
sums of squares are exchanged through pinned host buffers. Each rank adds the two partials in the same order,
so the replicas stay bit-identical.

Shards come from the int8 convrot checkpoint (int8 splits land on multiples of the 256-wide rotation group;
floating layers are split as fp16). block_forward mirrors BasicAVTransformerBlock.forward of ComfyUI 0.37.0
(inference path, no training branch). UNETLoaderLTXTensorParallel plugs it into sampling.
"""
import json
import logging
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.cli_args
import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.quant_ops
import comfy.sd
import folder_paths
import comfy_kitchen.backends.cuda as ck_cuda
from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock, CompressedTimestep
from comfy.ldm.lightricks.model import apply_rotary_emb, apply_rotary_emb_qk
from comfy.ldm.modules.attention import optimized_attention

logger = logging.getLogger("MultiGPU")

DEVICES = (torch.device("cuda:0"), torch.device("cuda:1"))
ATTNS = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")
FFS = ("ff", "audio_ff")
TABLES = ("scale_shift_table", "audio_scale_shift_table", "prompt_scale_shift_table", "audio_prompt_scale_shift_table",
          "scale_shift_table_a2v_ca_audio", "scale_shift_table_a2v_ca_video")
TIMING = os.environ.get("LTX_TP_TIMING", "0") == "1"
PIN_HOST = os.environ.get("LTX_TP_PIN_HOST", "1") != "0"  # GPU 1 shard host copy pinned (fast restore) or pageable  # sync around every exchange and report where the time goes


# ----------------------------------------------------------------------------------------------- shards

class Lin:
    __slots__ = ("w", "s", "b", "int8")

    def __init__(self, w, s, b, int8):
        self.w, self.s, self.b, self.int8 = w, s, b, int8


def _block_prefix(f):
    keys = [k for k in f.keys() if k.endswith("transformer_blocks.0.attn1.to_q.weight")]
    if len(keys) != 1:
        raise RuntimeError("cannot find transformer_blocks.0.attn1.to_q.weight in the checkpoint")
    return keys[0][:-len("0.attn1.to_q.weight")]


def _quant_conf(f, keys, meta, name):
    if name + ".comfy_quant" in keys:
        return json.loads(bytes(f.get_tensor(name + ".comfy_quant").tolist()).decode())
    short = name[name.index("transformer_blocks."):]
    hits = [v for k, v in meta.items() if k == name or k == short or k.endswith("." + short)]
    return hits[0] if len(hits) == 1 else {}


def _load_lin(f, keys, meta, name, axis, rank):
    """One rank's shard of linear `name`: axis 0 = column-parallel (output rows), 1 = row-parallel (input
    columns; bias only on rank 0, added once after the exchange)."""
    w = f.get_tensor(name + ".weight")
    b = f.get_tensor(name + ".bias") if name + ".bias" in keys else None
    n = w.shape[axis]
    if n % 2:
        raise RuntimeError("%s: odd dimension %d along axis %d" % (name, n, axis))
    lo, hi = rank * n // 2, (rank + 1) * n // 2
    if w.dtype == torch.int8:
        conf = _quant_conf(f, keys, meta, name)
        if conf.get("format") != "int8_tensorwise" or not conf.get("convrot") or int(conf.get("convrot_groupsize", 256)) != 256:
            raise RuntimeError("%s: tensor parallel needs int8_tensorwise convrot g256, got %s" % (name, conf))
        if axis == 1 and (hi - lo) % 256:
            raise RuntimeError("%s: row-parallel split %d is not a multiple of the 256 rotation group" % (name, hi - lo))
        s = f.get_tensor(name + ".weight_scale").float()
        if axis == 0 and s.numel() > 1:
            s = s.reshape(w.shape[0], -1)[lo:hi].reshape(-1) if s.dim() == 1 else s[lo:hi]
    else:
        w, s = w.half(), None
    w = w[lo:hi] if axis == 0 else w[:, lo:hi]
    if b is not None:
        b = b.half()
        b = b[lo:hi] if axis == 0 else (b if rank == 0 else None)
    return Lin(w.contiguous(), s, b, w.dtype == torch.int8)


def _block_cpu(f, keys, meta, pre, rank):
    """One rank's shard of the block at key prefix `pre`, on the CPU."""
    sh = {"tables": {}}
    for name in ATTNS:
        p = pre + name + "."
        a = {n: _load_lin(f, keys, meta, p + n, 0, rank) for n in ("to_q", "to_k", "to_v")}
        a["to_out"] = _load_lin(f, keys, meta, p + "to_out.0", 1, rank)
        a["gate"] = _load_lin(f, keys, meta, p + "to_gate_logits", 0, rank) if p + "to_gate_logits.weight" in keys else None
        inner = f.get_slice(p + "to_q.weight").get_shape()[0]
        lo, hi = rank * inner // 2, (rank + 1) * inner // 2
        a["q_norm"] = f.get_tensor(p + "q_norm.weight").half()[lo:hi].contiguous()
        a["k_norm"] = f.get_tensor(p + "k_norm.weight").half()[lo:hi].contiguous()
        a["inner"] = inner
        sh[name] = a
    for name in FFS:
        p = pre + name + ".net."
        sh[name] = {"proj": _load_lin(f, keys, meta, p + "0.proj", 0, rank), "out": _load_lin(f, keys, meta, p + "2", 1, rank)}
    for t in TABLES:
        if pre + t in keys:
            sh["tables"][t] = f.get_tensor(pre + t).half()
    return sh


def _open(path):
    f = safe_open(path, framework="pt", device="cpu")
    return f, set(f.keys()), json.loads((f.metadata() or {}).get("_quantization_metadata", "{}")).get("layers", {})


def load_block_shards(path, i):
    """[rank-0 shard, rank-1 shard] of transformer block i, on DEVICES[0] / DEVICES[1]."""
    f, keys, meta = _open(path)
    pre = _block_prefix(f) + "%d." % i
    return [_to(_block_cpu(f, keys, meta, pre, rank), dev) for rank, dev in enumerate(DEVICES)]


def load_all_shards(path, n_blocks, pin_rank1=True):
    """GPU shards [rank][block] for all blocks, plus the rank-1 shards as (pinned) host copies [block], which
    let GPU 1 give its memory back (VAE decode, text encoder) and get the shards back without reading the file."""
    f, keys, meta = _open(path)
    prefix = _block_prefix(f)
    gpu, host = [[], []], []
    t0 = time.perf_counter()
    for i in range(n_blocks):
        pre = prefix + "%d." % i
        gpu[0].append(_to(_block_cpu(f, keys, meta, pre, 0), DEVICES[0]))
        h = _block_cpu(f, keys, meta, pre, 1)
        if pin_rank1:
            h = _pin(h)
        host.append(h)
        gpu[1].append(_to(h, DEVICES[1]))
        if i % 12 == 0:
            logger.info("[MultiGPU LTX TP] sharded block %d/%d (%.0fs)", i, n_blocks, time.perf_counter() - t0)
    return gpu, host


def _pin(obj):
    if torch.is_tensor(obj):
        return obj.pin_memory()
    if isinstance(obj, Lin):
        return Lin(*(_pin(getattr(obj, k)) for k in ("w", "s", "b")), obj.int8)
    if isinstance(obj, dict):
        return {k: _pin(v) for k, v in obj.items()}
    return obj


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev, non_blocking=False)
    if isinstance(obj, Lin):
        return Lin(*(_to(getattr(obj, k), dev) for k in ("w", "s", "b")), obj.int8)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(v, dev) for v in obj)
    if isinstance(obj, CompressedTimestep):
        c = CompressedTimestep.__new__(CompressedTimestep)
        for k in CompressedTimestep.__slots__:
            setattr(c, k, _to(getattr(obj, k), dev))
        return c
    return obj


def shard_bytes(sh):
    total = 0
    stack = [sh]
    while stack:
        o = stack.pop()
        if torch.is_tensor(o):
            total += o.numel() * o.element_size()
        elif isinstance(o, Lin):
            stack += [o.w, o.s, o.b]
        elif isinstance(o, dict):
            stack += list(o.values())
    return total


# ----------------------------------------------------------------------------------------------- exchange

STATS = defaultdict(float)


def _ev(dev):
    e = torch.cuda.Event()
    e.record(torch.cuda.current_stream(dev))
    return e


CHUNKS = max(1, int(os.environ.get("LTX_TP_CHUNKS", "4")))
_CHUNK_MIN_BYTES = 16 << 20  # tensors below this go in one piece


class Exchange:
    """Sum of two per-rank tensors (B, T, ...) through pinned host buffers (no P2P), in token chunks on two side
    streams per GPU: one copies chunks out to the host, the other copies the other rank's chunks back in, so the
    two PCIe directions overlap (chunk j comes in while chunk j+1 goes out) and each landed chunk is added on the
    compute stream. The sum is done in place in parts[r] (rank 0: p0 += p1, rank 1: p1 += p0, the same values).
    Host buffers persist per (tag, shape, dtype, chunks); per-chunk events guard their reuse until the other rank
    has read them. No side-stream allocations: the receive buffer is allocated on the compute stream."""

    def __init__(self):
        self.bufs = {}
        self.streams = None

    def allreduce(self, parts, tag):
        t0 = time.perf_counter() if TIMING else None
        if TIMING:
            for d in DEVICES:
                torch.cuda.synchronize(d)
            t_ready = time.perf_counter()
        if self.streams is None:
            self.streams = [(torch.cuda.Stream(d), torch.cuda.Stream(d)) for d in DEVICES]
        shape, nbytes = tuple(parts[0].shape), parts[0].numel() * parts[0].element_size()
        k = CHUNKS if nbytes >= _CHUNK_MIN_BYTES and shape[1] >= CHUNKS else 1
        bounds = [(shape[1] * j // k, shape[1] * (j + 1) // k) for j in range(k)]
        key = (tag, shape, parts[0].dtype, k)
        buf = self.bufs.get(key)
        if buf is None:
            buf = self.bufs[key] = {"host": [torch.empty(shape, dtype=parts[0].dtype, pin_memory=True) for _ in DEVICES],
                                    "read": [None, None]}  # read[r][j]: rank r finished copying host[1-r] chunk j
        sent = [[None] * k for _ in DEVICES]
        for r, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                d2h = self.streams[r][0]
                d2h.wait_event(_ev(dev))  # parts[r] computed
                for j, (a, b) in enumerate(bounds):
                    if buf["read"][1 - r] is not None:
                        d2h.wait_event(buf["read"][1 - r][j])
                    with torch.cuda.stream(d2h):
                        buf["host"][r][:, a:b].copy_(parts[r][:, a:b], non_blocking=True)
                    sent[r][j] = torch.cuda.Event()
                    sent[r][j].record(d2h)
        for r, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                comp, h2d = torch.cuda.current_stream(dev), self.streams[r][1]
                recv = torch.empty_like(parts[r])
                h2d.wait_event(_ev(dev))  # recv allocated
                reads = []
                for j, (a, b) in enumerate(bounds):
                    h2d.wait_event(sent[1 - r][j])
                    with torch.cuda.stream(h2d):
                        recv[:, a:b].copy_(buf["host"][1 - r][:, a:b], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(h2d)
                    reads.append(ev)
                    comp.wait_event(ev)
                    comp.wait_event(sent[r][j])  # own chunk j is on the host before it is overwritten
                    parts[r][:, a:b] += recv[:, a:b]
                buf["read"][r] = reads
        out = parts
        if TIMING:
            for d in DEVICES:
                torch.cuda.synchronize(d)
            STATS["compute_" + tag] += t_ready - t0
            STATS["xchg_" + tag] += time.perf_counter() - t_ready
            STATS["xchg_bytes_" + tag] += parts[0].numel() * parts[0].element_size()
            STATS["n_" + tag] += 1
        return out


XCHG = Exchange()


# ----------------------------------------------------------------------------------------------- compute

def _lin(x, p, act=None):
    if p.int8:
        return ck_cuda.int8_linear(x, p.w, p.s, bias=p.b, convrot=True, convrot_groupsize=256, input_act=act)
    if act == "gelu_tanh":
        x = F.gelu(x, approximate="tanh")
    return F.linear(x, p.w, p.b)


def _ranks(fn):
    out = []
    for r, dev in enumerate(DEVICES):
        with torch.cuda.device(dev):
            out.append(fn(r))
    return out


def _ada(table, batch, timestep, indices=slice(None, None)):
    return BasicAVTransformerBlock.get_ada_values(None, table, batch, timestep, indices)


def _rms(x):
    return comfy.ldm.common_dit.rms_norm(x)


DEBUG = None  # dict: sub-layer name -> rank-0 output (set by the block test)


def _record(name, t):
    if DEBUG is not None:
        DEBUG[name] = t[0].detach().clone()


def _attention(sh, name, xs, ctxs, pe, k_pe, mask, topts, tag):
    """CrossAttention.forward split by heads; returns the per-rank full outputs (after the exchange)."""
    a = [sh[r][name] for r in range(2)]
    self_attn = ctxs is None
    ctxs = xs if ctxs is None else ctxs
    q = _ranks(lambda r: _lin(xs[r], a[r]["to_q"]))
    k = _ranks(lambda r: _lin(ctxs[r], a[r]["to_k"]))
    v = _ranks(lambda r: _lin(ctxs[r], a[r]["to_v"]))
    inner = a[0]["inner"]
    heads = HEADS[name] // 2
    if self_attn and topts[0].get("stg_skip_self_attn", False):
        out = v
    else:
        # q_norm / k_norm span all heads: exchange the per-token sums of squares of both halves
        tq = _ranks(lambda r: q[r].float().square().sum(-1, keepdim=True))
        tk = _ranks(lambda r: k[r].float().square().sum(-1, keepdim=True))
        nq = q[0].shape[1]
        tot = XCHG.allreduce(_ranks(lambda r: torch.cat((tq[r], tk[r]), 1)), tag + "_norm")
        q = _ranks(lambda r: (q[r].float() * torch.rsqrt(tot[r][:, :nq] / inner + 1e-5)).to(q[r].dtype) * a[r]["q_norm"])
        k = _ranks(lambda r: (k[r].float() * torch.rsqrt(tot[r][:, nq:] / inner + 1e-5)).to(k[r].dtype) * a[r]["k_norm"])
        if pe is not None:
            if k_pe is None and q[0].shape == k[0].shape:
                qk = _ranks(lambda r: apply_rotary_emb_qk(q[r], k[r], pe[r]))
                q, k = [x[0] for x in qk], [x[1] for x in qk]
            else:
                q = _ranks(lambda r: apply_rotary_emb(q[r], pe[r]))
                k = _ranks(lambda r: apply_rotary_emb(k[r], pe[r] if k_pe is None else k_pe[r]))
        out = _ranks(lambda r: optimized_attention(q[r], k[r], v[r], heads, mask=None if mask is None else mask[r],
                                                   transformer_options=topts[r]))
    if a[0]["gate"] is not None:
        def gated(r):
            g = 2.0 * torch.sigmoid(_lin(xs[r], a[r]["gate"]))
            b_, t_, _ = out[r].shape
            return (out[r].view(b_, t_, heads, -1) * g.unsqueeze(-1)).view(b_, t_, -1)
        out = _ranks(gated)
    part = _ranks(lambda r: _lin(out[r], a[r]["to_out"]))
    res = XCHG.allreduce(part, tag)
    _record(name, res)
    return res


def _ff(sh, name, xs, tag):
    f = [sh[r][name] for r in range(2)]
    h = _ranks(lambda r: _lin(xs[r], f[r]["proj"]))
    part = _ranks(lambda r: _lin(h[r], f[r]["out"], act="gelu_tanh"))
    res = XCHG.allreduce(part, tag)
    _record(name, res)
    return res


HEADS = {}


def set_heads(v_heads, a_heads):
    HEADS.update(attn1=v_heads, attn2=v_heads, audio_attn1=a_heads, audio_attn2=a_heads,
                 audio_to_video_attn=a_heads, video_to_audio_attn=a_heads)


def block_forward(sh, x, cross_attention_adaln, v_context=None, a_context=None, attention_mask=None, v_timestep=None,
                  a_timestep=None, v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None,
                  v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None, v_cross_gate_timestep=None,
                  a_cross_gate_timestep=None, transformer_options=None, self_attention_mask=None,
                  v_prompt_timestep=None, a_prompt_timestep=None):
    """BasicAVTransformerBlock.forward on both GPUs. Every argument except `sh` is a per-rank pair [rank0, rank1]
    (x = [(vx0, ax0), (vx1, ax1)]); pe arguments are per-rank head slices. Returns the per-rank (vx, ax)."""
    if attention_mask is not None and any(m is not None for m in attention_mask):
        raise NotImplementedError("tensor parallel: text attention masks are not supported yet")
    if self_attention_mask is not None and any(m is not None for m in self_attention_mask):
        raise NotImplementedError("tensor parallel: self-attention masks (guides/keyframes) are not supported yet")
    topts = transformer_options
    vx, ax = [x[r][0] for r in range(2)], [x[r][1] for r in range(2)]
    run_vx = topts[0].get("run_vx", True)
    run_ax = topts[0].get("run_ax", True) and ax[0].numel() > 0
    run_a2v = run_vx and topts[0].get("a2v_cross_attn", True) and ax[0].numel() > 0
    run_v2a = run_ax and topts[0].get("v2a_cross_attn", True)
    T = [sh[r]["tables"] for r in range(2)]
    B = vx[0].shape[0]

    def text_ca(xs, ctx, name, table, ptable, ts, pts, tag):
        if cross_attention_adaln:
            ada = _ranks(lambda r: _ada(T[r][table], B, ts[r], slice(6, 9)))
            def inputs(r):
                shift_q, scale_q, _ = ada[r]
                shift_kv, scale_kv = (T[r][ptable][None, None].to(device=xs[r].device, dtype=xs[r].dtype)
                                      + pts[r].reshape(B, pts[r].shape[1], 2, -1)).unbind(dim=2)
                return comfy.quant_ops.ck.rms_adaln(xs[r], scale_q, shift_q), ctx[r] * (1 + scale_kv) + shift_kv
            ins = _ranks(inputs)
            out = _attention(sh, name, [i[0] for i in ins], [i[1] for i in ins], None, None, None, topts, tag)
            return _ranks(lambda r: out[r] * ada[r][2])
        return _attention(sh, name, _ranks(lambda r: _rms(xs[r])), ctx, None, None, None, topts, tag)

    if run_vx:
        def pre1(r):
            shift, scale = _ada(T[r]["scale_shift_table"], B, v_timestep[r], slice(0, 2))
            return comfy.quant_ops.ck.rms_adaln(vx[r], scale, shift)
        a1 = _attention(sh, "attn1", _ranks(pre1), None, v_pe, None, None, topts, "v_attn1")
        _ranks(lambda r: vx[r].addcmul_(a1[r], _ada(T[r]["scale_shift_table"], B, v_timestep[r], slice(2, 3))[0]))
        del a1
        a2 = text_ca(vx, v_context, "attn2", "scale_shift_table", "prompt_scale_shift_table", v_timestep, v_prompt_timestep, "v_attn2")
        _ranks(lambda r: vx[r].add_(a2[r]))
        del a2

    if run_ax:
        def pre_a(r):
            shift, scale = _ada(T[r]["audio_scale_shift_table"], B, a_timestep[r], slice(0, 2))
            return _rms(ax[r]) * (1 + scale) + shift
        a1 = _attention(sh, "audio_attn1", _ranks(pre_a), None, a_pe, None, None, topts, "a_attn1")
        _ranks(lambda r: ax[r].addcmul_(a1[r], _ada(T[r]["audio_scale_shift_table"], B, a_timestep[r], slice(2, 3))[0]))
        del a1
        a2 = text_ca(ax, a_context, "audio_attn2", "audio_scale_shift_table", "audio_prompt_scale_shift_table",
                     a_timestep, a_prompt_timestep, "a_attn2")
        _ranks(lambda r: ax[r].add_(a2[r]))
        del a2

    if run_a2v or run_v2a:
        ax_norm3 = _ranks(lambda r: _rms(ax[r]))
        if run_a2v:
            def a2v_in(r):
                sa, sha = _ada(T[r]["scale_shift_table_a2v_ca_audio"][:4, :], ax[r].shape[0], a_cross_scale_shift_timestep[r])[:2]
                sv, shv = _ada(T[r]["scale_shift_table_a2v_ca_video"][:4, :], B, v_cross_scale_shift_timestep[r])[:2]
                return comfy.quant_ops.ck.rms_adaln(vx[r], sv, shv), ax_norm3[r] * (1 + sa) + sha
            ins = _ranks(a2v_in)
            o = _attention(sh, "audio_to_video_attn", [i[0] for i in ins], [i[1] for i in ins], v_cross_pe, a_cross_pe,
                           None, topts, "v_a2v")
            del ins
            _ranks(lambda r: vx[r].addcmul_(o[r], _ada(T[r]["scale_shift_table_a2v_ca_video"][4:, :], B, v_cross_gate_timestep[r])[0]))
            del o
        if run_v2a:
            def v2a_in(r):
                sa, sha = _ada(T[r]["scale_shift_table_a2v_ca_audio"][:4, :], ax[r].shape[0], a_cross_scale_shift_timestep[r])[2:4]
                sv, shv = _ada(T[r]["scale_shift_table_a2v_ca_video"][:4, :], B, v_cross_scale_shift_timestep[r])[2:4]
                return ax_norm3[r] * (1 + sa) + sha, comfy.quant_ops.ck.rms_adaln(vx[r], sv, shv)
            ins = _ranks(v2a_in)
            o = _attention(sh, "video_to_audio_attn", [i[0] for i in ins], [i[1] for i in ins], a_cross_pe, v_cross_pe,
                           None, topts, "a_v2a")
            del ins
            _ranks(lambda r: ax[r].addcmul_(o[r], _ada(T[r]["scale_shift_table_a2v_ca_audio"][4:, :], ax[r].shape[0], a_cross_gate_timestep[r])[0]))
            del o

    if run_vx:
        def pre_ff(r):
            shift, scale = _ada(T[r]["scale_shift_table"], B, v_timestep[r], slice(3, 5))
            return comfy.quant_ops.ck.rms_adaln(vx[r], scale, shift)
        o = _ff(sh, "ff", _ranks(pre_ff), "v_ff")
        _ranks(lambda r: vx[r].addcmul_(o[r], _ada(T[r]["scale_shift_table"], B, v_timestep[r], slice(5, 6))[0]))
        del o

    if run_ax:
        def pre_aff(r):
            shift, scale = _ada(T[r]["audio_scale_shift_table"], B, a_timestep[r], slice(3, 5))
            return _rms(ax[r]) * (1 + scale) + shift
        o = _ff(sh, "audio_ff", _ranks(pre_aff), "a_ff")
        _ranks(lambda r: ax[r].addcmul_(o[r], _ada(T[r]["audio_scale_shift_table"], B, a_timestep[r], slice(5, 6))[0]))
        del o

    return [(vx[r], ax[r]) for r in range(2)]


# ----------------------------------------------------------------------------------------------- per-rank inputs

def slice_pe(pe, rank):
    """A rank's head slice of a (rotation_matrix, split_pe) RoPE tuple (heads on dim 2; broadcast dim kept)."""
    if pe is None:
        return None
    rot, split = pe
    if rot.shape[2] == 1:
        return rot, split
    n = rot.shape[2] // 2
    return rot[:, :, rank * n:(rank + 1) * n].contiguous(), split


def replicate(obj, dev):
    return _to(obj, dev)


# ----------------------------------------------------------------------------------------------- sampling integration

_KW = ("v_context", "a_context", "attention_mask", "v_timestep", "a_timestep", "v_pe", "a_pe", "v_cross_pe", "a_cross_pe",
       "v_cross_scale_shift_timestep", "a_cross_scale_shift_timestep", "v_cross_gate_timestep", "a_cross_gate_timestep",
       "self_attention_mask", "v_prompt_timestep", "a_prompt_timestep")
_PE = ("v_pe", "a_pe", "v_cross_pe", "a_cross_pe")
_INSTANCES = {}


def _gpu_free():
    return " ".join("cuda:%d %.1f/%.1f GB free" % ((d.index,) + tuple(v / 2**30 for v in torch.cuda.mem_get_info(d))) for d in DEVICES)


class LTXTensorParallel:
    """Runs every LTX AV transformer block on both GPUs. Shards are built from the checkpoint on the first sampling
    run (the model's own block weights are never paged in) and stay resident; GPU 1's half also has a pinned host
    copy, so park() frees GPU 1 instantly (before a VAE decode) and unpark() restores it over PCIe."""

    def __init__(self, path, dm):
        self.path = path
        self.blocks = dm.transformer_blocks
        self.cadaln = bool(getattr(self.blocks[0], "cross_attention_adaln", False))
        set_heads(self.blocks[0].attn1.heads, self.blocks[0].audio_attn1.heads)
        self.gpu = None   # [rank][block]
        self.host = None  # rank-1 host copies [block]
        self.parked = False
        self.state = None
        self.checked = os.environ.get("LTX_TP_CHECK", "1") == "0"
        self.fallback_logged = False

    def _load(self):
        need = int(sum(p.numel() * p.element_size() for p in self.blocks.parameters()) / 2 * 1.05)
        for d in DEVICES:
            comfy.model_management.free_memory(need + (1 << 30), d)
        t0 = time.perf_counter()
        self.gpu, self.host = load_all_shards(self.path, len(self.blocks), PIN_HOST)
        logger.info("[MultiGPU LTX TP] %d blocks sharded in %.0fs: %.2f / %.2f GB on %s / %s, GPU 1 host copy %s | %s | %s",
                    len(self.blocks), time.perf_counter() - t0, sum(map(shard_bytes, self.gpu[0])) / 2**30,
                    sum(map(shard_bytes, self.gpu[1])) / 2**30, DEVICES[0], DEVICES[1],
                    "pinned" if PIN_HOST else "pageable", _gpu_free(), _ram())

    def park(self):
        """Drop GPU 1's shards (host copies kept) so another model can use GPU 1."""
        if self.gpu is None or self.parked:
            return
        torch.cuda.synchronize(DEVICES[1])
        self.gpu[1] = None
        self.state = None
        XCHG.bufs.clear()
        with torch.cuda.device(DEVICES[1]):
            torch.cuda.empty_cache()
        self.parked = True
        logger.info("[MultiGPU LTX TP] parked GPU 1 shards: %s", _gpu_free())

    def unpark(self):
        if not self.parked:
            return
        need = sum(map(shard_bytes, self.host))
        comfy.model_management.free_memory(need + (1 << 30), DEVICES[1])
        t0 = time.perf_counter()
        with torch.cuda.device(DEVICES[1]):
            self.gpu[1] = [_to(h, DEVICES[1]) for h in self.host]
            torch.cuda.synchronize(DEVICES[1])
        self.parked = False
        logger.info("[MultiGPU LTX TP] restored %.2f GB of GPU 1 shards in %.1fs: %s", need / 2**30, time.perf_counter() - t0, _gpu_free())

    def _inputs(self, i, args):
        """Per-rank copies of the block inputs, built at block 0 of every forward (the same for all its blocks)."""
        if i != 0 and self.state is not None:
            return self.state
        kws = []
        for r, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                kw = {}
                for k in _KW:
                    v = args.get(k)
                    if k in _PE:
                        v = slice_pe(v, r)
                    kw[k] = v if r == 0 else _to(v, dev)
                kws.append(kw)
        self.state = {"kw": {k: [kws[0][k], kws[1][k]] for k in _KW}, "x0": None, "x1": None}
        return self.state

    def block(self, i, args, original):
        if i == 0:
            for d in DEVICES:
                torch.cuda.synchronize(d)
            self.t_fwd = time.perf_counter()
            if not getattr(self, "first_block_logged", True):
                self.first_block_logged = True
                logger.info("[MultiGPU LTX TP] forward: prefetch_dynamic_vbars=%s (False = unused block weights not paged in)",
                            args["transformer_options"].get("prefetch_dynamic_vbars"))
        if self.gpu is None:
            with comfy.model_prefetch.pause_malloc_graph(sync=True):
                self._load()
        self.unpark()
        st = self._inputs(i, args)
        vx, ax = args["img"]
        if st["x0"] is not None and st["x0"][0] is vx and st["x0"][1] is ax:
            x1 = st["x1"]
        else:
            with torch.cuda.device(DEVICES[1]):
                x1 = (vx.to(DEVICES[1]), ax.to(DEVICES[1]))
        ref = None
        if not self.checked:
            ref = original({**args, "img": (vx.clone(), ax.clone())})["img"]
        topts = args["transformer_options"]
        try:
            out = block_forward([self.gpu[0][i], self.gpu[1][i]], [(vx, ax), x1], self.cadaln,
                                transformer_options=[topts, topts], **st["kw"])
        except NotImplementedError as e:
            if not self.fallback_logged:
                logger.warning("[MultiGPU LTX TP] %s: block %d runs on one GPU", e, i)
                self.fallback_logged = True
            st["x0"] = None
            return original(args)
        if ref is not None:
            self.checked = True
            for s, name in ((0, "video"), (1, "audio")):
                rel = float((out[0][s].float() - ref[s].float()).norm() / ref[s].float().norm().clamp_min(1e-20))
                drift = float((out[0][s].float() - out[1][s].float().to(DEVICES[0])).abs().max())
                logger.info("[MultiGPU LTX TP] block %d check, %s: rel err vs one GPU %.2e, replica drift %.2e", i, name, rel, drift)
        st["x0"], st["x1"] = out[0], out[1]
        if i == len(self.blocks) - 1:
            for d in DEVICES:
                torch.cuda.synchronize(d)
            logger.info("[MultiGPU LTX TP] forward: %d blocks, video %d tokens, %.2fs", len(self.blocks), vx.shape[1],
                        time.perf_counter() - self.t_fwd)
        return {"img": out[0]}


def park_for(device):
    """Called before a large decode on `device`: free the tensor-parallel shards there."""
    if torch.device(device) == DEVICES[1]:
        for tp in _INSTANCES.values():
            tp.park()


def _topts(args, kwargs):
    """transformer_options of a diffusion-model call: LTX passes (x, timestep, context, attention_mask, frame_rate,
    transformer_options, ...), so find it by content, not position."""
    t = kwargs.get("transformer_options")
    if isinstance(t, dict):
        return t
    for a in args:
        if isinstance(a, dict) and ("patches_replace" in a or "prefetch_dynamic_vbars" in a or "cond_or_uncond" in a):
            return a
    return None


_PREFETCH_LOGGED = []


def _no_block_prefetch(executor, *args, **kwargs):
    # the model's own block weights are unused; prefetching would page all of them in every step
    topts = _topts(args, kwargs)
    if topts is None:
        if not _PREFETCH_LOGGED:
            _PREFETCH_LOGGED.append(1)
            logger.warning("[MultiGPU LTX TP] transformer_options not found in the diffusion model call: block prefetch stays ON "
                           "(the unused block weights get paged in every step)")
    else:
        topts["prefetch_dynamic_vbars"] = False
    return executor(*args, **kwargs)


def _ram():
    try:
        mi = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        return "host RAM available %.1f GB" % (int(mi["MemAvailable"].split()[0]) / 2**20)
    except Exception:
        return "host RAM ?"


def _sampling(tp, executor, *args, **kwargs):
    logger.info("[MultiGPU LTX TP] sampling run starts | %s | %s", _gpu_free(), _ram())
    tp.first_block_logged = False
    patched = [k for k in getattr(executor.class_obj.model_patcher, "patches", {}) if ".transformer_blocks." in k]
    if patched:
        logger.warning("[MultiGPU LTX TP] %d LoRA/patch keys on transformer blocks are NOT applied under tensor parallel "
                       "(e.g. %s)", len(patched), patched[0])
    t0 = time.perf_counter()
    try:
        return executor(*args, **kwargs)
    finally:
        tp.state = None
        logger.info("[MultiGPU LTX TP] sampling run %.1fs | %s | %s", time.perf_counter() - t0, _gpu_free(), _ram())


class UNETLoaderLTXTensorParallel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "multigpu"

    def load(self, unet_name):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("LTX tensor parallel needs two CUDA devices")
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path)
        dm = model.model.diffusion_model
        if not hasattr(dm, "transformer_blocks") or not isinstance(dm.transformer_blocks[0], BasicAVTransformerBlock):
            raise ValueError("UNETLoaderLTXTensorParallel only supports LTX 2.x audio-video checkpoints")
        # the Comfy compiler records one device's allocations per forward; TP allocates on two
        comfy.cli_args.args.disable_comfy_compiler = True
        tp = _INSTANCES.get(path)
        if tp is None:
            tp = _INSTANCES[path] = LTXTensorParallel(path, dm)
        else:
            tp.blocks = dm.transformer_blocks  # shards stay; everything they need comes from the checkpoint
        for i in range(len(dm.transformer_blocks)):
            model.set_model_patch_replace(lambda args, extra, i=i: tp.block(i, args, extra["original_block"]),
                                          "dit", "double_block", i)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "ltx_tp", _no_block_prefetch)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "ltx_tp",
                                   lambda executor, *a, **k: _sampling(tp, executor, *a, **k))
        return (model,)
