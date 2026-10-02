"""Tensor-parallel LTX 2.x audio-video DiT block across two GPUs without P2P (Kaggle 2x T4).

Megatron-style split of every BasicAVTransformerBlock linear: each attention by heads (to_q / to_k / to_v /
to_gate_logits by output rows, to_out by input columns, bias added once), each feed-forward by hidden width
(net.0.proj by output rows, net.2 by input columns). Both GPUs keep a replica of the video and audio residual
streams and compute every AdaLN / gate / norm on it; row-parallel partial sums and the all-head q/k RMSNorm
sums of squares are exchanged through pinned host buffers. Each rank adds the two partials in the same order,
so the replicas stay bit-identical.

Shards come from the int8 convrot checkpoint (int8 splits land on multiples of the 256-wide rotation group;
floating layers are split as fp16). block_forward mirrors BasicAVTransformerBlock.forward of ComfyUI 0.37.0
(inference path, no training branch). Not wired into sampling yet: used by the standalone block test.
"""
import json
import logging
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.ldm.common_dit
import comfy.quant_ops
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
TIMING = os.environ.get("LTX_TP_TIMING", "0") == "1"  # sync around every exchange and report where the time goes


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


def load_block_shards(path, i):
    """[rank-0 shard, rank-1 shard] of transformer block i, on DEVICES[0] / DEVICES[1]."""
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        meta = json.loads((f.metadata() or {}).get("_quantization_metadata", "{}")).get("layers", {})
        pre = _block_prefix(f) + "%d." % i
        shards = []
        for rank, dev in enumerate(DEVICES):
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
            shards.append(_to(sh, dev))
    return shards


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
