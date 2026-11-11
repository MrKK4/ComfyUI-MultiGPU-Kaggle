"""Tensor-parallel MiniMax H3 DiT across two GPUs without P2P (Kaggle 2x T4).

Every DiT block is split Megatron-style: qkv_proj / fc1 by output rows (half the heads and
half the MLP width per GPU), out_proj / fc2 by input columns. Each GPU keeps a replica of the
fp32 residual stream; the two row-parallel partial sums are exchanged through pinned host
buffers in row chunks on side copy streams, so the PCIe round trip of one chunk hides behind
the compute of the next (MMH3_TP_CHUNKS, default 4; per-row int8 activation quant keeps the
chunked math identical). Shards come straight from
the int8 convrot checkpoint (splits land on multiples of the 256 rotation group) and stay
resident on both GPUs; the model's own block weights are never paged in.

Numerics follow h3_mixed_precision: fp32 residual, fp16 block compute, fc2 / out_proj on a
1/S-scaled input rescaled in fp32.
"""
import json
import logging
import os
import time
import threading
from contextlib import contextmanager, nullcontext
from collections import defaultdict

import torch
from safetensors import safe_open

from .tp_source import open_weights
from .h3_bundle_source import H3BundleSource

import comfy.cli_args
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.quant_ops
import comfy.rmsnorm
import comfy.sd
import comfy.weight_adapter
import comfy_kitchen.backends.cuda as ck_cuda
import folder_paths
import comfy.ldm.minimax.model as h3
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

logger = logging.getLogger("MultiGPU")

S = 64.0
HEAD = 128
DEVICES = (torch.device("cuda:0"), torch.device("cuda:1"))
_SHARD_CACHE = {}
_BUNDLE_LOCKS = {}
_BUNDLE_LEASES = defaultdict(int)


def _bundle_lock(source):
    return _BUNDLE_LOCKS.setdefault(source, threading.RLock())


def _complete(shards, n_blocks):
    return (shards is not None and len(shards) == 2 and
            all(rank is not None and len(rank) == n_blocks and all(block is not None for block in rank)
                for rank in shards))


@contextmanager
def bundle_sampler_lease(tp):
    """Serialize sampling against piece eviction for source-free H3 bundles."""
    if not isinstance(tp.path, H3BundleSource):
        yield
        return
    lock = _bundle_lock(tp.path)
    with lock:
        _BUNDLE_LEASES[tp.path] += 1
        try:
            yield
        finally:
            _BUNDLE_LEASES[tp.path] -= 1


def evict_bundle(model, free_gib_by_rank):
    """Free the fewest whole pieces needed to reach each device's free-VRAM target."""
    dm = getattr(getattr(model, 'model', None), 'diffusion_model', None)
    tp = getattr(dm, '_mgpu_tp', None)
    if tp is None or not isinstance(tp.path, H3BundleSource):
        raise ValueError('TPEvictH3 requires a model from UNETLoaderH3TPBundle')
    return evict_bundle_tp(tp, free_gib_by_rank)


def evict_bundle_tp(tp, free_gib_by_rank):
    """Evict from a known bundle instance, for both explicit and automatic barriers."""
    if not isinstance(tp.path, H3BundleSource):
        raise ValueError('Expected a source-free H3 bundle instance')
    source = tp.path
    targets = tuple(max(0, int(float(value)*(1 << 30))) for value in free_gib_by_rank)
    if len(targets) != 2:
        raise ValueError('Expected one free-VRAM target per GPU')
    with _bundle_lock(source):
        if _BUNDLE_LEASES[source]:
            raise RuntimeError('Cannot evict H3 pieces while sampling')
        shards = _SHARD_CACHE.get(source)
        if shards is None:
            return (0, 0)
        manifest = getattr(tp, 'bundle_manifest', None)
        if manifest is None:
            manifest, _, _ = source.validate()
        tp.release()  # exchange and LoRA scratch can hold GPU memory
        released = [0, 0]
        for rank, device in enumerate(DEVICES):
            if not targets[rank] or shards[rank] is None:
                continue
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(device)
                pieces = sorted((p for p in manifest['pieces'] if p['rank'] == rank),
                                key=lambda p:p['block_start'], reverse=True)
                for piece in pieces:
                    if free >= targets[rank]:
                        break
                    a, b = piece['block_start'], piece['block_end']
                    if all(shards[rank][i] is None for i in range(a,b)):
                        continue
                    for i in range(a,b):
                        shards[rank][i] = None
                    released[rank] += 1
                    free += piece['resident_bytes']
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                actual, _ = torch.cuda.mem_get_info(device)
                if actual < targets[rank]:
                    raise RuntimeError(f'[TP-STATE] GPU{rank} free {actual/2**30:.2f} GiB, need {targets[rank]/2**30:.2f} GiB; lower the requested memory or release other models')
                logger.info('[TP-STATE] GPU%d free=%.2fGiB evicted=%d pieces',
                            rank,actual/2**30,released[rank])
        return tuple(released)
CHUNKS = max(1, int(os.environ.get("MMH3_TP_CHUNKS", "4")))
# Timing aid: with XCHG_OFF (env MULTIGPU_TP_XCHG_OFF=1, or set the module attribute at run time) the partial sums are not exchanged. The output is wrong; the step time
# shows what the exchange costs.
XCHG_OFF = os.environ.get("MULTIGPU_TP_XCHG_OFF", "0") == "1"
# next block's norm1 + qkv for finished chunks, queued behind the last mlp exchange of this block
QKV_PREFETCH = os.environ.get("MMH3_TP_QKV_PREFETCH", "1") != "0"
# Bound norm, quantization and LoRA output temporaries independently of video length.
QKV_ROWS = max(1, int(os.environ.get("MMH3_TP_QKV_ROWS", "4096")))
LORA_MODE = os.environ.get("MMH3_TP_LORA_MODE", "auto")
if LORA_MODE not in ("auto", "resident", "stream"):
    raise ValueError("MMH3_TP_LORA_MODE must be auto, resident or stream")


def _record(dev, stream=None):
    ev = torch.cuda.Event()
    ev.record(stream if stream is not None else torch.cuda.current_stream(dev))
    return ev


class _ChunkedExchange:
    """Row-parallel partial sums swapped chunk by chunk through pinned host buffers on side copy
    streams. Pinned buffers persist per (phase, shape), events guard their reuse by the next block;
    device receive buffers are allocated per block on the compute streams. No record_stream and no
    copy-stream allocations: every partial is kept alive until the compute stream has waited on the
    copy that read it (side streams that allocated OOMed next to dynamic VRAM)."""

    def __init__(self):
        self.copy = None
        self.bufs = {}

    def _buf(self, phase, shape, dtype=torch.float16):
        if self.copy is None:
            # MMH3_TP_SIDE_STREAMS=0 keeps the copies on the compute streams (no overlap, ~0.4 s/step slower)
            side = os.environ.get("MMH3_TP_SIDE_STREAMS", "1") != "0"
            self.copy = [torch.cuda.Stream(d) if side else torch.cuda.current_stream(d) for d in DEVICES]
        key = (phase, shape)
        if key not in self.bufs:
            self.bufs[key] = {"host": [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in DEVICES],
                              "read": [None, None]}  # rank r's copy stream finished reading host[1-r]
        return self.bufs[key]

    def send(self, phase, shape, rows, parts):
        """D2H of parts[r] (rank r's partial for `rows`) into host[r]; returns the copy-done events."""
        if XCHG_OFF:
            return parts
        a, b = rows
        buf, sent = self._buf(phase, shape, parts[0].dtype), []
        for r, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                cs = self.copy[r]
                cs.wait_event(_record(dev))
                if a == 0 and buf["read"][1 - r] is not None:
                    cs.wait_event(buf["read"][1 - r])
                with torch.cuda.stream(cs):
                    buf["host"][r][a:b].copy_(parts[r], non_blocking=True)
                sent.append(_record(dev, cs))
        return sent

    def begin(self, phase, shape, dtype=torch.float16):
        """Per-block receive buffers, allocated on the compute streams before any copy is queued."""
        buf = self._buf(phase, shape, dtype)
        buf["recv"] = [torch.empty(shape, dtype=dtype, device=d) for d in DEVICES]
        buf["alloc"] = [_record(d) for d in DEVICES]

    def recv(self, phase, shape, rows, sent):
        """H2D of the other rank's rows; returns per-rank views, ready on each compute stream."""
        if XCHG_OFF:
            return list(sent)
        a, b = rows
        buf, out = self._buf(phase, shape), []
        for r, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                cs = self.copy[r]
                cs.wait_event(sent[1 - r])
                if a == 0:
                    cs.wait_event(buf["alloc"][r])
                dst = buf["recv"][r][a:b]
                with torch.cuda.stream(cs):
                    dst.copy_(buf["host"][1 - r][a:b], non_blocking=True)
                ev = _record(dev, cs)
                if b == shape[0]:
                    buf["read"][r] = ev
                    buf["recv"][r] = None  # freed on the compute stream once the block's users are queued
                torch.cuda.current_stream(dev).wait_event(ev)
                out.append(dst)
        return out


def _chunk_segs(segs, a, b):
    """mod_segments restricted to rows [a, b), rebased to a."""
    out = []
    for sa, sb, row in segs:
        lo, hi = max(sa, a), min(sb, b)
        if lo < hi:
            out.append((lo - a, hi - a, row[lo - sa:hi - sa] if torch.is_tensor(row) and row.dim() else row))
    return out


# torch.compile fuses the per-block small ops (rmsnorm + modulation + fp16 cast, and the gated residual add) into one
# Triton kernel each; eager runs them as 4-6 memory-bound kernels per segment. Probe on T4: 15.5 -> 2.1 ms per call.
# MMH3_TP_COMPILE=0 keeps the eager ops; any compile failure falls back to eager for the rest of the session.
TP_COMPILE = os.environ.get("MMH3_TP_COMPILE", "1") != "0"


def _norm_mod_eager(x, w, eps, scale, shift):
    h = torch.nn.functional.rms_norm(x, w.shape, weight=w, eps=eps)
    return (h * (1.0 + scale) + shift).to(torch.float16)


def _gate_delta_eager(own, got, gate, s):
    return (own + got).float() * s * gate


_NORM_MOD = torch.compile(_norm_mod_eager, dynamic=True)
_GATE_DELTA = torch.compile(_gate_delta_eager, dynamic=True)


def _fused(compiled, eager, *args):
    global TP_COMPILE
    if TP_COMPILE:
        try:
            return compiled(*args)
        except torch.OutOfMemoryError:
            raise  # a real OOM, not a compile problem: let ComfyUI's OOM handling see it, keep compile on
        except Exception as e:  # ponytail: one failure disables compile for the session; eager is the old exact path
            TP_COMPILE = False
            logger.warning("[MultiGPU TP] torch.compile of the small ops failed, using eager: %s: %s", type(e).__name__, str(e)[:300])
    return eager(*args)


def _norm_mod(x, w, eps, shift, scale, segs):
    """rmsnorm(x) * (1 + scale) + shift per modulation segment, as fp16 (the int8 linear's input)."""
    if not TP_COMPILE:
        return h3._mod_scale_shift(comfy.rmsnorm.rms_norm(x, w, eps), shift, scale, segs).to(torch.float16)
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    for a, b, row in segs:
        out[a:b] = _fused(_NORM_MOD, _norm_mod_eager, x[a:b], w.to(x.dtype), eps, scale[row].to(x.dtype), shift[row].to(x.dtype))
    return out


def _gate_add(x, gate, own, got, segs):
    """x += (own + got) * S * gate per segment: the gated residual of a row-parallel partial sum pair, in place."""
    if not TP_COMPILE:
        return h3._mod_gate(x, gate, (own + got).float() * S, segs)
    for a, b, row in segs:
        x[a:b].add_(_fused(_GATE_DELTA, _gate_delta_eager, own[a:b], got[a:b], gate[row].to(x.dtype), S))
    return x


MEMDUMP = os.environ.get("MMH3_TP_MEMDUMP", "0") != "0"


def _memdump(shape):
    """MMH3_TP_MEMDUMP=1: at block 0 of every forward, what holds VRAM on each card (the ~3 GB besides the shards)."""
    import gc
    lines = [f"[MultiGPU TP] MEMDUMP tokens/img {tuple(shape)}"]
    for d in DEVICES:
        lines.append(f"  {d}: allocated {torch.cuda.memory_allocated(d) / 2**30:.2f} GiB, reserved {torch.cuda.memory_reserved(d) / 2**30:.2f} GiB")
    seen, rows = set(), []
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda and o.untyped_storage().data_ptr() not in seen:
                seen.add(o.untyped_storage().data_ptr())
                rows.append((o.untyped_storage().nbytes(), str(o.device), tuple(o.shape), str(o.dtype), o.untyped_storage().data_ptr()))
        except Exception:
            pass
    shard_ptrs = set()
    for tp in _INSTANCES.values():
        for rank in tp.shards or []:
            if rank is None:
                continue
            for blk in rank:
                for v in blk.values():
                    for t in (v if isinstance(v, tuple) else (v,)):
                        if torch.is_tensor(t):
                            shard_ptrs.add(t.untyped_storage().data_ptr())
    shard, other = {}, {}
    for nb, dev, shp, dt, ptr in rows:
        bucket = shard if ptr in shard_ptrs else other
        bucket[dev] = bucket.get(dev, 0) + nb
    lines.append("  TP shards: " + ", ".join(f"{k} {v / 2**30:.2f} GiB" for k, v in sorted(shard.items())))
    lines.append("  everything else python can see: " + ", ".join(f"{k} {v / 2**30:.2f} GiB" for k, v in sorted(other.items())))
    for nb, dev, shp, dt, ptr in sorted((r for r in rows if r[4] not in shard_ptrs), reverse=True)[:25]:
        lines.append(f"    {nb / 2**20:9.1f} MiB  {dev}  {dt}  {shp}")
    logger.warning(chr(10).join(lines))


def _per_rank(fn):
    out = []
    for r, dev in enumerate(DEVICES):
        with torch.cuda.device(dev):
            out.append(fn(r))
    return out


def _rows(t, chunk, part, rank):
    """Rows [rank*part, (rank+1)*part) of every consecutive `chunk`-row group, concatenated."""
    return torch.cat([t[j * chunk + rank * part: j * chunk + (rank + 1) * part] for j in range(t.shape[0] // chunk)])


def _load_shards(path, n_blocks):
    cached = _SHARD_CACHE.get(path)
    if _complete(cached, n_blocks):
        logger.info('[MultiGPU TP] reusing both resident shard ranks')
        return cached
    missing = [rank for rank in range(len(DEVICES)) if cached is None or cached[rank] is None]
    if isinstance(path, H3BundleSource):
        from .h3_piece_loader import restore_bundle_ranks
        shards = restore_bundle_ranks(path, n_blocks, cached, DEVICES,
            comfy.model_management.free_memory, comfy.model_management.get_free_memory)
        _SHARD_CACHE[path] = shards
        return shards
    if isinstance(path, str) and os.environ.get('MMH3_TP_PIECES', '0') == '1':
        from .h3_piece_loader import PieceFileError, restore_ranks
        failure = None
        try:
            shards = restore_ranks(path, n_blocks, os.environ.get('MMH3_TP_PIECES_DIR', ''),
                os.environ.get('MMH3_TP_PIECES_MANIFEST_SHA256', ''), cached, DEVICES,
                comfy.model_management.free_memory, comfy.model_management.get_free_memory)
        except PieceFileError as error:
            failure = str(error)
        if failure is None:
            _SHARD_CACHE[path] = shards
            return shards
        logger.warning('[TP-PIECES] rejected: %s; rebuilding from original source', failure)
    started = time.perf_counter()
    logger.info('[MultiGPU TP] loading shard ranks %s; retained ranks %s', missing,
                [rank for rank in range(len(DEVICES)) if rank not in missing])
    with open_weights(path) as f:   # checkpoint path, or tp_source.ModelWeights of a loaded model
        layers = json.loads(f.metadata()["_quantization_metadata"])["layers"]
        need = 0
        for k in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            conf = layers.get(f"blocks.0.{k}", {})
            if conf.get("format") != "int8_tensorwise" or not conf.get("convrot"):
                raise ValueError(f"tensor parallel needs the int8 convrot checkpoint; blocks.0.{k} is {conf}")
            need += f.get_slice(f"blocks.0.{k}.weight").get_shape()[0] * f.get_slice(f"blocks.0.{k}.weight").get_shape()[1]
        need = int(need * n_blocks / 2 * 1.08)
        for rank in missing:
            comfy.model_management.free_memory(need + (2 << 30), DEVICES[rank])
        # Build into a separate list so a failed restore cannot expose a
        # half-built rank through the persistent cache.
        shards = list(cached) if cached is not None else [None, None]
        for rank in missing:
            shards[rank] = []
        for i in range(n_blocks):
            p = f"blocks.{i}."
            g = lambda k: f.get_tensor(p + k)
            qkv, qkv_s, out, out_s = g("attn.qkv_proj.weight"), g("attn.qkv_proj.weight_scale"), g("attn.out_proj.weight"), g("attn.out_proj.weight_scale")
            fc1, fc1_s, fc2, fc2_s = g("mlp.fc1.weight"), g("mlp.fc1.weight_scale"), g("mlp.fc2.weight"), g("mlp.fc2.weight_scale")
            small = {n: g(k).float() for n, k in (("norm1", "norm1.weight"), ("norm2", "norm2.weight"))}
            qn, kn = g("attn.q_norm.weight"), g("attn.k_norm.weight")
            inner, ffn = qkv.shape[0] // 3, fc1.shape[0] // 2
            for rank in missing:
                dev = DEVICES[rank]
                with torch.cuda.device(dev):
                    half_i, half_f = inner // 2, ffn // 2
                    shards[rank].append({
                        "heads": half_i // HEAD, "inner": half_i,
                        "qkv": (_rows(qkv, inner, half_i, rank).to(dev), _rows(qkv_s, inner, half_i, rank).to(dev)),
                        "out": (out[:, rank * half_i:(rank + 1) * half_i].contiguous().to(dev), out_s.to(dev)),
                        "fc1": (_rows(fc1, ffn, half_f, rank).to(dev), _rows(fc1_s, ffn, half_f, rank).to(dev)),
                        "fc2": (fc2[:, rank * half_f:(rank + 1) * half_f].contiguous().to(dev), fc2_s.to(dev)),
                        "norm1": small["norm1"].to(dev), "norm2": small["norm2"].to(dev),
                        "qn": qn.to(dev), "kn": kn.to(dev),
                    })
            if i % 10 == 0:
                logger.info(f"[MultiGPU TP] sharded block {i}/{n_blocks}")
    _SHARD_CACHE[path] = shards
    logger.info('[MultiGPU TP] %d blocks ready over %s; loaded ranks %s in %.2f seconds',
                n_blocks, ', '.join(map(str, DEVICES)), missing, time.perf_counter()-started)
    return shards


def _lin(x, w, act=None):
    return ck_cuda.int8_linear(x, w[0], w[1], convrot=True, convrot_groupsize=256, input_act=act)


# LoRA: the shards stay the file's int8 weights; each patched linear gets a low-rank side path x @ down.T @ up.T,
# sharded like its weight (column-parallel: rows of up; row-parallel: columns of down), so the partial sums still add up.
LORA_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def _lora_factors(entries):
    """(down with strength * alpha / rank folded in, up) of the LoRA patches on one weight, stacked along the rank."""
    downs, ups = [], []
    for strength, adapter, strength_model, offset, function in entries:
        w = adapter.weights if isinstance(adapter, comfy.weight_adapter.LoRAAdapter) else None
        if w is None or strength_model != 1.0 or offset is not None or function is not None or w[3] is not None or w[4] is not None:
            raise ValueError("H3 tensor parallel supports plain LoRA only (no LoCon mid, DoRA, LoHa/LoKr, offsets or model strength != 1)")
        down, up = (v.flatten(1).to(device="cpu", dtype=torch.float32) for v in (w[1], w[0]))
        alpha = float(w[2]) / down.shape[0] if w[2] is not None else 1.0
        downs.append(down * (strength * alpha))
        ups.append(up)
    return torch.cat(downs), torch.cat(ups, dim=1)


def _shard_lora(name, down, up, rank):
    if name == "attn.qkv_proj":
        inner = up.shape[0] // 3
        return down, _rows(up, inner, inner // 2, rank)
    if name == "mlp.fc1":
        ffn = up.shape[0] // 2
        return down, _rows(up, ffn, ffn // 2, rank)
    half = down.shape[1] // 2  # out_proj / fc2 are row-parallel: each rank holds half the input columns
    return down[:, rank * half:(rank + 1) * half], up


def _lora(x, pair):
    return torch.nn.functional.linear(torch.nn.functional.linear(x, pair[0]), pair[1])


# One instance per checkpoint: a ComfyUI cache reset (the studio frees host RAM after each face swap) re-runs the
# loader, and without this every job would re-shard ~20 GB (~80 s) while the old shards wait for garbage collection.
_INSTANCES = {}


def release_gpu(drop_shards=False):
    """Free the exchange buffers (device receive + pinned host) and, with drop_shards, the shards themselves
    (re-read on the next sampling run) so another model (SAM 3.1) can use the VRAM."""
    for tp in _INSTANCES.values():
        tp.release(drop_shards)


def release_encoder_rank(device):
    """Make room for Molmo on its GPU while retaining the other rank.

    Used only outside sampling. Do not park ~9GB of weights in host RAM:
    the encoder and base checkpoint already compete for Kaggle's host RAM.
    The missing rank is reconstructed from the checkpoint at next sampling.
    """
    rank = DEVICES.index(torch.device(device))
    released = 0
    for tp in _INSTANCES.values():
        guard = _bundle_lock(tp.path) if isinstance(tp.path, H3BundleSource) else nullcontext()
        with guard:
            if isinstance(tp.path, H3BundleSource) and _BUNDLE_LEASES[tp.path]:
                raise RuntimeError('Cannot release H3 encoder rank while sampling')
            if tp.shards is None or tp.shards[rank] is None:
                continue
            tp.release()  # drop exchange/LoRA scratch without dropping weights
            tp.shards[rank] = None
            cached = _SHARD_CACHE.get(tp.path)
            if cached is not None:
                cached[rank] = None
            released += 1
    with torch.cuda.device(DEVICES[rank]):
        torch.cuda.synchronize(DEVICES[rank])
        torch.cuda.empty_cache()
    logger.info('[MultiGPU TP] released rank %d for encoder; retained other GPU rank (%d model instances)',rank,released)
    return released


class H3TensorParallel:
    def release(self, drop_shards=False):
        if drop_shards and isinstance(self.path, H3BundleSource) and _BUNDLE_LEASES[self.path]:
            raise RuntimeError('Cannot drop H3 bundle shards while sampling')
        for d in DEVICES:
            torch.cuda.synchronize(d)
        self.xchg.bufs.clear()
        # GPU LoRA copies belong to sampling, not the decoder or the next workflow.
        self.lora = None
        self.lora_mode = None
        if drop_shards:
            _SHARD_CACHE.pop(self.path, None)
            self.shards = None
            self.lora_entries = {}
            self.lora_sig = ()
            logger.warning("[MultiGPU TP] shards released to make room; the next sampling run re-shards them")
        for d in DEVICES:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()

    def __init__(self, path, diffusion_model):
        self.path = path
        self.blocks = diffusion_model.blocks
        self.shards = None
        self.xchg = _ChunkedExchange()
        blk = self.blocks[0]
        self.eps1, self.eps2 = blk.norm1.eps, blk.norm2.eps
        self.qk_eps = blk.attn.q_norm.eps
        requested_reference_check = os.environ.get("MMH3_TP_CHECK", "0") == "1"
        self.checked = not requested_reference_check or isinstance(path, H3BundleSource)
        if requested_reference_check and isinstance(path, H3BundleSource):
            logger.info('[TP-BUNDLE] skipping original-block reference check; bundle owns only split projections')
        self.prefetch_oom_n = float("inf")
        self.lora, self.lora_sig = None, ()
        self.lora_entries = {}
        self.lora_mode = None
        self.lora_bytes = 0

    def set_lora(self, patches):
        """Validate and retain adapters on the host; choose GPU residency once the token shape is known."""
        entries, ignored = {}, []
        for key, value in patches.items():
            parts = key.split(".")
            if len(parts) < 5 or parts[:2] != ["diffusion_model", "blocks"]:
                continue  # non-block layers run on ComfyUI's own patched modules
            name = ".".join(parts[3:-1])
            if name in LORA_TARGETS and parts[-1] == "weight":
                entries[(int(parts[2]), name)] = value
            elif not name.startswith("adaln_proj"):  # adaln_proj also runs on the patched original module
                ignored.append(key)
        sig = tuple(sorted((k, tuple((id(p[1]), p[0], p[2], id(p[3]), id(p[4])) for p in v)) for k, v in entries.items()))
        if sig == self.lora_sig:
            return
        if ignored:
            raise ValueError("H3 tensor parallel cannot apply these block patches correctly: " + str(ignored[:5]))
        self.lora = None
        # Construct one block at a time. Avoid duplicating every CPU adapter or allocating every GPU side path
        # before the base shards, activations and other workflow models have been accounted for.
        by_block, nbytes = {}, 0
        for (i, name), value in entries.items():
            by_block.setdefault(i, {})[name] = value
            for strength, adapter, strength_model, offset, function in value:
                w = adapter.weights if isinstance(adapter, comfy.weight_adapter.LoRAAdapter) else None
                if (w is None or strength_model != 1.0 or offset is not None or function is not None
                        or w[3] is not None or w[4] is not None):
                    raise ValueError("H3 tensor parallel supports plain LoRA only; unsupported patch must not be silently skipped")
                up, down = w[0], w[1]
                nbytes += (down.numel() + up.numel() // 2 if name in ("attn.qkv_proj", "mlp.fc1")
                           else down.numel() // 2 + up.numel()) * 2
        self.lora_entries, self.lora_bytes, self.lora_mode = by_block, nbytes, None
        logger.info("[MultiGPU TP] LoRA: %d block linears, %.1f MiB per GPU if resident", len(entries), nbytes / 2**20)
        self.lora_sig = sig

    def _configure_lora(self, x):
        if self.lora_mode is not None:
            return
        # Budget for both exchanges, residual/attention/QKV, chunked MLP work and allocator slack.
        # This deliberately favours streaming high-rank/stacked adapters on 15-GB cards at 1MP.
        reserve = (2 << 30) + x.shape[0] * x.shape[1] * 2 * 8
        free = min(comfy.model_management.get_free_memory(d) for d in DEVICES)
        self.lora_mode = ("resident" if free >= reserve + self.lora_bytes else "stream") if LORA_MODE == "auto" else LORA_MODE
        self.lora = {}
        logger.info("[MultiGPU TP] memory policy: %d tokens, QKV rows %d, LoRA %s (%.1f MiB/GPU; free %.2f GiB; activation budget %.2f GiB)",
                    x.shape[0], QKV_ROWS, self.lora_mode, self.lora_bytes / 2**20, free / 2**30, reserve / 2**30)

    def _block_lora(self, i):
        entries = self.lora_entries.get(i)
        if not entries:
            return ({}, {})
        if i not in self.lora:
            if self.lora_mode == "stream":
                self.lora.clear()
            pair = ({}, {})
            for name, value in entries.items():
                down, up = _lora_factors(value)
                for rank, dev in enumerate(DEVICES):
                    d, u = _shard_lora(name, down, up, rank)
                    pair[rank][name] = (d.to(dev, torch.float16).contiguous(), u.to(dev, torch.float16).contiguous())
            self.lora[i] = pair
        return self.lora[i]

    def _qkv(self, x, sh, shift, scale, segs, lora):
        out = torch.empty((x.shape[0], 3 * sh["inner"]), dtype=torch.float16, device=x.device)
        for a in range(0, x.shape[0], QKV_ROWS):
            b = min(a + QKV_ROWS, x.shape[0])
            h = _norm_mod(x[a:b], sh["norm1"], self.eps1, shift, scale, _chunk_segs(segs, a, b))
            part = _lin(h, sh["qkv"])
            if "attn.qkv_proj" in lora:
                part += _lora(h, lora["attn.qkv_proj"])
            out[a:b].copy_(part)
        return out

    def _attn(self, qkv, sh, rope, topts):
        s, heads = qkv.shape[0], sh["heads"]
        q, k, v = qkv.split(sh["inner"], dim=-1)
        v = v.view(s, heads, HEAD)
        q = q.view(1, s, heads, HEAD)
        k = k.view(1, s, heads, HEAD)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope, sh["qn"], sh["kn"], epsilon=self.qk_eps, rot_dim=rope.shape[-3] * 2)
        q = AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        return optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, transformer_options=topts).squeeze(0)

    def _mlp(self, x, sh, shift, scale, segs, lora):
        h = _norm_mod(x, sh["norm2"], self.eps2, shift, scale, segs)
        a = _lin(h, sh["fc1"])
        if "mlp.fc1" in lora:
            a += _lora(h, lora["mlp.fc1"])
        a[..., a.shape[-1] // 2:].mul_(1.0 / S)
        out = _lin(a, sh["fc2"], act="swiglu")
        if "mlp.fc2" in lora:
            gate, up = a.chunk(2, dim=-1)  # the swiglu the int8 kernel fuses, in the same 1/S-scaled space
            out += _lora(torch.nn.functional.silu(gate).mul_(up), lora["mlp.fc2"])
        return out

    def _mods(self, i, t_emb):
        mods0 = self.blocks[i].adaln_proj(t_emb)
        with torch.cuda.device(DEVICES[1]):
            mods1 = [m.to(DEVICES[1], non_blocking=True) for m in mods0]
        return mods0, mods1

    def _step_state(self, x0, args):
        """Per-forward replicas on cuda:1, rebuilt when a new forward starts (block 0)."""
        topts = args["transformer_options"]
        st = topts.get("_mmh3_tp")
        if st is not None and st["x0"] is x0:
            return st
        d1 = DEVICES[1]
        rope = args["rope_freqs"].to(torch.float16)
        segs = args["mod_segments"]
        with torch.cuda.device(d1):
            st = {"x1": x0.float().to(d1), "rope0": rope, "rope1": rope.to(d1),
                  "segs1": [(a, b, r.to(d1) if torch.is_tensor(r) else r) for a, b, r in segs]}
        topts["_mmh3_tp"] = st
        return st

    def block(self, i, args, original):
        d0, d1 = DEVICES
        if not _complete(self.shards, len(self.blocks)):
            if isinstance(self.path, H3BundleSource) or (os.environ.get('MMH3_TP_PIECES', '0') == '1' and isinstance(self.path, str)):
                raise RuntimeError('H3 shards missing during forward; expected pre-sampling preload')
            # permanent allocations: keep them out of the compiler's per-forward allocation graph
            with comfy.model_prefetch.pause_malloc_graph(sync=True):
                self.shards = _load_shards(self.path, len(self.blocks))
        if i == 0 and MEMDUMP:
            _memdump(args["img"].shape)
        blk = self.blocks[i]
        x0 = args["img"].float()
        st = self._step_state(args["img"], args)
        x1, segs0, segs1, topts = st["x1"], args["mod_segments"], st["segs1"], args["transformer_options"]
        sh0, sh1 = self.shards[0][i], self.shards[1][i]
        self._configure_lora(x0)
        lo = self._block_lora(i)
        pre = st.pop("pre", None)
        if pre is not None and pre[0] == i:
            _, mods0, mods1, qkv = pre
        else:
            mods0, mods1 = self._mods(i, args["t_emb"])
            qkv = None
        ref = None
        if not self.checked:
            ref = original({**args, "img": args["img"].clone()})["img"].float()

        xs, shs, mods, segs = (x0, x1), (sh0, sh1), (mods0, mods1), (segs0, segs1)
        if qkv is None:
            qkv = _per_rank(lambda r: self._qkv(xs[r], shs[r], mods[r][0], mods[r][1], segs[r], lo[r]))
        os_ = _per_rank(lambda r: self._attn(qkv[r], shs[r], st["rope%d" % r], topts))
        del qkv
        # per-token tail pipelined in row chunks: out_proj -> swap -> gate -> mlp -> swap -> gate
        n = x0.shape[0]
        shape, size = (n, x0.shape[1]), -(-n // CHUNKS)
        chunks = [(a, min(a + size, n)) for a in range(0, n, size)]
        csegs = [[_chunk_segs(segs[r], a, b) for r in range(2)] for a, b in chunks]
        self.xchg.begin("attn", shape)  # after attention: keeps the qkv peak unchanged
        own_a, sent_a = [], []
        for a, b in chunks:
            def out_proj(r):
                xs_ = os_[r][a:b] * (1.0 / S)
                out = _lin(xs_, shs[r]["out"])
                if "attn.out_proj" in lo[r]:
                    out += _lora(xs_, lo[r]["attn.out_proj"])  # partial sum, same 1/S-scaled space as the shard's
                return out
            own_a.append(_per_rank(out_proj))
            sent_a.append(self.xchg.send("attn", shape, (a, b), own_a[-1]))
        del os_
        self.xchg.begin("mlp", shape)
        own_m, sent_m = [], []
        for j, (a, b) in enumerate(chunks):
            got = self.xchg.recv("attn", shape, (a, b), sent_a[j])

            def attn_then_mlp(r):
                xc = xs[r][a:b]
                _gate_add(xc, mods[r][2], own_a[j][r], got[r], csegs[j][r])
                own_a[j][r] = None
                return self._mlp(xc, shs[r], mods[r][3], mods[r][4], csegs[j][r], lo[r])
            own_m.append(_per_rank(attn_then_mlp))
            sent_m.append(self.xchg.send("mlp", shape, (a, b), own_m[j]))
        # Prefetch is optional: leave room for two QKV buffers, chunk work and allocator slack before attempting it.
        prefetch_need = 2 * n * 3 * sh0["inner"] * 2 + (512 << 20)
        can_prefetch = (not self.lora_entries or self.lora_mode == "resident") and all(
            comfy.model_management.get_free_memory(d) >= prefetch_need for d in DEVICES)
        nxt = i + 1 if QKV_PREFETCH and can_prefetch and i + 1 < len(self.blocks) and ref is None and n < self.prefetch_oom_n else None
        if nxt is not None:
            nmods = self._mods(nxt, args["t_emb"])
            nsh = (self.shards[0][nxt], self.shards[1][nxt])
            nlo = self._block_lora(nxt)
            try:
                nqkv = _per_rank(lambda r: torch.empty((n, 3 * nsh[r]["inner"]), dtype=torch.float16, device=DEVICES[r]))
            except torch.OutOfMemoryError:
                # long sequences (~100k tokens): the extra qkv buffer does not fit next to this block's partials
                self.prefetch_oom_n = n
                nqkv = None
                logger.info(f"[MultiGPU TP] qkv prefetch off for {n}+ tokens (out of memory)")
                nxt = None
        for j, (a, b) in enumerate(chunks):
            got = self.xchg.recv("mlp", shape, (a, b), sent_m[j])

            def gate_then_next_qkv(r):
                xc = xs[r][a:b]
                _gate_add(xc, mods[r][5], own_m[j][r], got[r], csegs[j][r])
                if nxt is not None:
                    nqkv[r][a:b] = self._qkv(xc, nsh[r], nmods[r][0], nmods[r][1], csegs[j][r], nlo[r])
            _per_rank(gate_then_next_qkv)
            own_m[j] = None
        if nxt is not None:
            st["pre"] = (nxt, nmods[0], nmods[1], nqkv)

        if ref is not None:
            self.checked = True
            rel = float((x0 - ref).norm() / ref.norm())
            drift = float((x0 - x1.to(d0)).abs().max())
            logger.info(f"[MultiGPU TP] block {i} check: rel err vs single-GPU {rel:.2e}, replica drift {drift:.2e}")
            logger.info(f"[TP-CHECK] block{i} rel_err={rel:.3e} drift={drift:.3e} finite={bool(torch.isfinite(x0).all())}")
        st["x0"], st["x1"] = x0, x1
        return {"img": x0}


PROFILE_FLAG = "tp_profile.request"  # in ComfyUI's working dir: the next sampling step is profiled into tp_profile.txt


def _profiled(executor, *args, **kwargs):
    os.remove(PROFILE_FLAG)
    for d in DEVICES:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = executor(*args, **kwargs)
        for d in DEVICES:
            torch.cuda.synchronize(d)
    wall = time.perf_counter() - t0
    busy = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            busy[e.device_index] += e.self_device_time_total / 1e6
    report = ["one TP forward (profiler on): wall %.2fs | kernel busy %s" % (wall, "  ".join(
              "cuda:%d %.2fs" % (d, t) for d, t in sorted(busy.items()))),
              prof.key_averages().table(sort_by="self_device_time_total", row_limit=35)]
    with open("tp_profile.txt", "w") as f:
        f.write("\n".join(report))
    logger.info("[MultiGPU TP] profile written to %s: %s", os.path.abspath("tp_profile.txt"), report[0])
    return out


def _no_block_prefetch(executor, *args, **kwargs):
    # the model's own block weights are unused; prefetching would page all 21 GB in every step
    topts = kwargs.get("transformer_options", args[3] if len(args) > 3 else None)
    if topts is not None:
        topts["prefetch_dynamic_vbars"] = False
    try:
        if os.path.exists(PROFILE_FLAG):
            return _profiled(executor, *args, **kwargs)
        return executor(*args, **kwargs)
    finally:
        # Forward-local residual replicas and optional prefetch must not live in cached model_options.
        if topts is not None:
            topts.pop("_mmh3_tp", None)


def _prepare_sampling(tp, executor, *args, **kwargs):
    # models loaded earlier in the session (e.g. SAM 3.1 in the face swap workflow) stay on cuda:0, and ComfyUI
    # does not count the shards and TP activations it never allocated; unload them before the sampler loads
    with bundle_sampler_lease(tp):
        try:
            # Text encoders and VAE helpers can be left on either card by preceding UI jobs.
            for dev in DEVICES:
                comfy.model_management.free_memory(1e30, dev)
            if ((isinstance(getattr(tp, 'path', None), H3BundleSource) or
                 (os.environ.get('MMH3_TP_PIECES', '0') == '1' and isinstance(getattr(tp, 'path', None), str)))
                    and not _complete(tp.shards, len(tp.blocks))):
                with comfy.model_prefetch.pause_malloc_graph(sync=True):
                    tp.shards = _load_shards(tp.path, len(tp.blocks))
            if isinstance(getattr(tp, 'path', None), H3BundleSource) and not _complete(tp.shards, len(tp.blocks)):
                raise RuntimeError('H3 bundle pieces are incomplete; refusing to sample')
            # The guider's patcher carries LoRAs added after the loader, including stacked adapters.
            patcher = executor.class_obj.model_patcher
            if isinstance(getattr(tp, 'path', None), H3BundleSource):
                from .h3_bundle_lora import sampling_patches
                tp.set_lora(sampling_patches(patcher))
            else:
                tp.set_lora(patcher.patches)
            return executor(*args, **kwargs)
        finally:
            tp.release()  # ~0.6 GB of receive buffers per card and ~1.2 GB pinned host per token count


class UNETLoaderH3TensorParallel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "multigpu"

    def load(self, unet_name):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("H3 tensor parallel needs two CUDA devices")
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path)
        if not supports(model.model.diffusion_model):
            raise ValueError("UNETLoaderH3TensorParallel only supports MiniMax H3 checkpoints")
        attach(model, path, path)
        return (model,)


def supports(dm):
    return isinstance(dm, h3.MiniMaxH3Model)


def attach(model, source, key):
    """Runs the blocks of `model` (a ModelPatcher) on both GPUs; shards come from `source` (checkpoint path or
    tp_source.ModelWeights) and are cached under `key`."""
    dm = model.model.diffusion_model
    # the Comfy compiler records one device's allocations per forward; TP allocates on two.
    # ponytail: process-wide, Split in the same session runs without the compiler
    comfy.cli_args.args.disable_comfy_compiler = True
    for other_key, other in _INSTANCES.items():
        if other_key != key and (other.shards is not None or other.lora is not None):
            other.release(drop_shards=True)
    tp = _INSTANCES.get(key)
    if tp is None:
        tp = _INSTANCES[key] = H3TensorParallel(source, dm)
    else:
        tp.blocks = dm.blocks  # shards stay; norms/adaln come from the freshly loaded model
    dm._mgpu_tp = tp
    for i in range(len(dm.blocks)):
        model.set_model_patch_replace(lambda args, extra, i=i: tp.block(i, args, extra["original_block"]),
                                      "dit", "double_block", i)
    if not model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "mmh3_tp"):
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "mmh3_tp", _no_block_prefetch)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "mmh3_tp",
                                   lambda executor, *a, **k: _prepare_sampling(tp, executor, *a, **k))
    return tp
