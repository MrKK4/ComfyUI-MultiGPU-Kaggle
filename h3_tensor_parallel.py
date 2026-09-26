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

import torch
from safetensors import safe_open

import comfy.cli_args
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.quant_ops
import comfy.rmsnorm
import comfy.sd
import comfy_kitchen.backends.cuda as ck_cuda
import folder_paths
import comfy.ldm.minimax.model as h3
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

logger = logging.getLogger("MultiGPU")

S = 64.0
HEAD = 128
DEVICES = (torch.device("cuda:0"), torch.device("cuda:1"))
_SHARD_CACHE = {}
CHUNKS = max(1, int(os.environ.get("MMH3_TP_CHUNKS", "4")))


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
            # MMH3_TP_SIDE_STREAMS=0 keeps the copies on the compute streams (no overlap; A/B knob)
            side = os.environ.get("MMH3_TP_SIDE_STREAMS", "0") == "1"
            self.copy = [torch.cuda.Stream(d) if side else torch.cuda.current_stream(d) for d in DEVICES]
        key = (phase, shape)
        if key not in self.bufs:
            self.bufs[key] = {"host": [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in DEVICES],
                              "read": [None, None]}  # rank r's copy stream finished reading host[1-r]
        return self.bufs[key]

    def send(self, phase, shape, rows, parts):
        """D2H of parts[r] (rank r's partial for `rows`) into host[r]; returns the copy-done events."""
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
    if path in _SHARD_CACHE:
        return _SHARD_CACHE[path]
    with safe_open(path, framework="pt", device="cpu") as f:
        layers = json.loads(f.metadata()["_quantization_metadata"])["layers"]
        need = 0
        for k in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            conf = layers.get(f"blocks.0.{k}", {})
            if conf.get("format") != "int8_tensorwise" or not conf.get("convrot"):
                raise ValueError(f"tensor parallel needs the int8 convrot checkpoint; blocks.0.{k} is {conf}")
            need += f.get_slice(f"blocks.0.{k}.weight").get_shape()[0] * f.get_slice(f"blocks.0.{k}.weight").get_shape()[1]
        need = int(need * n_blocks / 2 * 1.08)
        for dev in DEVICES:
            comfy.model_management.free_memory(need + (2 << 30), dev)
        shards = [[], []]
        for i in range(n_blocks):
            p = f"blocks.{i}."
            g = lambda k: f.get_tensor(p + k)
            qkv, qkv_s, out, out_s = g("attn.qkv_proj.weight"), g("attn.qkv_proj.weight_scale"), g("attn.out_proj.weight"), g("attn.out_proj.weight_scale")
            fc1, fc1_s, fc2, fc2_s = g("mlp.fc1.weight"), g("mlp.fc1.weight_scale"), g("mlp.fc2.weight"), g("mlp.fc2.weight_scale")
            small = {n: g(k).float() for n, k in (("norm1", "norm1.weight"), ("norm2", "norm2.weight"))}
            qn, kn = g("attn.q_norm.weight"), g("attn.k_norm.weight")
            inner, ffn = qkv.shape[0] // 3, fc1.shape[0] // 2
            for rank, dev in enumerate(DEVICES):
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
    logger.info("[MultiGPU TP] %d blocks sharded over %s", n_blocks, ", ".join(map(str, DEVICES)))
    return shards


def _lin(x, w, act=None):
    return ck_cuda.int8_linear(x, w[0], w[1], convrot=True, convrot_groupsize=256, input_act=act)


class H3TensorParallel:
    def __init__(self, path, diffusion_model):
        self.path = path
        self.blocks = diffusion_model.blocks
        self.shards = None
        self.xchg = _ChunkedExchange()
        blk = self.blocks[0]
        self.eps1, self.eps2 = blk.norm1.eps, blk.norm2.eps
        self.qk_eps = blk.attn.q_norm.eps
        self.checked = os.environ.get("MMH3_TP_CHECK", "0") != "1"

    def _attn(self, x, sh, shift, scale, segs, rope, topts):
        h = h3._mod_scale_shift(comfy.rmsnorm.rms_norm(x, sh["norm1"], self.eps1), shift, scale, segs).to(torch.float16)
        s, heads = h.shape[0], sh["heads"]
        q, k, v = _lin(h, sh["qkv"]).split(sh["inner"], dim=-1)
        v = v.view(s, heads, HEAD)
        q = q.view(1, s, heads, HEAD)
        k = k.view(1, s, heads, HEAD)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope, sh["qn"], sh["kn"], epsilon=self.qk_eps, rot_dim=rope.shape[-3] * 2)
        q = AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        return optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, transformer_options=topts).squeeze(0)

    def _mlp(self, x, sh, shift, scale, segs):
        h = h3._mod_scale_shift(comfy.rmsnorm.rms_norm(x, sh["norm2"], self.eps2), shift, scale, segs).to(torch.float16)
        a = _lin(h, sh["fc1"])
        a[..., a.shape[-1] // 2:].mul_(1.0 / S)
        return _lin(a, sh["fc2"], act="swiglu")

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
        if self.shards is None:
            # permanent allocations: keep them out of the compiler's per-forward allocation graph
            with comfy.model_prefetch.pause_malloc_graph(sync=True):
                self.shards = _load_shards(self.path, len(self.blocks))
        blk = self.blocks[i]
        x0 = args["img"].float()
        st = self._step_state(args["img"], args)
        x1, segs0, segs1, topts = st["x1"], args["mod_segments"], st["segs1"], args["transformer_options"]
        sh0, sh1 = self.shards[0][i], self.shards[1][i]
        mods0 = blk.adaln_proj(args["t_emb"])
        with torch.cuda.device(d1):
            mods1 = [m.to(d1, non_blocking=True) for m in mods0]
        ref = None
        if not self.checked:
            ref = original({**args, "img": args["img"].clone()})["img"].float()

        xs, shs, mods, segs = (x0, x1), (sh0, sh1), (mods0, mods1), (segs0, segs1)
        os_ = _per_rank(lambda r: self._attn(xs[r], shs[r], mods[r][0], mods[r][1], segs[r], st["rope%d" % r], topts))
        # per-token tail pipelined in row chunks: out_proj -> swap -> gate -> mlp -> swap -> gate
        n = x0.shape[0]
        shape, size = (n, x0.shape[1]), -(-n // CHUNKS)
        chunks = [(a, min(a + size, n)) for a in range(0, n, size)]
        csegs = [[_chunk_segs(segs[r], a, b) for r in range(2)] for a, b in chunks]
        self.xchg.begin("attn", shape)  # after attention: keeps the qkv peak unchanged
        own_a, sent_a = [], []
        for a, b in chunks:
            own_a.append(_per_rank(lambda r: _lin(os_[r][a:b] * (1.0 / S), shs[r]["out"])))
            sent_a.append(self.xchg.send("attn", shape, (a, b), own_a[-1]))
        del os_
        self.xchg.begin("mlp", shape)
        own_m, sent_m = [], []
        for j, (a, b) in enumerate(chunks):
            got = self.xchg.recv("attn", shape, (a, b), sent_a[j])

            def attn_then_mlp(r):
                xc = xs[r][a:b]
                h3._mod_gate(xc, mods[r][2], (own_a[j][r] + got[r]).float() * S, csegs[j][r])
                own_a[j][r] = None
                return self._mlp(xc, shs[r], mods[r][3], mods[r][4], csegs[j][r])
            own_m.append(_per_rank(attn_then_mlp))
            sent_m.append(self.xchg.send("mlp", shape, (a, b), own_m[j]))
        for j, (a, b) in enumerate(chunks):
            got = self.xchg.recv("mlp", shape, (a, b), sent_m[j])
            _per_rank(lambda r: h3._mod_gate(xs[r][a:b], mods[r][5], (own_m[j][r] + got[r]).float() * S, csegs[j][r]))
            own_m[j] = None

        if ref is not None:
            self.checked = True
            rel = float((x0 - ref).norm() / ref.norm())
            drift = float((x0 - x1.to(d0)).abs().max())
            logger.info(f"[MultiGPU TP] block {i} check: rel err vs single-GPU {rel:.2e}, replica drift {drift:.2e}")
        st["x0"], st["x1"] = x0, x1
        return {"img": x0}


def _no_block_prefetch(executor, *args, **kwargs):
    # the model's own block weights are unused; prefetching would page all 21 GB in every step
    topts = kwargs.get("transformer_options", args[3] if len(args) > 3 else None)
    if topts is not None:
        topts["prefetch_dynamic_vbars"] = False
    return executor(*args, **kwargs)


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
        dm = model.model.diffusion_model
        if not isinstance(dm, h3.MiniMaxH3Model):
            raise ValueError("UNETLoaderH3TensorParallel only supports MiniMax H3 checkpoints")
        # the Comfy compiler records one device's allocations per forward; TP allocates on two.
        # ponytail: process-wide, Split in the same session runs without the compiler
        comfy.cli_args.args.disable_comfy_compiler = True
        tp = H3TensorParallel(path, dm)
        for i in range(len(dm.blocks)):
            model.set_model_patch_replace(lambda args, extra, i=i: tp.block(i, args, extra["original_block"]),
                                          "dit", "double_block", i)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "mmh3_tp", _no_block_prefetch)
        return (model,)
