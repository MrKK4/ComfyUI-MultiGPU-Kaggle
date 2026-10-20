"""Tensor-parallel Qwen-Image 2.1 transformer block across two GPUs without P2P (Kaggle 2x T4).

Megatron-style split of QwenImage21TransformerBlock: attention by heads (to_q / to_k / to_v by output rows, 16 of 32
heads per GPU), to_out by input columns; SwiGLU by hidden width (gate_up holds [gate; up], so each GPU takes its rows of
both halves and keeps the [gate; up] order for the fused swiglu input of `out`; out by input columns). QK-norm is per
head, so one all-reduce after to_out and one after out per block. Both GPUs keep a replica of the residual stream.

The model's attention closure carries the block-causal segments and the text prefix K/V cache; block_forward reads it
(ComfyUI v0.41 block_causal_attention / prefix_cached_attention) and runs the same attention per head half: the prefix
pass gathers both halves into the cache, cached steps slice the cached heads. Unknown closures run the original block.
fp16 compute (the model config allows only bf16 / fp32; Phase 0 on T4: fp16 no overflow, max |x| ~8.4k, same quality).
"""
import logging
import os
import re
import time

import torch
import torch.nn.functional as F

import comfy.cli_args
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.quant_ops
import comfy.sd
import comfy.supported_models
import folder_paths
from comfy.ldm.qwen_image21 import model as qm

from . import ltx_tensor_parallel as L

logger = logging.getLogger("MultiGPU")

DEVICES = L.DEVICES
COLUMN = {"to_q": "attn.to_q", "to_k": "attn.to_k", "to_v": "attn.to_v"}
ROW = {"to_out": "attn.to_out.0", "out": "img_mlp.out"}


def _gate_up_rows(n, rank):
    """Rows of the fused [gate; up] weight (n rows) that rank holds: its half of gate, then its half of up."""
    h = n // 2
    return torch.cat([torch.arange(rank * h // 2, (rank + 1) * h // 2), torch.arange(h + rank * h // 2, h + (rank + 1) * h // 2)])


def _load_gate_up(f, keys, name, rank):
    w = f.get_tensor(name + ".weight")
    rows = _gate_up_rows(w.shape[0], rank)
    if w.dtype != torch.int8:
        return L.Lin(w.half()[rows].contiguous(), None, None, False)
    conf = L._quant_conf(f, keys, {}, name)
    if conf.get("format") != "int8_tensorwise" or not conf.get("convrot") or int(conf.get("convrot_groupsize", 256)) != 256:
        raise RuntimeError("%s: tensor parallel needs int8_tensorwise convrot g256, got %s" % (name, conf))
    s = f.get_tensor(name + ".weight_scale").float()
    if s.numel() > 1:   # per-row scale, [N] or [N, 1]
        s = s[rows]
    return L.Lin(w[rows].contiguous(), s.contiguous(), None, True)


def _block_cpu(f, keys, pre, rank):
    sh = {n: L._load_lin(f, keys, {}, pre + k, 0, rank) for n, k in COLUMN.items()}
    sh.update({n: L._load_lin(f, keys, {}, pre + k, 1, rank) for n, k in ROW.items()})
    sh["gate_up"] = _load_gate_up(f, keys, pre + "img_mlp.gate_up", rank)
    for n in ("norm_q", "norm_k"):
        sh[n] = f.get_tensor(pre + "attn.%s.weight" % n).half()
    return sh


def block_prefix(f):
    keys = [k for k in f.keys() if k.endswith("transformer_blocks.0.attn.to_q.weight")]
    if len(keys) != 1:
        raise RuntimeError("cannot find transformer_blocks.0.attn.to_q.weight in the checkpoint")
    return keys[0][:-len("0.attn.to_q.weight")]


def _closure(fn):
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ or ())))


def attn_kind(attn_fn):
    """'target' (cached prefix K/V, target rows only), 'prefix' (fills the cache) or 'full' (whole sequence, no cache);
    None for an attention closure this module does not know."""
    name = getattr(attn_fn, "__qualname__", "")
    if name == "prefix_cached_attention.<locals>.attn":
        return "target"
    if name == "block_causal_attention.<locals>.attn":
        return "full" if _closure(attn_fn).get("cache") is None else "prefix"
    return None


def _swiglu(y):
    gate, up = y.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def _mlp_out(y, p):
    if p.int8 and not p.lora:   # swiglu folded into out's input quantizer, as SwiGLUFeedForward does
        return comfy.quant_ops.ck.int8_linear(y, p.w, p.s, None, y.dtype, convrot=True, convrot_groupsize=256, input_act="swiglu")
    return L._lin(_swiglu(y), p)   # LoRA side path needs the activated input


def block_forward(blk, sh, xs, mods, pes, attn_fn, kind, prefix_len, heads, topts, extra):
    """QwenImage21TransformerBlock.forward on both GPUs: xs / mods / pes are per-rank, sh = [rank0, rank1] shards.
    extra: per-rank segments ('prefix' / 'full'). Returns the per-rank (identical) block outputs."""
    h = heads // 2
    B, N, D = xs[0].shape
    pref = blk.attn.comfy_attention
    C = qm.AttentionTensorContainer
    pre = L._ranks(lambda r: qm._modulated_norm(blk.img_norm1, xs[r], mods[r][0], prefix_len, mods[r][4]))

    def qkv(r):
        s, x = sh[r], pre[r]
        q = L._lin(x, s["to_q"]).view(B, N, h, -1)
        k = L._lin(x, s["to_k"]).view(B, N, h, -1)
        v = L._lin(x, s["to_v"]).view(B, N, h, -1)
        q, k = comfy.quant_ops.ck.rms_rope(q, k, pes[r], s["norm_q"].to(q.dtype), s["norm_k"].to(k.dtype), blk.attn.norm_q.eps)
        return q, k, v

    qkvs = L._ranks(qkv)
    del pre
    if kind == "target":
        c = _closure(attn_fn)

        def kv(r):
            pk, pv = (t[:, :, r * h:(r + 1) * h].to(DEVICES[r]) for t in (c["prefix_k"], c["prefix_v"]))
            return torch.cat([pk, qkvs[r][1]], dim=1).flatten(2), torch.cat([pv, qkvs[r][2]], dim=1).flatten(2)

        kvs = L._ranks(kv)

        def attn_part(r, lo, hi):
            o = qm.optimized_attention(C(qkvs[r][0][:, lo:hi].flatten(2)), C(kvs[r][0]), C(kvs[r][1]), h,
                                       transformer_options=topts, preferred_attention=pref)
            return L._lin(o, sh[r]["to_out"])
    else:
        c = _closure(attn_fn)
        if kind == "prefix":   # the cache holds every head: gather GPU 1's half (prefix rows only, small)
            pl = c["prefix_len"]
            k = torch.cat([qkvs[0][1][:, :pl], qkvs[1][1][:, :pl].to(DEVICES[0])], dim=2)
            v = torch.cat([qkvs[0][2][:, :pl], qkvs[1][2][:, :pl].to(DEVICES[0])], dim=2)
            c["cache"].put(c["block_index"], torch.stack([k, v], dim=1))
            del k, v
        os_ = L._ranks(lambda r: qm.block_causal_attention(extra[r], topts, None, c["block_index"], c["prefix_len"])(
            C(qkvs[r][0]), C(qkvs[r][1]), C(qkvs[r][2]), h, preferred_attention=pref))

        def attn_part(r, lo, hi):
            return L._lin(os_[r][:, lo:hi], sh[r]["to_out"])

    attn = L.XCHG.allreduce_produce(attn_part, (B, N, D), xs[0].dtype, "q21_attn")
    del qkvs
    xs = L._ranks(lambda r: qm._gated_residual(xs[r], attn[r], mods[r][1], prefix_len))
    del attn
    post = L._ranks(lambda r: qm._modulated_norm(blk.img_norm2, xs[r], mods[r][2], prefix_len, mods[r][4]))

    def mlp_part(r, lo, hi):
        return _mlp_out(L._lin(post[r][:, lo:hi], sh[r]["gate_up"]), sh[r]["out"])

    mlp = L.XCHG.allreduce_produce(mlp_part, (B, N, D), xs[0].dtype, "q21_mlp")
    del post

    def fin(r):
        x = qm._gated_residual(xs[r], mlp[r], mods[r][3], prefix_len)
        return x.clamp_(-65504, 65504) if x.dtype == torch.float16 else x

    return L._ranks(fin)


# ----------------------------------------------------------------------------------------------- sampling integration

_INSTANCES = {}


class Qwen21TensorParallel:
    """Runs every transformer block on both GPUs. Shards are built from the checkpoint at the first block call (the
    model's own block weights are never paged in) and stay resident on both GPUs for the server's lifetime."""

    def __init__(self, path, dm):
        self.path, self.blocks = path, dm.transformer_blocks
        self.heads = dm.transformer_blocks[0].attn.heads
        self.gpu = None              # [rank][block]
        self.state = {}              # kind -> rank-1 copies of this forward's inputs + last outputs
        self.checked = os.environ.get("QWEN21_TP_CHECK", "1") == "0"
        self.lora = ({}, (), [])
        self.lora_applied = ()
        self.fallback = 0

    def _load(self):
        need = int(sum(p.numel() * p.element_size() for p in self.blocks.parameters()) / 2 * 1.05)
        for d in DEVICES:
            comfy.model_management.free_memory(need + (1 << 30), d)
        f, keys, _ = L._open(self.path)
        prefix, t0 = block_prefix(f), time.perf_counter()
        self.gpu = [[], []]
        for i in range(len(self.blocks)):
            for rank, dev in enumerate(DEVICES):
                self.gpu[rank].append(L._to(_block_cpu(f, keys, prefix + "%d." % i, rank), dev))
        logger.info("[MultiGPU Qwen21 TP] %d blocks sharded in %.0fs: %.2f / %.2f GB | %s", len(self.blocks), time.perf_counter() - t0,
                    sum(map(L.shard_bytes, self.gpu[0])) / 2**30, sum(map(L.shard_bytes, self.gpu[1])) / 2**30, L._gpu_free())

    def _apply_lora(self):
        spec, key, _ = self.lora
        for rank in range(2):
            for sh in self.gpu[rank]:
                for v in sh.values():
                    if isinstance(v, L.Lin):
                        v.lora = None
        n_bytes = 0
        for (block, name, axis), loras in spec.items():
            for rank, dev in enumerate(DEVICES):
                parts = [lora_rank_parts(a, b, scale, axis, rank, dev) for a, b, scale in loras]
                self.gpu[rank][block][name].lora = parts
                n_bytes += sum(t.numel() * t.element_size() for pr in parts for t in pr)
        self.lora_applied = key
        if spec:
            logger.info("[MultiGPU Qwen21 TP] LoRA: %d layers in %d blocks applied as tensor-parallel side paths (%.0f MB on both GPUs)",
                        len(spec), len({b for b, _, _ in spec}), n_bytes / 2**20)

    def block(self, i, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
        blk = self.blocks[i]
        kind = attn_kind(attn_fn)
        if kind is None or comfy.model_management.in_training or transformer_options.get("patches", {}).get("attn1_patch"):
            self.fallback += 1
            return qm.QwenImage21TransformerBlock.forward(blk, x, mod, pe, attn_fn, prefix_len, transformer_options)
        if self.gpu is None:
            with comfy.model_prefetch.pause_malloc_graph(sync=True):
                self._load()
        if self.lora_applied != self.lora[1]:
            self._apply_lora()
        st = self.state.get(kind)
        if i == 0 or st is None:
            for d in DEVICES:
                torch.cuda.synchronize(d)
            st = self.state[kind] = {"t": time.perf_counter(), "x0": None, "x1": None, "ids": None}
        if st["ids"] != (id(mod), id(pe)):
            with torch.cuda.device(DEVICES[1]):
                st["mod1"], st["pe1"] = L._to(mod, DEVICES[1]), pe.to(DEVICES[1])
            st["ids"] = (id(mod), id(pe))
        extra = None
        if kind != "target":
            segs = _closure(attn_fn)["segments"]
            with torch.cuda.device(DEVICES[1]):
                extra = [segs, L._to(segs, DEVICES[1])]
        x1 = st["x1"] if st["x0"] is x else x.to(DEVICES[1])
        ref = None
        if not self.checked and kind != "prefix":
            # one-GPU reference of this block (ComfyUI's own path, incl. its LoRA patching), once per server
            ref = qm.QwenImage21TransformerBlock.forward(blk, x.clone(), mod, pe, attn_fn, prefix_len, transformer_options)
        out = block_forward(blk, [self.gpu[0][i], self.gpu[1][i]], [x, x1], [mod, st["mod1"]], [pe, st["pe1"]], attn_fn, kind,
                            prefix_len, self.heads, transformer_options, extra)
        if ref is not None:
            self.checked = True
            rel = float((out[0].float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20))
            drift = float((out[0].float() - out[1].float().to(DEVICES[0])).abs().max())
            logger.info("[MultiGPU Qwen21 TP] block %d check (%s, %d tokens, %s, LoRA layers %d): rel err vs one GPU %.2e, replica drift %.2e",
                        i, kind, x.shape[1], x.dtype, len(self.lora[0]), rel, drift)
        st["x0"], st["x1"] = out[0], out[1]
        if i == len(self.blocks) - 1:
            for d in DEVICES:
                torch.cuda.synchronize(d)
            logger.info("[MultiGPU Qwen21 TP] %s forward: %d tokens, %.2fs", kind, x.shape[1], time.perf_counter() - st["t"])
            self.state.pop(kind, None)
        return out[0]


# patch key (block-relative) -> (shard name, split axis: 0 output rows, 1 input columns, "gu" fused gate/up rows)
_LORA_LAYERS = {"attn.to_q": ("to_q", 0), "attn.to_k": ("to_k", 0), "attn.to_v": ("to_v", 0), "attn.to_out.0": ("to_out", 1),
                "img_mlp.gate_up": ("gate_up", "gu"), "img_mlp.out": ("out", 1)}


def lora_rank_parts(a, b, scale, axis, rank, dev):
    if axis != "gu":
        return L.lora_rank_parts(a, b, scale, axis, rank, dev)
    b = b[_gate_up_rows(b.shape[0], rank).to(b.device)]
    return a.to(dev, torch.float16).t().contiguous(), (b.to(dev, torch.float32) * scale).to(torch.float16).t().contiguous()


def lora_spec(patches):
    """Plain LoRA patches on block linears -> ({(block, shard name, axis): [(A, B, scale)]}, change key, unsupported keys)."""
    spec, key, unsupported = {}, [], []
    for k, plist in patches.items():
        m = re.search(r"(?:^|\.)transformer_blocks\.(\d+)\.(.+)\.weight$", k)
        if not m:
            continue
        layer = _LORA_LAYERS.get(m.group(2))
        for strength, adapter, strength_model, offset, function in plist:
            w = getattr(adapter, "weights", None)
            if not (layer is not None and offset is None and function is None and strength_model == 1.0
                    and type(adapter).__name__ == "LoRAAdapter" and w is not None and len(w) >= 3 and all(x is None for x in w[3:])):
                unsupported.append(k)
                continue
            up, down, alpha = w[0], w[1], w[2]
            scale = float(strength) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
            spec.setdefault((int(m.group(1)),) + layer, []).append((down, up, scale))
            key.append((k, id(adapter), float(strength)))
    return spec, tuple(sorted(key)), unsupported


def _sampling(tp, executor, *args, **kwargs):
    tp.lora = lora_spec(getattr(executor.class_obj.model_patcher, "patches", {}))
    if tp.lora[2]:
        logger.warning("[MultiGPU Qwen21 TP] %d patch keys on blocks are NOT applied under tensor parallel (only plain LoRA on the "
                       "block linears is; e.g. %s)", len(tp.lora[2]), tp.lora[2][0])
    t0 = time.perf_counter()
    try:
        return executor(*args, **kwargs)
    finally:
        tp.state = {}
        L.XCHG.bufs.clear()  # pinned exchange buffers: rebuilt next run
        logger.info("[MultiGPU Qwen21 TP] sampling run %.1fs | one-GPU fallback blocks %d | %s | %s",
                    time.perf_counter() - t0, tp.fallback, L._gpu_free(), L._ram())
        tp.fallback = 0


class UNETLoaderQwen21TensorParallel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "multigpu"

    def load(self, unet_name):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Qwen-Image 2.1 tensor parallel needs two CUDA devices")
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        cfg = comfy.supported_models.QwenImage21
        allowed = cfg.supported_inference_dtypes
        cfg.supported_inference_dtypes = [torch.float16] + [d for d in allowed if d != torch.float16]  # the shards compute in fp16
        try:
            model = comfy.sd.load_diffusion_model(path)
        finally:
            cfg.supported_inference_dtypes = allowed
        dm = model.model.diffusion_model
        if not hasattr(dm, "transformer_blocks") or not isinstance(dm.transformer_blocks[0], qm.QwenImage21TransformerBlock):
            raise ValueError("UNETLoaderQwen21TensorParallel only supports Qwen-Image 2.1 checkpoints")
        comfy.cli_args.args.disable_comfy_compiler = True  # the compiler records one device's allocations; TP allocates on two
        tp = _INSTANCES.get(path)
        if tp is None:
            tp = _INSTANCES[path] = Qwen21TensorParallel(path, dm)
        tp.blocks = dm.transformer_blocks  # shards stay; only the fallback path and the check use the model's own blocks
        for i, blk in enumerate(dm.transformer_blocks):
            blk.forward = lambda *a, i=i, **k: tp.block(i, *a, **k)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "qwen21_tp", L._no_block_prefetch)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "qwen21_tp",
                                   lambda executor, *a, **k: _sampling(tp, executor, *a, **k))
        return (model,)
