"""Tensor-parallel Krea 2 single-stream DiT block across two GPUs without P2P (Kaggle 2x T4).

Megatron-style split of SingleStreamBlock: attention by heads (wq / gate by output rows, 24 of 48 query heads per GPU;
wk / wv by output rows, 6 of 12 GQA kv heads, so each GPU's query heads use only its own kv heads), wo by input
columns; SwiGLU by hidden width (gate / up by output rows, down by input columns). QK-norm is per head, so no norm
exchange is needed: one all-reduce after wo and one after down per block. Both GPUs keep a replica of the residual
stream and compute modulation and the pre/post norms on it. Exchange, int8 linear and LoRA side paths are shared
with ltx_tensor_parallel. block_forward mirrors SingleStreamBlock.forward of ComfyUI 0.37.0 without reference
latents (timestep_zero_index; those calls run the original block). UNETLoaderKrea2TensorParallel plugs it into sampling.
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
import comfy.sd
import folder_paths
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.krea2.model import SingleStreamBlock
from comfy.ldm.modules.attention import optimized_attention_masked

from . import ltx_tensor_parallel as L

logger = logging.getLogger("MultiGPU")

DEVICES = L.DEVICES
COLUMN = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "mlp.gate", "mlp.up")
ROW = ("attn.wo", "mlp.down")


def _block_cpu(f, keys, pre, rank):
    sh = {n: L._load_lin(f, keys, {}, pre + n, 0, rank) for n in COLUMN}
    sh.update({n: L._load_lin(f, keys, {}, pre + n, 1, rank) for n in ROW})
    for n in ("mod.lin", "prenorm.scale", "postnorm.scale", "attn.qknorm.qnorm.scale", "attn.qknorm.knorm.scale"):
        sh[n] = f.get_tensor(pre + n)
    return sh


def block_prefix(f):
    keys = [k for k in f.keys() if k.endswith("blocks.0.attn.wq.weight") and "txtfusion" not in k]
    if len(keys) != 1:
        raise RuntimeError("cannot find blocks.0.attn.wq.weight in the checkpoint")
    return keys[0][:-len("0.attn.wq.weight")]


def load_block_shards(path, i):
    """[rank-0 shard, rank-1 shard] of block i, on DEVICES[0] / DEVICES[1]."""
    f, keys, _ = L._open(path)
    pre = block_prefix(f) + "%d." % i
    return [L._to(_block_cpu(f, keys, pre, rank), dev) for rank, dev in enumerate(DEVICES)]


def _rms(x, scale):
    w = comfy.model_management.cast_to(scale, dtype=torch.float32, device=x.device) + 1.0
    return F.rms_norm(x.float(), (x.shape[-1],), weight=w, eps=1e-5).to(x.dtype)


def block_forward(sh, xs, vecs, freqs, heads, kvheads, topts):
    """SingleStreamBlock.forward on both GPUs: xs / vecs / freqs / topts are per-rank lists, sh = [rank0, rank1] shards.
    heads / kvheads are the full-model counts. Returns the per-rank (identical) block outputs."""
    h, kvh = heads // 2, kvheads // 2
    mods = L._ranks(lambda r: (vecs[r] + comfy.model_management.cast_to(sh[r]["mod.lin"], dtype=vecs[r].dtype, device=vecs[r].device)).chunk(6, dim=-1))
    B, T, _ = xs[0].shape

    pre = L._ranks(lambda r: (1 + mods[r][0]) * _rms(xs[r], sh[r]["prenorm.scale"]) + mods[r][1])

    def qkv(r):
        s, x = sh[r], pre[r]
        q = L._lin(x, s["attn.wq"]).view(B, T, h, -1).transpose(1, 2)
        k = L._lin(x, s["attn.wk"]).view(B, T, kvh, -1).transpose(1, 2)
        v = L._lin(x, s["attn.wv"]).view(B, T, kvh, -1).transpose(1, 2)
        q, k = _rms(q, s["attn.qknorm.qnorm.scale"]), _rms(k, s["attn.qknorm.knorm.scale"])
        q, k = apply_rope(q, k, freqs[r])
        rep = h // kvh
        return q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

    qkvs = L._ranks(qkv)

    def attn_part(r, lo, hi):
        q, k, v = qkvs[r]
        o = optimized_attention_masked(q[:, :, lo:hi].contiguous(), k, v, h, mask=None, skip_reshape=True, transformer_options=topts[r])
        return L._lin(o * torch.sigmoid(L._lin(pre[r][:, lo:hi], sh[r]["attn.gate"])), sh[r]["attn.wo"])

    attn = L.XCHG.allreduce_produce(attn_part, (B, T, xs[0].shape[-1]), xs[0].dtype, "k2_attn")
    del qkvs, pre
    xs = L._ranks(lambda r: xs[r] + mods[r][2] * attn[r])
    del attn
    post = L._ranks(lambda r: (1 + mods[r][3]) * _rms(xs[r], sh[r]["postnorm.scale"]) + mods[r][4])

    def mlp_part(r, lo, hi):
        x, s = post[r][:, lo:hi], sh[r]
        return L._lin(F.silu(L._lin(x, s["mlp.gate"])).mul_(L._lin(x, s["mlp.up"])), s["mlp.down"])

    mlp = L.XCHG.allreduce_produce(mlp_part, (B, T, xs[0].shape[-1]), xs[0].dtype, "k2_mlp")
    return L._ranks(lambda r: xs[r] + mods[r][5] * mlp[r])


# ----------------------------------------------------------------------------------------------- sampling integration

_INSTANCES = {}


class Krea2TensorParallel:
    """Runs every SingleStreamBlock on both GPUs. Shards are built from the checkpoint at the first block call (the
    model's own block weights are never paged in) and stay resident on both GPUs for the server's lifetime."""

    def __init__(self, path, dm):
        self.path, self.blocks = path, dm.blocks
        self.heads, self.kvheads = dm.blocks[0].attn.heads, dm.blocks[0].attn.kvheads
        self.gpu = None              # [rank][block]
        self.state = None            # per-forward rank-1 inputs + last outputs
        self.checked = os.environ.get("KREA2_TP_CHECK", "1") == "0"
        self.lora = ({}, (), [])     # (spec, key, unsupported) from the model's patches at sampling start
        self.lora_applied = ()

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
        logger.info("[MultiGPU Krea2 TP] %d blocks sharded in %.0fs: %.2f / %.2f GB | %s", len(self.blocks), time.perf_counter() - t0,
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
                parts = [L.lora_rank_parts(a, b, scale, axis, rank, dev) for a, b, scale in loras]
                self.gpu[rank][block][name].lora = parts
                n_bytes += sum(t.numel() * t.element_size() for pr in parts for t in pr)
        self.lora_applied = key
        if spec:
            logger.info("[MultiGPU Krea2 TP] LoRA: %d layers in %d blocks applied as tensor-parallel side paths (%.0f MB on both GPUs)",
                        len(spec), len({b for b, _, _ in spec}), n_bytes / 2**20)

    def block(self, i, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
        blk = self.blocks[i]
        if mask is not None or timestep_zero_index is not None:  # reference latents: not split yet
            self.state = None
            return SingleStreamBlock.forward(blk, x, vec, freqs, mask, timestep_zero_index, transformer_options)
        if self.gpu is None:
            with comfy.model_prefetch.pause_malloc_graph(sync=True):
                self._load()
        if i == 0:
            if self.lora_applied != self.lora[1]:
                self._apply_lora()
            for d in DEVICES:
                torch.cuda.synchronize(d)
            self.t_fwd = time.perf_counter()
            with torch.cuda.device(DEVICES[1]):
                self.state = {"in": [vec.to(DEVICES[1]), freqs.to(DEVICES[1])], "x0": None, "x1": None}
        st = self.state
        x1 = st["x1"] if st["x0"] is x else x.to(DEVICES[1])
        ref = None
        if not self.checked:
            # one-GPU reference of this block (ComfyUI's own path, incl. its LoRA patching), once per server
            ref = SingleStreamBlock.forward(blk, x.clone(), vec, freqs, None, None, transformer_options)
        out = block_forward([self.gpu[0][i], self.gpu[1][i]], [x, x1], [vec, st["in"][0]], [freqs, st["in"][1]],
                            self.heads, self.kvheads, [transformer_options, transformer_options])
        if ref is not None:
            self.checked = True
            rel = float((out[0].float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20))
            drift = float((out[0].float() - out[1].float().to(DEVICES[0])).abs().max())
            logger.info("[MultiGPU Krea2 TP] block %d check (%d tokens, LoRA layers %d): rel err vs one GPU %.2e, replica drift %.2e",
                        i, x.shape[1], len(self.lora[0]), rel, drift)
        st["x0"], st["x1"] = out[0], out[1]
        if i == len(self.blocks) - 1:
            for d in DEVICES:
                torch.cuda.synchronize(d)
            logger.info("[MultiGPU Krea2 TP] forward: %d tokens, %.2fs", x.shape[1], time.perf_counter() - self.t_fwd)
            self.state = None
        return out[0]


# block-relative module path -> split axis (0 = output rows, 1 = input columns)
_LORA_AXIS = dict([(n, 0) for n in COLUMN] + [(n, 1) for n in ROW])


def lora_spec(patches):
    """Plain LoRA patches on block linears -> ({(block, name, axis): [(A, B, scale)]}, change key, unsupported keys)."""
    spec, key, unsupported = {}, [], []
    for k, plist in patches.items():
        m = re.search(r"(?:^|\.)blocks\.(\d+)\.(.+)\.weight$", k)
        if not m or "txtfusion" in k:
            continue
        axis = _LORA_AXIS.get(m.group(2))
        for strength, adapter, strength_model, offset, function in plist:
            w = getattr(adapter, "weights", None)
            if not (axis is not None and offset is None and function is None and strength_model == 1.0
                    and type(adapter).__name__ == "LoRAAdapter" and w is not None and len(w) >= 3 and all(x is None for x in w[3:])):
                unsupported.append(k)
                continue
            up, down, alpha = w[0], w[1], w[2]
            scale = float(strength) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
            spec.setdefault((int(m.group(1)), m.group(2), axis), []).append((down, up, scale))
            key.append((k, id(adapter), float(strength)))
    return spec, tuple(sorted(key)), unsupported


def _sampling(tp, executor, *args, **kwargs):
    tp.lora = lora_spec(getattr(executor.class_obj.model_patcher, "patches", {}))
    if tp.lora[2]:
        logger.warning("[MultiGPU Krea2 TP] %d patch keys on blocks are NOT applied under tensor parallel (only plain LoRA on the "
                       "block linears is; e.g. %s)", len(tp.lora[2]), tp.lora[2][0])
    t0 = time.perf_counter()
    try:
        return executor(*args, **kwargs)
    finally:
        tp.state = None
        L.XCHG.bufs.clear()  # pinned exchange buffers: rebuilt next run
        logger.info("[MultiGPU Krea2 TP] sampling run %.1fs | %s | %s", time.perf_counter() - t0, L._gpu_free(), L._ram())


class UNETLoaderKrea2TensorParallel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "multigpu"

    def load(self, unet_name):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Krea 2 tensor parallel needs two CUDA devices")
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path)
        dm = model.model.diffusion_model
        if not hasattr(dm, "blocks") or not isinstance(dm.blocks[0], SingleStreamBlock):
            raise ValueError("UNETLoaderKrea2TensorParallel only supports Krea 2 checkpoints")
        comfy.cli_args.args.disable_comfy_compiler = True  # the compiler records one device's allocations; TP allocates on two
        tp = _INSTANCES.get(path)
        if tp is None:
            tp = _INSTANCES[path] = Krea2TensorParallel(path, dm)
        tp.blocks = dm.blocks  # shards stay; only the fallback path and the check use the model's own blocks
        for i, blk in enumerate(dm.blocks):
            blk.forward = lambda *a, i=i, **k: tp.block(i, *a, **k)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_tp", L._no_block_prefetch)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "krea2_tp",
                                   lambda executor, *a, **k: _sampling(tp, executor, *a, **k))
        return (model,)
