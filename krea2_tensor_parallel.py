"""Tensor-parallel Krea 2 single-stream DiT block across two GPUs without P2P (Kaggle 2x T4).

Megatron-style split of SingleStreamBlock: attention by heads (wq / gate by output rows, 24 of 48 query heads per GPU;
wk / wv by output rows, 6 of 12 GQA kv heads, so each GPU's query heads use only its own kv heads), wo by input
columns; SwiGLU by hidden width (gate / up by output rows, down by input columns). QK-norm is per head, so no norm
exchange is needed: one all-reduce after wo and one after down per block. Both GPUs keep a replica of the residual
stream and compute modulation and the pre/post norms on it. Exchange, int8 linear and LoRA side paths are shared
with ltx_tensor_parallel. block_forward mirrors SingleStreamBlock.forward of ComfyUI 0.37.0 without reference
latents (timestep_zero_index).
"""
import torch
import torch.nn.functional as F

import comfy.model_management
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked

from . import ltx_tensor_parallel as L

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
