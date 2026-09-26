"""Tensor-parallel MiniMax H3 DiT across two GPUs without P2P (Kaggle 2x T4).

Every DiT block is split Megatron-style: qkv_proj / fc1 by output rows (half the heads and
half the MLP width per GPU), out_proj / fc2 by input columns. Each GPU keeps a replica of the
fp32 residual stream; the two row-parallel partial sums are exchanged through pinned host
buffers, chained with CUDA events so neither GPU waits on the host. Shards come straight from
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


class _HostAllReduce:
    """p0 (cuda:0) + p1 (cuda:1) on both devices, via a two-slot ring of pinned buffers."""

    def __init__(self):
        self.slots = {}
        self.turn = {}

    def __call__(self, p0, p1):
        d0, d1 = DEVICES
        key = tuple(p0.shape)
        if key not in self.slots:
            self.slots[key] = [{"h0": torch.empty(key, dtype=p0.dtype, pin_memory=True),
                                "h1": torch.empty(key, dtype=p1.dtype, pin_memory=True),
                                "read0": None, "read1": None} for _ in range(2)]
            self.turn[key] = 0
        slot = self.slots[key][self.turn[key]]
        self.turn[key] ^= 1
        s0, s1 = torch.cuda.current_stream(d0), torch.cuda.current_stream(d1)
        # a slot's buffers may still be read by the previous exchange that used it
        if slot["read1"] is not None:
            s0.wait_event(slot["read1"])
        if slot["read0"] is not None:
            s1.wait_event(slot["read0"])
        with torch.cuda.device(d0):
            slot["h0"].copy_(p0, non_blocking=True)
            sent0 = torch.cuda.Event()
            sent0.record(s0)
        with torch.cuda.device(d1):
            slot["h1"].copy_(p1, non_blocking=True)
            sent1 = torch.cuda.Event()
            sent1.record(s1)
        with torch.cuda.device(d0):
            s0.wait_event(sent1)
            r0 = p0 + slot["h1"].to(d0, non_blocking=True)
            slot["read0"] = torch.cuda.Event()
            slot["read0"].record(s0)
        with torch.cuda.device(d1):
            s1.wait_event(sent0)
            r1 = p1 + slot["h0"].to(d1, non_blocking=True)
            slot["read1"] = torch.cuda.Event()
            slot["read1"].record(s1)
        return r0, r1


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
        self.allreduce = _HostAllReduce()
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
        o = optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, transformer_options=topts)
        return _lin(o.squeeze(0) * (1.0 / S), sh["out"])

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

        a0 = self._attn(x0, sh0, mods0[0], mods0[1], segs0, st["rope0"], topts)
        with torch.cuda.device(d1):
            a1 = self._attn(x1, sh1, mods1[0], mods1[1], segs1, st["rope1"], topts)
        r0, r1 = self.allreduce(a0, a1)
        x0 = h3._mod_gate(x0, mods0[2], r0.float() * S, segs0)
        with torch.cuda.device(d1):
            x1 = h3._mod_gate(x1, mods1[2], r1.float() * S, segs1)

        m0 = self._mlp(x0, sh0, mods0[3], mods0[4], segs0)
        with torch.cuda.device(d1):
            m1 = self._mlp(x1, sh1, mods1[3], mods1[4], segs1)
        r0, r1 = self.allreduce(m0, m1)
        x0 = h3._mod_gate(x0, mods0[5], r0.float() * S, segs0)
        with torch.cuda.device(d1):
            x1 = h3._mod_gate(x1, mods1[5], r1.float() * S, segs1)

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
        # the Comfy compiler records one device's allocations per forward; TP allocates on two
        comfy.cli_args.args.disable_comfy_compiler = True
        tp = H3TensorParallel(path, dm)
        for i in range(len(dm.blocks)):
            model.set_model_patch_replace(lambda args, extra, i=i: tp.block(i, args, extra["original_block"]),
                                          "dit", "double_block", i)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "mmh3_tp", _no_block_prefetch)
        return (model,)
