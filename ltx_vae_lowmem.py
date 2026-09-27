"""Lower-VRAM, exact LTX 2.x diffusion-VAE decode.

NeighborhoodAttention3D materializes full-clip q, k, v and the attention output (4x the stage
activation; ~7 GB of the 12.8 GB peak at 896x512x121). Neighbourhood attention is local in time, so it
is run here over frame chunks with a k_t//2 halo on each side: a query keeps the same window as in the
full clip (NATTEN shifts windows inward only at the real grid ends, and chunks touching an end keep
it), so the output is the same. The residual stream is updated in place, so each chunk's result is
written only after the next chunk has read its halo. The chunk length is sized to the free VRAM; when
the whole clip fits, the original forward runs unchanged.

The VAE's decode estimate (1700 x 512 x latent voxels) makes ComfyUI evict ~12.5 GB before an
896x512 decode; it is replaced by the full-clip x + context tensors plus room for small chunks.
"""
import logging

import torch

import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
import comfy.model_management
import comfy.sd
import comfy_kitchen

logger = logging.getLogger("MultiGPU")

FORCE_CHUNK_FRAMES = None  # tests: fixed chunk length instead of the free-VRAM fit
_RESERVE = 512 * 1024 * 1024
_orig_forward = nd.NeighborhoodAttention3D.forward


def _chunk_frames(self, x):
    t, h, w = x.shape[1:4]
    halo = self.kernel_size[0] // 2
    if FORCE_CHUNK_FRAMES is not None:
        return max(FORCE_CHUNK_FRAMES, halo, 1)
    frame = x.shape[0] * h * w * self.dim * x.element_size()
    free = comfy.model_management.get_free_memory(x.device) - _RESERVE
    # per chunk of tc frames: pre(x) + qkv output (3) + q, k, v + na3d out over tc + 2*halo, plus the pending projection (tc)
    tc = int((free / frame - 7 * 2 * halo) // 8)
    return max(tc, halo, 1)


def forward(self, x, pre=None, add_to=None):
    batch, t, h, w, _ = x.shape
    tc = _chunk_frames(self, x)
    if tc >= t:
        return _orig_forward(self, x, pre, add_to)
    halo = self.kernel_size[0] // 2
    inv_freqs = tuple(nd.rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
    tables = nd._rope_tables((t, h, w), inv_freqs, x.device)
    q_weight = (self.q_norm.weight.detach() * self.scale).to(x.dtype)
    k_weight = self.k_norm.weight.detach().to(x.dtype)
    res = add_to if add_to is not None else torch.empty_like(x)
    kt = self.kernel_size[0]
    chunks = []
    for t0 in range(0, t, tc):
        t1 = min(t0 + tc, t)
        a, b = max(0, t0 - halo), min(t, t1 + halo)
        # na3d clips the kernel to a short grid: widen short end chunks to one full kernel (extra context only)
        chunks.append((t0, t1, max(0, min(a, b - kt)), min(t, max(b, a + kt))))
    pending = []
    for j, (t0, t1, a, b) in enumerate(chunks):
        sl = x[:, a:b] if pre is None else pre(x[:, a:b])
        cshape = (batch, b - a, h, w, self.num_heads, self.head_dim)
        q, k, v = (c.reshape(cshape) for c in self.qkv(sl).chunk(3, dim=-1))
        del sl
        freqs = nd._rope_matrices_slice(tables, a, b, h, w)
        nt = (b - a) * h * w
        for bi in range(batch):
            comfy_kitchen.rms_rope_(q[bi].view(1, nt, self.num_heads, self.head_dim),
                                    k[bi].view(1, nt, self.num_heads, self.head_dim), freqs, q_weight, k_weight)
        out = comfy_kitchen.na3d(q, k, v, list(self.kernel_size), None, 1.0)
        del q, k, v
        pending.append((t0, t1, self.proj(out[:, t0 - a:t1 - a].reshape(batch, t1 - t0, h, w, self.dim))))
        del out
        # in place: a result is written only once no later chunk reads those frames
        next_read = min((c[2] for c in chunks[j + 1:]), default=t)
        while pending and pending[0][1] <= next_read:
            p0, p1, py = pending.pop(0)
            if add_to is not None:
                res[:, p0:p1] += py
            else:
                res[:, p0:p1] = py
    return res


_orig_vae_init = comfy.sd.VAE.__init__


def _vae_init(self, *args, **kwargs):
    _orig_vae_init(self, *args, **kwargs)
    if isinstance(getattr(self, "first_stage_model", None), nd.CausalDiffusionVAE):
        # full-clip stage-5 x + context: 512 tokens x 256 ch per latent voxel each, + chunk/workspace headroom
        self.memory_used_decode = lambda shape, dtype: (2 * 512 * 256 * shape[2] * shape[3] * shape[4]) * comfy.model_management.dtype_size(dtype) + 2 * 1024 ** 3


def patch_ltx_vae_lowmem():
    if getattr(nd.NeighborhoodAttention3D.forward, "_mgpu_lowmem", False):
        return
    forward._mgpu_lowmem = True
    nd.NeighborhoodAttention3D.forward = forward
    comfy.sd.VAE.__init__ = _vae_init
    logger.info("[MultiGPU] LTX diffusion VAE: frame-chunked neighbourhood attention + lower decode estimate")
