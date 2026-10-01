"""Lower-VRAM, exact LTX 2.x diffusion-VAE decode, optionally spread over two GPUs.

NeighborhoodAttention3D materializes full-clip q, k, v and the attention output (4x the stage
activation; ~7 GB of the 12.8 GB peak at 896x512x121). Neighbourhood attention is local in time, so it
is run here over frame chunks with a k_t//2 halo on each side: a query keeps the same window as in the
full clip (NATTEN shifts windows inward only at the real grid ends, and chunks touching an end keep
it), so the output is the same. The residual stream is updated in place, so each chunk's result is
written only after the next chunk has read its halo. The chunk length is sized to the free VRAM.

The original full-clip forward still runs whenever its q, k, v and output fit in free VRAM (it is
faster: 34 s vs 41 s for one 896x512 decode on a T4); chunks are used only when they do not, e.g. at
1 MP. ComfyUI's own decode estimate is kept, so the VAE's GPU is cleared before a decode as before.

Dual GPU (LTX_VAE_DUAL=1): at 1 MP the last decoder stage only fits 7-frame chunks on the VAE's GPU,
so each chunk recomputes 10 halo frames (2.4x the work; 144 of 163 s on a T4). The frames are split
between the VAE's GPU and the other GPU, in proportion to how much useful work each GPU's chunk size
gives. The other GPU reads its frames (plus halo) from a pinned-host snapshot taken before any frame
is written and keeps its results on its own memory until the VAE's GPU has finished reading, so the
output is the same as the single-GPU path. Before a decode, LTX_VAE_DUAL_GB of the other GPU is freed.
"""
import logging
import os
import threading

import torch
import torch.nn.functional as F

import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
import comfy.model_management
import comfy.sd
import comfy_kitchen

logger = logging.getLogger("MultiGPU")

FORCE_CHUNK_FRAMES = None  # tests: fixed chunk length instead of the free-VRAM fit
FORCE_DUAL = False         # tests: split frames over both GPUs even when the clip fits
DUAL = os.environ.get("LTX_VAE_DUAL", "0") == "1"
DUAL_GB = float(os.environ.get("LTX_VAE_DUAL_GB", "10"))
_RESERVE = 512 * 1024 * 1024
_STAGE_FRAMES = 8  # frames per pinned staging copy
_LOGGED = set()
_PINNED = {}
_HELPER_WEIGHTS = {}
_orig_forward = nd.NeighborhoodAttention3D.forward


def _fit(free, frame, halo):
    # per chunk of tc frames: pre(x) + qkv output (3) + q, k, v + na3d out over tc + 2*halo, plus the pending projection (tc)
    return int((free / frame - 7 * 2 * halo) // 8)


def _chunk_frames(self, x):
    t, h, w = x.shape[1:4]
    halo = self.kernel_size[0] // 2
    if FORCE_CHUNK_FRAMES is not None:
        return max(FORCE_CHUNK_FRAMES, halo, 1)
    frame = x.shape[0] * h * w * self.dim * x.element_size()
    free = comfy.model_management.get_free_memory(x.device) - _RESERVE
    # original forward: full-clip q, k, v, out + one qkv slice of at most 2**25 elements per tensor
    if free >= 4 * t * frame + 4 * (2 ** 25) * x.element_size():
        return t
    return max(_fit(free, frame, halo), halo, 1)


def _helper_device(device):
    if not (DUAL or FORCE_DUAL) or device.type != "cuda" or torch.cuda.device_count() < 2:
        return None
    return torch.device("cuda", (device.index + 1) % torch.cuda.device_count())


def _pinned(name, shape, dtype):
    n = 1
    for d in shape:
        n *= d
    buf = _PINNED.get(name)
    if buf is None or buf.numel() < n or buf.dtype != dtype:
        _PINNED.pop(name, None)
        buf = torch.empty(n, dtype=dtype, pin_memory=True)
        _PINNED[name] = buf
    return buf[:n].view(shape)


def _chunks(lo, hi, tc, t, halo, kt):
    out = []
    for t0 in range(lo, hi, tc):
        t1 = min(t0 + tc, hi)
        a, b = max(0, t0 - halo), min(t, t1 + halo)
        # na3d clips the kernel to a short grid: widen short end chunks to one full kernel (extra context only)
        out.append((t0, t1, max(0, min(a, b - kt)), min(t, max(b, a + kt))))
    return out


def _attend(self, sl, a, b, t0, t1, tables, q_weight, k_weight, qkv, proj):
    batch, _, h, w, _ = sl.shape
    cshape = (batch, b - a, h, w, self.num_heads, self.head_dim)
    q, k, v = (c.reshape(cshape) for c in qkv(sl).chunk(3, dim=-1))
    freqs = nd._rope_matrices_slice(tables, a, b, h, w)
    nt = (b - a) * h * w
    for bi in range(batch):
        comfy_kitchen.rms_rope_(q[bi].view(1, nt, self.num_heads, self.head_dim),
                                k[bi].view(1, nt, self.num_heads, self.head_dim), freqs, q_weight, k_weight)
    out = comfy_kitchen.na3d(q, k, v, list(self.kernel_size), None, 1.0)
    del q, k, v
    return proj(out[:, t0 - a:t1 - a].reshape(batch, t1 - t0, h, w, self.dim))


def _norm_weights(self, dtype):
    return (self.q_norm.weight.detach() * self.scale).to(dtype), self.k_norm.weight.detach().to(dtype)


def _run_chunks(self, x, pre, res, add, chunks, tables, q_weight, k_weight):
    pending = []
    for j, (t0, t1, a, b) in enumerate(chunks):
        sl = x[:, a:b] if pre is None else pre(x[:, a:b])
        pending.append((t0, t1, _attend(self, sl, a, b, t0, t1, tables, q_weight, k_weight, self.qkv, self.proj)))
        del sl
        # in place: a result is written only once no later chunk reads those frames
        next_read = min((c[2] for c in chunks[j + 1:]), default=x.shape[1])
        while pending and pending[0][1] <= next_read:
            p0, p1, py = pending.pop(0)
            if add:
                res[:, p0:p1] += py
            else:
                res[:, p0:p1] = py


def _helper_weights(self, dev, dtype):
    key = (id(self), dev, self.qkv.weight.data_ptr())
    if key not in _HELPER_WEIGHTS:
        q_weight, k_weight = _norm_weights(self, dtype)
        _HELPER_WEIGHTS[key] = tuple(None if p is None else p.detach().to(dev) for p in (
            self.qkv.weight, self.qkv.bias, self.proj.weight, self.proj.bias, q_weight, k_weight))
    return _HELPER_WEIGHTS[key]


def _dual(self, x, pre, res, add, helper, tc_p):
    batch, t, h, w, _ = x.shape
    halo, kt = self.kernel_size[0] // 2, self.kernel_size[0]
    frame = batch * h * w * self.dim * x.element_size()
    free_h = comfy.model_management.get_free_memory(helper) - _RESERVE
    if FORCE_CHUNK_FRAMES is not None or FORCE_DUAL:
        tc_h, s = tc_p, t // 2
    else:
        tc_h, s = _fit(free_h, frame, halo), t
        for _ in range(3):  # the helper also holds its output frames, which shrinks its chunks
            if tc_h < max(halo, 1):
                return False
            r_p, r_h = tc_p / (tc_p + 2 * halo), tc_h / (tc_h + 2 * halo)
            s = int(round(t * r_p / (r_p + r_h)))
            tc_h = _fit(free_h - (t - s) * frame, frame, halo)
    if tc_h < max(halo, 1) or not 0 < s < t:
        return False
    tc_h = max(tc_h, halo, 1)
    p_chunks = _chunks(0, s, tc_p, t, halo, kt)
    h_chunks = _chunks(s, t, tc_h, t, halo, kt)
    key = ("dual", tuple(x.shape), tc_p, tc_h, s)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info("[MultiGPU] LTX VAE attention %s: %d frames on %s (%d-frame chunks), %d on %s (%d-frame chunks)",
                    tuple(x.shape[:4]), s, x.device, tc_p, t - s, helper, tc_h)

    # snapshot of every frame the helper reads, taken before any frame of x is written
    ha = min(c[2] for c in h_chunks)
    host_in = _pinned("in", (batch, t - ha, h, w, self.dim), x.dtype)
    for f0 in range(ha, t, _STAGE_FRAMES):
        f1 = min(f0 + _STAGE_FRAMES, t)
        piece = x[:, f0:f1] if pre is None else pre(x[:, f0:f1])
        host_in[:, f0 - ha:f1 - ha].copy_(piece, non_blocking=True)
        del piece
    snapshot = torch.cuda.Event()
    snapshot.record(torch.cuda.current_stream(x.device))

    state = {}
    weights = _helper_weights(self, helper, x.dtype)

    def work():
        try:
            with torch.inference_mode(), torch.cuda.device(helper), torch.cuda.stream(torch.cuda.Stream(helper)):
                qkv_w, qkv_b, proj_w, proj_b, q_weight, k_weight = weights
                inv = tuple(nd.rope_inv_freqs(d, self.rope_base, device=helper) for d in self.rope_split)
                tables = nd._rope_tables((t, h, w), inv, helper)
                out = torch.empty((batch, t - s, h, w, self.dim), dtype=x.dtype, device=helper)
                snapshot.synchronize()
                for t0, t1, a, b in h_chunks:
                    sl = host_in[:, a - ha:b - ha].to(helper, non_blocking=True)
                    out[:, t0 - s:t1 - s] = _attend(self, sl, a, b, t0, t1, tables, q_weight, k_weight,
                                                    lambda z: F.linear(z, qkv_w, qkv_b), lambda z: F.linear(z, proj_w, proj_b))
                    del sl
                torch.cuda.current_stream(helper).synchronize()
                state["out"] = out
        except BaseException as e:  # re-raised on the caller's thread
            state["error"] = e

    thread = threading.Thread(target=work, name="ltx-vae-helper")
    thread.start()
    try:
        inv = tuple(nd.rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
        tables = nd._rope_tables((t, h, w), inv, x.device)
        q_weight, k_weight = _norm_weights(self, x.dtype)
        _run_chunks(self, x, pre, res, add, p_chunks, tables, q_weight, k_weight)
    finally:
        thread.join()
    if "error" in state:
        raise state["error"]

    # the helper's frames are written only now: the last primary chunk read their halo
    out = state.pop("out")
    stage = _pinned("out", (batch, _STAGE_FRAMES, h, w, self.dim), x.dtype)
    for f0 in range(0, t - s, _STAGE_FRAMES):
        f1 = min(f0 + _STAGE_FRAMES, t - s)
        part = stage[:, :f1 - f0]
        part.copy_(out[:, f0:f1])  # synchronous: the staging buffer is reused next iteration
        py = part.to(x.device)
        if add:
            res[:, s + f0:s + f1] += py
        else:
            res[:, s + f0:s + f1] = py
        del py
    del out
    return True


def forward(self, x, pre=None, add_to=None):
    batch, t, h, w, _ = x.shape
    tc = _chunk_frames(self, x)
    helper = _helper_device(x.device)
    if tc >= t and not FORCE_DUAL:
        return _orig_forward(self, x, pre, add_to)
    res = add_to if add_to is not None else torch.empty_like(x)
    if helper is not None and _dual(self, x, pre, res, add_to is not None, helper, min(tc, t)):
        return res
    key = (tuple(x.shape), tc)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info("[MultiGPU] LTX VAE attention %s: %d-frame chunks (full clip does not fit)", tuple(x.shape[:4]), tc)
    halo = self.kernel_size[0] // 2
    inv = tuple(nd.rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
    tables = nd._rope_tables((t, h, w), inv, x.device)
    q_weight, k_weight = _norm_weights(self, x.dtype)
    _run_chunks(self, x, pre, res, add_to is not None, _chunks(0, t, tc, t, halo, self.kernel_size[0]),
                tables, q_weight, k_weight)
    return res


def _dual_decode(orig):
    def run(self, *args, **kwargs):
        helper = None
        if DUAL and isinstance(getattr(self, "first_stage_model", None), nd.CausalDiffusionVAE):
            helper = _helper_device(torch.device(self.device))
        if helper is None:
            return orig(self, *args, **kwargs)
        comfy.model_management.free_memory(DUAL_GB * 1024 ** 3, helper)
        try:
            return orig(self, *args, **kwargs)
        finally:
            _PINNED.clear()
            _HELPER_WEIGHTS.clear()
            with torch.cuda.device(helper):
                torch.cuda.empty_cache()
    return run


def patch_ltx_vae_lowmem():
    if getattr(nd.NeighborhoodAttention3D.forward, "_mgpu_lowmem", False):
        return
    forward._mgpu_lowmem = True
    nd.NeighborhoodAttention3D.forward = forward
    comfy.sd.VAE.decode = _dual_decode(comfy.sd.VAE.decode)
    comfy.sd.VAE.decode_tiled = _dual_decode(comfy.sd.VAE.decode_tiled)
    logger.info("[MultiGPU] LTX diffusion VAE: frame-chunked neighbourhood attention when the full clip does not fit%s",
                " (dual GPU on)" if DUAL else "")
