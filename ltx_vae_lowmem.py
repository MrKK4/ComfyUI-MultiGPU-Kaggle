"""Lower-VRAM, exact LTX 2.x diffusion-VAE decode, optionally spread over two GPUs.

NeighborhoodAttention3D materializes full-clip q, k, v and the attention output (4x the stage
activation; ~7 GB of the 12.8 GB peak at 896x512x121). Neighbourhood attention is local in time, so it
is run here over frame chunks with a k_t//2 halo on each side. Only the chunk's own frames are queried
(comfy_kitchen's eager na3d restricted to those rows, with the whole clip's window geometry); the halo
frames are only read as keys and values, so a chunk costs the attention of its own frames, not
2*halo more. The residual stream is updated in place, so each chunk's result is written only after
the next chunk has read its halo. The chunk length is sized to the free VRAM.

The original full-clip forward still runs whenever its q, k, v and output fit in free VRAM (it is
faster: 34 s vs 41 s for one 896x512 decode on a T4); chunks are used only when they do not, e.g. at
1 MP. ComfyUI's own decode estimate is kept, so the VAE's GPU is cleared before a decode as before.

Dual GPU (LTX_VAE_DUAL=1): at 1 MP the last decoder stage only fits 7-frame chunks on the VAE's GPU,
so each chunk recomputes 10 halo frames (2.4x the work; 144 of 163 s on a T4). The frames are split
between the VAE's GPU and the other GPU, in proportion to how much useful work each GPU's chunk size
gives. The other GPU first copies its frames (plus halo) into its own memory, before any frame is
written; its results replace that copy in place once no later chunk of its own reads them, and are
copied back after the VAE's GPU has read its halo, so the output is the same as the single-GPU path.
Copies between the GPUs go through two fixed 512 MB pinned buffers. Before a decode larger than
LTX_VAE_DUAL_MIN_VOXELS latent voxels, every model is unloaded from both GPUs.
"""
import logging
import os
import threading
import time
import traceback

import torch
import torch.nn.functional as F

import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
import comfy.model_management
import comfy.sd
import comfy_kitchen
import comfy_kitchen.backends.eager.na as ckna

logger = logging.getLogger("MultiGPU")

FORCE_CHUNK_FRAMES = None  # tests: fixed chunk length instead of the free-VRAM fit
FORCE_DUAL = False         # tests: split frames over both GPUs even when the clip fits
DUAL = os.environ.get("LTX_VAE_DUAL", "0") == "1"
_RESERVE = 512 * 1024 * 1024
_HELPER_MARGIN = 1536 * 1024 * 1024   # the other GPU: na3d workspace, allocator slack
_PRIMARY_MARGIN = 768 * 1024 * 1024   # the VAE's GPU: snapshot/copy-back staging beside its own chunks
_STAGE_BYTES = 512 * 1024 * 1024  # two fixed pinned staging buffers between the GPUs (never resized)
MIN_VOXELS = int(os.environ.get("LTX_VAE_DUAL_MIN_VOXELS", "9000"))  # latent t*h*w above which models are unloaded first
PARK_CPU_GB = float(os.environ.get("LTX_VAE_PARK_CPU_GB", "6"))  # host RAM allowed for parked DiT blocks during a decode
_LOGGED = set()
_PINNED = {}
_HELPER_WEIGHTS = {}
_orig_forward = nd.NeighborhoodAttention3D.forward


_STACK_BUDGET = 2 ** 26  # elements of stacked K/V per batched SDPA call (comfy_kitchen eager uses 2**28)


def _fit(free, frame, halo):
    # per chunk of tc frames: k, v over tc + 2*halo frames; q, the na3d output and the pending projection over tc;
    # one frame each of pre(x), the q/k/v projection and the RoPE matrices; the stacked K/V tiles of one SDPA call
    free -= 6 * _STACK_BUDGET
    return int((free / frame - 4 * halo - 4) // 5)


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


def _stage(k, shape, dtype):
    if k not in _PINNED:
        _PINNED[k] = torch.empty(_STAGE_BYTES, dtype=torch.uint8, pin_memory=True)
    n = dtype.itemsize
    for d in shape:
        n *= d
    return _PINNED[k][:n].view(dtype).view(shape)


def _chunks(lo, hi, tc, t, halo, kt):
    out = []
    for t0 in range(lo, hi, tc):
        t1 = min(t0 + tc, hi)
        a, b = max(0, t0 - halo), min(t, t1 + halo)
        # na3d clips the kernel to a short grid: widen short end chunks to one full kernel (extra context only)
        out.append((t0, t1, max(0, min(a, b - kt)), min(t, max(b, a + kt))))
    return out


def _na3d_rows(q, k, v, kernel_size, t_total, q0, k0):
    """comfy_kitchen's eager na3d restricted to query frames [q0, q0 + tq) of a t_total-frame clip.

    Keys/values cover frames [k0, k0 + tk), which must contain every queried window. Windows are those of
    the whole clip (shifted inward only at its real ends), so each output equals the full-clip na3d output
    for that frame; halo frames are only read as keys, never computed as queries. Same tiling, geometry
    grouping, masks and SDPA calls as ckna.na3d (non-causal, dilation 1, pre-scaled queries).
    """
    batch, tq, h, w, nh, hd = q.shape
    kernels = [min(kernel_size[0], t_total), min(kernel_size[1], h), min(kernel_size[2], w)]
    st, en = ckna._window_bounds(t_total, kernels[0], False)
    bt = ([x - k0 for x in st[q0:q0 + tq]], [x - k0 for x in en[q0:q0 + tq]])
    bh = ckna._window_bounds(h, kernels[1], False)
    bw = ckna._window_bounds(w, kernels[2], False)
    tile_t, tile_h, tile_w = ckna._pick_tiles((tq, h, w), kernels)
    groups = {}
    for t0 in range(0, tq, tile_t):
        t1 = min(t0 + tile_t, tq)
        rt0, rt1 = bt[0][t0], bt[1][t1 - 1]
        rel_t = (tuple(x - rt0 for x in bt[0][t0:t1]), tuple(x - rt0 for x in bt[1][t0:t1]))
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bh[0][h0], bh[1][h1 - 1]
            rel_h = (tuple(x - rh0 for x in bh[0][h0:h1]), tuple(x - rh0 for x in bh[1][h0:h1]))
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bw[0][w0], bw[1][w1 - 1]
                rel_w = (tuple(x - rw0 for x in bw[0][w0:w1]), tuple(x - rw0 for x in bw[1][w0:w1]))
                groups.setdefault((rel_t, rel_h, rel_w), []).append((
                    (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                    (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                ))
    out = torch.empty((batch, tq, h, w, nh, hd), device=q.device, dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = ckna._group_mask(rel, q.dtype, q.device)
        nq, nk = mask.shape[2], mask.shape[3]
        g_max = max(1, min(ckna.NA_KV_STACK_BUDGET, _STACK_BUDGET) // max(1, batch * nh * nk * hd * 2))
        qs0, _ = tiles[0]
        tt, th, tw = (qs0[0].stop - qs0[0].start, qs0[1].stop - qs0[1].start, qs0[2].stop - qs0[2].start)
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0:c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, scale=1.0)
            o = o.view(g, batch, nh, tt, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]
            del q_s, k_s, v_s, o
    return out


def _attend(self, src, a, b, t0, t1, t_total, tables, q_weight, k_weight, qkv_w, qkv_b, proj):
    """Attention output for frames [t0, t1), reading frames [a, b) through src(f0, f1) (pre-applied input).

    Projects and rotates one frame at a time straight into q (own frames only), k and v (own frames + halo),
    so no fused qkv output, full-chunk input copy or chunk-wide RoPE matrices are ever allocated."""
    dim, nh, hd = self.dim, self.num_heads, self.head_dim
    wq, wk, wv = qkv_w[:dim], qkv_w[dim:2 * dim], qkv_w[2 * dim:]
    bq, bk, bv = (None, None, None) if qkv_b is None else (qkv_b[:dim], qkv_b[dim:2 * dim], qkv_b[2 * dim:])
    q = k = v = None
    for f in range(a, b):
        sl = src(f, f + 1)
        if k is None:
            batch, _, h, w, _ = sl.shape
            k = torch.empty((batch, b - a, h, w, nh, hd), dtype=sl.dtype, device=sl.device)
            v = torch.empty_like(k)
            q = torch.empty((batch, t1 - t0, h, w, nh, hd), dtype=sl.dtype, device=sl.device)
            nt = h * w
        qf = F.linear(sl, wq, bq).view(batch, h, w, nh, hd)
        k[:, f - a] = F.linear(sl, wk, bk).view(batch, h, w, nh, hd)
        v[:, f - a] = F.linear(sl, wv, bv).view(batch, h, w, nh, hd)
        del sl
        freqs = nd._rope_matrices_slice(tables, f, f + 1, h, w)
        for bi in range(batch):
            comfy_kitchen.rms_rope_(qf[bi].view(1, nt, nh, hd), k[bi, f - a].view(1, nt, nh, hd), freqs, q_weight, k_weight)
        del freqs
        if t0 <= f < t1:
            q[:, f - t0] = qf
        del qf
    out = _na3d_rows(q, k, v, list(self.kernel_size), t_total, t0, a)
    del q, k, v
    return proj(out.reshape(batch, t1 - t0, h, w, self.dim))


def _norm_weights(self, dtype):
    return (self.q_norm.weight.detach() * self.scale).to(dtype), self.k_norm.weight.detach().to(dtype)


def _run_chunks(self, x, pre, res, add, chunks, tables, q_weight, k_weight):
    pending = []
    for j, (t0, t1, a, b) in enumerate(chunks):
        src = (lambda f0, f1: x[:, f0:f1]) if pre is None else (lambda f0, f1: pre(x[:, f0:f1]))
        pending.append((t0, t1, _attend(self, src, a, b, t0, t1, x.shape[1], tables, q_weight, k_weight,
                                        self.qkv.weight, self.qkv.bias, self.proj)))
        # in place: a result is written only once no later chunk reads those frames
        next_read = min((c[2] for c in chunks[j + 1:]), default=x.shape[1])
        while pending and pending[0][1] <= next_read:
            p0, p1, py = pending.pop(0)
            if add:
                res[:, p0:p1] += py
            else:
                res[:, p0:p1] = py


def _log_oom(dev, what):
    logger.warning("[MultiGPU] LTX VAE dual decode out of memory on %s (%s): allocated %.2f GB, reserved %.2f GB, free %.2f GB",
                   dev, what, torch.cuda.memory_allocated(dev) / 2**30, torch.cuda.memory_reserved(dev) / 2**30,
                   torch.cuda.mem_get_info(dev)[0] / 2**30)


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
    free_h = comfy.model_management.get_free_memory(helper) - _RESERVE - _HELPER_MARGIN
    if FORCE_CHUNK_FRAMES is not None or FORCE_DUAL:
        tc_h, s = tc_p, t // 2
    else:
        free_p = comfy.model_management.get_free_memory(x.device) - _RESERVE - _PRIMARY_MARGIN
        tc_p = max(_fit(free_p, frame, halo), halo, 1)
        tc_h, s = _fit(free_h, frame, halo), t
        for _ in range(3):  # the helper also holds a snapshot of its frames (+ halo), which shrinks its chunks
            if tc_h < max(halo, 1):
                return False
            r_p, r_h = tc_p / (tc_p + 2 * halo), tc_h / (tc_h + 2 * halo)
            s = int(round(t * r_p / (r_p + r_h)))
            tc_h = _fit(free_h - (t - s + kt) * frame, frame, halo)
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

    # snapshot of every frame the helper reads, taken before any frame of x is written; it lives on the helper
    # and later holds the helper's results (written in place once no later helper chunk reads those frames)
    ha = min(c[2] for c in h_chunks)
    n = max(1, _STAGE_BYTES // frame)
    ps = torch.cuda.current_stream(x.device)
    hs = torch.cuda.Stream(helper)
    with torch.cuda.device(helper), torch.cuda.stream(hs):
        snap = torch.empty((batch, t - ha, h, w, self.dim), dtype=x.dtype, device=helper)
    freed = [None, None]
    for i, f0 in enumerate(range(ha, t, n)):
        f1, k = min(f0 + n, t), i % 2
        buf = _stage(k, (batch, f1 - f0, h, w, self.dim), x.dtype)
        if freed[k] is not None:
            ps.wait_event(freed[k])
        piece = x[:, f0:f1] if pre is None else pre(x[:, f0:f1])
        buf.copy_(piece, non_blocking=True)
        del piece
        ready = torch.cuda.Event()
        ready.record(ps)
        hs.wait_event(ready)
        with torch.cuda.device(helper), torch.cuda.stream(hs):
            snap[:, f0 - ha:f1 - ha].copy_(buf, non_blocking=True)
        freed[k] = torch.cuda.Event()
        freed[k].record(hs)

    state = {}
    weights = _helper_weights(self, helper, x.dtype)

    def work():
        try:
            with torch.inference_mode(), torch.cuda.device(helper), torch.cuda.stream(hs):
                qkv_w, qkv_b, proj_w, proj_b, q_weight, k_weight = weights
                inv = tuple(nd.rope_inv_freqs(d, self.rope_base, device=helper) for d in self.rope_split)
                tables = nd._rope_tables((t, h, w), inv, helper)
                pending = []
                for j, (t0, t1, a, b) in enumerate(h_chunks):
                    pending.append((t0, t1, _attend(self, lambda f0, f1: snap[:, f0 - ha:f1 - ha], a, b, t0, t1, t, tables,
                                                    q_weight, k_weight, qkv_w, qkv_b, lambda z: F.linear(z, proj_w, proj_b))))
                    next_read = min((c[2] for c in h_chunks[j + 1:]), default=t)
                    while pending and pending[0][1] <= next_read:
                        p0, p1, py = pending.pop(0)
                        snap[:, p0 - ha:p1 - ha] = py
                del pending, tables
                hs.synchronize()
        except BaseException as e:  # re-raised on the caller's thread
            if isinstance(e, torch.OutOfMemoryError):
                _log_oom(helper, "%d frames in %d-frame chunks" % (t - s, tc_h))
            state["error"] = e

    thread = threading.Thread(target=work, name="ltx-vae-helper")
    thread.start()
    try:
        inv = tuple(nd.rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
        tables = nd._rope_tables((t, h, w), inv, x.device)
        q_weight, k_weight = _norm_weights(self, x.dtype)
        _run_chunks(self, x, pre, res, add, p_chunks, tables, q_weight, k_weight)
    except torch.OutOfMemoryError:
        _log_oom(x.device, "%d frames in %d-frame chunks" % (s, tc_p))
        raise
    finally:
        thread.join()
    if "error" in state:
        hs.synchronize()
        del snap
        raise state["error"]

    # the helper's frames are written only now: the last primary chunk read their halo
    used = [None, None]
    for i, f0 in enumerate(range(s, t, n)):
        f1, k = min(f0 + n, t), i % 2
        buf = _stage(k, (batch, f1 - f0, h, w, self.dim), x.dtype)
        if used[k] is not None:
            hs.wait_event(used[k])
        with torch.cuda.device(helper), torch.cuda.stream(hs):
            buf.copy_(snap[:, f0 - ha:f1 - ha], non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(hs)
        ps.wait_event(ready)
        py = buf.to(x.device, non_blocking=True)
        if add:
            res[:, f0:f1] += py
        else:
            res[:, f0:f1] = py
        del py
        used[k] = torch.cuda.Event()
        used[k].record(ps)
    ps.synchronize()
    hs.synchronize()
    del snap
    return True


def _forward_diff_step(self, context, x_t, t):
    """NADiffusionDecoder.forward_diff_step with the closing norm_out + conv_out run over frame chunks: the
    original materializes norm_out(x) for the whole clip next to x and the context volume (3 x 4.2 GB at
    1344x768x121). Both are per-token, so the output is the same."""
    x = nd.patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
    x = self.conv_in_x_t(x.permute(0, 2, 3, 4, 1))
    t_emb = self.t_embedder(self.timestep_scale_multiplier * t, dtype=x.dtype)
    modulation = self.shared_adaln(t_emb)
    for block in self.diff_blocks:
        x = block(x, context, modulation)
    out = torch.empty(x.shape[:-1] + (self.conv_out.out_features,), dtype=x.dtype, device=x.device)
    chunk = max(1, nd.MLP_TOKEN_CHUNK // max(x.shape[2] * x.shape[3], 1))
    for t0 in range(0, x.shape[1], chunk):
        out[:, t0:t0 + chunk] = self.conv_out(self.norm_out(x[:, t0:t0 + chunk]))
    del x
    return nd.unpatchify(out.permute(0, 4, 1, 2, 3), patch_size_hw=self.patch_size, patch_size_t=1)


def forward(self, x, pre=None, add_to=None):
    batch, t, h, w, _ = x.shape
    tc = _chunk_frames(self, x)
    _LAST_NA.update(shape=tuple(x.shape[:4]), dim=x.shape[-1], kernel=tuple(self.kernel_size), tc=min(tc, t),
                    free_gb=round(comfy.model_management.get_free_memory(x.device) / 2**30, 2))
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


def _gpu_free():
    return " ".join("cuda:%d %.1f/%.1f GB free" % ((d,) + tuple(v / 2**30 for v in torch.cuda.mem_get_info(d)))
                    for d in range(torch.cuda.device_count()))


def _decode_need(voxels):
    # last decoder stage keeps the residual stream and the context volume (512 tokens x 256 ch fp16 per latent
    # voxel each) for the whole clip; plus the VAE weights, pixel noise and chunk workspace
    return 2 * voxels * 512 * 256 * 2 + int(4.5 * 1024 ** 3)


def _park_dit(device, need):
    """Move DisTorch donor blocks off ``device`` until ``need`` bytes are free there: first into another GPU's
    spare memory, then (up to PARK_CPU_GB) into host RAM. They run with comfy_cast_weights, so where they live
    between sampling runs does not matter; _unpark puts them back. Returns [(module, device, bytes, target)]."""
    free = torch.cuda.mem_get_info(device)[0]
    if free >= need:
        return []
    blocks = []
    for loaded in list(comfy.model_management.current_loaded_models):
        patcher = getattr(loaded, "model", None)
        model = getattr(patcher, "model", None)
        if model is None or not hasattr(patcher, "_distorch_cached_assignments"):
            continue
        for m in model.modules():
            if next(m.children(), None) is not None:
                continue
            tensors = list(m.parameters(recurse=False)) + list(m.buffers(recurse=False))
            if tensors and all(x.device == device for x in tensors):
                blocks.append((m, sum(x.numel() * x.element_size() for x in tensors)))
    others = [torch.device("cuda", d) for d in range(torch.cuda.device_count()) if d != device.index]
    spare = {o: torch.cuda.mem_get_info(o)[0] - 768 * 1024 ** 2 for o in others}
    cpu_left = PARK_CPU_GB * 1024 ** 3
    to_free, parked = need - free, []
    for m, size in sorted(blocks, key=lambda b: -b[1]):
        if to_free <= 0:
            break
        target = next((o for o in others if spare[o] > size), None)
        if target is not None:
            spare[target] -= size
        elif cpu_left >= size:
            target, cpu_left = torch.device("cpu"), cpu_left - size
        else:
            continue
        m.to(target)
        parked.append((m, device, size, target))
        to_free -= size
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    if parked:
        moved = {}
        for _, _, size, target in parked:
            moved[str(target)] = moved.get(str(target), 0) + size
        logger.info("[MultiGPU] LTX VAE decode: parked %.1f GB of DiT blocks from %s (%s), now %s",
                    sum(moved.values()) / 2**30, device, ", ".join("%.1f GB to %s" % (v / 2**30, k) for k, v in moved.items()),
                    _gpu_free())
    return parked


def _models_on_gpus():
    rows = []
    for lm in comfy.model_management.current_loaded_models:
        try:
            mp = lm.model
            rows.append("%s@%s%s %.1f GB loaded" % (type(getattr(mp, "model", mp)).__name__, lm.device,
                                                   " dynamic" if mp.is_dynamic() else "", mp.loaded_size() / 2**30))
        except Exception as e:  # dead weakref etc.
            rows.append("? (%s)" % type(e).__name__)
    torch_gb = " ".join("cuda:%d torch %.1f/%.1f GB alloc/reserved" % (
        d, torch.cuda.memory_allocated(d) / 2**30, torch.cuda.memory_reserved(d) / 2**30) for d in range(torch.cuda.device_count()))
    return "; ".join(rows) + " | " + torch_gb


def _free_vae_gpu(device, need):
    """Dynamic VRAM skips dynamic models when another dynamic model (the VAE) loads, so a text encoder staged on
    the VAE's GPU stays there; the decode's own activations cannot evict it. Free the VAE's GPU explicitly
    (non-dynamic free: only models loaded on this device, e.g. the text encoder; the DiT lives on another)."""
    if torch.cuda.mem_get_info(device)[0] >= need:
        return
    unloaded = comfy.model_management.free_memory(need, device)
    comfy.model_management.soft_empty_cache()
    logger.info("[MultiGPU] LTX VAE decode: freed %s for a %.1f GB decode, unloaded %d model(s): %s; now %s",
                device, need / 2**30, len(unloaded),
                ", ".join(type(getattr(m.model, "model", m.model)).__name__ for m in unloaded if m.model is not None) or "-",
                _gpu_free())


def _unpark(parked):
    if not parked:
        return
    for d in range(torch.cuda.device_count()):
        with torch.cuda.device(d):
            torch.cuda.empty_cache()
    for m, device, _, _ in parked:
        m.to(device)
    logger.info("[MultiGPU] LTX VAE decode: restored %.1f GB of DiT blocks to %s",
                sum(p[2] for p in parked) / 2**30, parked[0][1])


def _dual_decode(orig):
    def run(self, samples, *args, **kwargs):
        if not isinstance(getattr(self, "first_stage_model", None), nd.CausalDiffusionVAE):
            return orig(self, samples, *args, **kwargs)
        voxels = samples.shape[-3] * samples.shape[-2] * samples.shape[-1]
        helper = _helper_device(torch.device(self.device)) if DUAL else None
        if voxels > MIN_VOXELS:
            logger.info("[MultiGPU] LTX VAE decode (%d latent voxels) on %s: %s | models: %s", voxels, self.device,
                        _gpu_free(), _models_on_gpus())
        if helper is None:
            device = torch.device(self.device)
            parked = []
            if voxels > MIN_VOXELS and device.type == "cuda":
                _free_vae_gpu(device, _decode_need(voxels))
                parked = _park_dit(device, _decode_need(voxels))
            try:
                return orig(self, samples, *args, **kwargs)
            finally:
                _unpark(parked)
        if voxels > MIN_VOXELS:
            # a large dual decode needs most of both GPUs: unload every model (the next job reloads them)
            for d in range(torch.cuda.device_count()):
                comfy.model_management.free_memory(1e30, torch.device("cuda", d))
            comfy.model_management.soft_empty_cache()
            for d in range(torch.cuda.device_count()):
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
            logger.info("[MultiGPU] LTX VAE dual decode: unloaded models, now %s", _gpu_free())
        try:
            return orig(self, samples, *args, **kwargs)
        finally:
            _HELPER_WEIGHTS.clear()
            with torch.cuda.device(helper):
                torch.cuda.empty_cache()
    return run


_LAST_NA = {}


def _vae_core_decode(orig):
    """CausalDiffusionVAE.decode with diagnostics: ComfyUI catches an out-of-memory error here and silently
    retries with tiled decoding, so log where it ran out (call stack, last attention layer and chunk, memory
    per GPU) before re-raising; on success log the time and peak memory."""
    def run(self, x):
        dev = x.device
        if dev.type != "cuda":
            return orig(self, x)
        _LAST_NA.clear()
        torch.cuda.reset_peak_memory_stats(dev)
        start, t0 = torch.cuda.memory_allocated(dev), time.perf_counter()
        try:
            out = orig(self, x)
        except torch.OutOfMemoryError as e:
            frames = traceback.extract_tb(e.__traceback__)[-6:]
            where = " <- ".join("%s:%d %s" % (f.filename.rsplit("/", 1)[-1], f.lineno, f.name) for f in reversed(frames))
            logger.warning("[MultiGPU] LTX VAE OOM in decode of latent %s after %.1fs: %s | last attention %s | peak %.2f GB "
                           "above start (%.2f GB) | %s | torch %.2f/%.2f GB alloc/reserved | at: %s",
                           tuple(x.shape), time.perf_counter() - t0, str(e).splitlines()[0][:160], _LAST_NA or "-",
                           (torch.cuda.max_memory_allocated(dev) - start) / 2**30, start / 2**30, _gpu_free(),
                           torch.cuda.memory_allocated(dev) / 2**30, torch.cuda.memory_reserved(dev) / 2**30, where)
            raise
        torch.cuda.synchronize(dev)
        logger.info("[MultiGPU] LTX VAE decode OK: latent %s in %.1fs, peak %.2f GB above start (%.2f GB) on %s",
                    tuple(x.shape), time.perf_counter() - t0, (torch.cuda.max_memory_allocated(dev) - start) / 2**30,
                    start / 2**30, dev)
        return out
    return run


def patch_ltx_vae_lowmem():
    if getattr(nd.NeighborhoodAttention3D.forward, "_mgpu_lowmem", False):
        return
    forward._mgpu_lowmem = True
    nd.NeighborhoodAttention3D.forward = forward
    nd.NADiffusionDecoder.forward_diff_step = _forward_diff_step
    nd.CausalDiffusionVAE.decode = _vae_core_decode(nd.CausalDiffusionVAE.decode)
    comfy.sd.VAE.decode = _dual_decode(comfy.sd.VAE.decode)
    comfy.sd.VAE.decode_tiled = _dual_decode(comfy.sd.VAE.decode_tiled)
    logger.info("[MultiGPU] LTX diffusion VAE: frame-chunked neighbourhood attention when the full clip does not fit%s",
                " (dual GPU on)" if DUAL else "")
