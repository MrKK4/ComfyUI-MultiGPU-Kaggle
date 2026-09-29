"""Attention backends for the H3 tensor-parallel path on Turing (T4).

The measured starting point: the kitchen kernel (`qk_int_sv_i8`) runs H3's attention at
~17.5e12 MAC/s (35 TOPS) — 27% of the T4's int8 tensor-core peak (65e12 MAC/s), i.e. about
the fp16 peak. PyTorch's own SDPA is 2.6x slower again, so the wrong move is to swap int8 for
fp16: at 141 frames (28.5k packed tokens, 28 heads per GPU) attention is the single largest
kernel class in the step, and the only thing that can beat the kitchen kernel is another
*int8* kernel with a better schedule — which is what `int8` below is.

    python turing_attention.py --selftest                    # H3 TP shapes on this box
    python turing_attention.py --selftest --n 28500 --heads 28

    MMH3_TP_ATTN=triton|int8|sdpa|math   # opt in; unset keeps ComfyUI's path (default)

Backends:
  comfy   comfy.ldm.modules.attention.optimized_attention -- what the TP path uses today
          (the selftest prints which kernels this actually ran, so the baseline is honest)
  int8    Triton int8 flash attention: per-row int8 Q/K, per-channel int8 V, P at a fixed
          1/127 scale so the online-softmax accumulator stays consistent
  triton  fp16 flash-attention style forward in Triton (ships with torch, no build step)
  sdpa    torch SDPA on plain (B,H,S,D) tensors, efficient backend pinned
  math    chunked fp32 reference (memory bounded; also the correctness baseline)

`--selftest` reports ms/call, the max-abs error against the fp32 reference, the max-abs
difference against the kernel it would replace, and the projected s/step of attention for a
50-block DiT. It also prints a roofline table for each whole-config sweep: TOPS and % of the int8
peak, GB/s of K/V traffic and % of HBM bandwidth, and the tile-size sensitivity that says which of
the two binds. Measured on the kitchen kernel's own numbers, neither does: 27% of compute peak and
42% of bandwidth peak at BLOCK_M=128, and a 4x tile cut that removes 4x the traffic buys only
~10-20% -- so the remaining distance is scheduling, not a wall. Nothing here changes behaviour unless MMH3_TP_ATTN names a backend that passes
its smoke test on this GPU; otherwise the ComfyUI call is used exactly as before.
"""
import argparse
import logging
import math
import os
import sys
import time

logger = logging.getLogger("MultiGPU")

REPORT = "turing_attention.txt"
DEFAULT_N = int(os.environ.get("MMH3_TP_DIAG_ROWS", "28500"))     # measured packed length, 141 frames
DEFAULT_HEADS = int(os.environ.get("MMH3_TP_DIAG_HEADS", "28"))   # 56 heads split over 2 GPUs
DEFAULT_DIM = 128
HEAD_DIM = DEFAULT_DIM


def _torch():
    import torch
    return torch


def _device(requested=None):
    torch = _torch()
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _scale_for(q, scale=None):
    return (1.0 / math.sqrt(q.shape[-1])) if scale is None else scale


def _like(out_shape, dtype, device):
    torch = _torch()
    return torch.empty(out_shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# backends: signature (q, k, v, scale=None) -> (B, H, S, D)
# ---------------------------------------------------------------------------

def reference_attention(q, k, v, scale=None, block=256):
    """fp32 math reference, chunked over queries: n=15k would otherwise need ~25 GB of scores."""
    torch = _torch()
    scale = _scale_for(q, scale)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    kf = k.float().transpose(-1, -2).contiguous()
    vf = v.float()
    length = q.shape[-2]
    for start in range(0, length, block):
        stop = min(start + block, length)
        scores = torch.matmul(q[..., start:stop, :].float(), kf) * scale
        scores = scores - scores.amax(dim=-1, keepdim=True)
        probs = torch.softmax(scores, dim=-1)
        out[..., start:stop, :] = torch.matmul(probs, vf)
    return out.to(q.dtype)


def sdpa_attention(q, k, v, scale=None):
    """Plain SDPA with the memory-efficient (cutlass/xformers) backend pinned on Turing."""
    torch = _torch()
    scale = _scale_for(q, scale)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    except Exception:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def sdpa_math_attention(q, k, v, scale=None):
    """SDPA forced onto the math backend (fp32, materializes scores; slow, memory hungry)."""
    torch = _torch()
    scale = _scale_for(q, scale)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def comfy_attention(q, k, v, scale=None):
    """Exactly what h3_tensor_parallel._attn calls today (scale comes from ComfyUI).

    Normalized to (B, H, S, D) for comparison; ComfyUI returns (B, S, H*D) for the
    skip_reshape=True call the TP path makes, which is what `os_[r][a:b]` indexes into.
    """
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    heads = q.shape[1]
    out = optimized_attention(
        AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
        heads, mask=None, skip_reshape=True)
    if out.dim() == 3:
        out = out.reshape(out.shape[0], out.shape[1], heads, q.shape[-1]).transpose(1, 2)
    return out


# --- triton -----------------------------------------------------------------

_TRITON = {"loaded": False, "reason": "not requested (MMH3_TP_ATTN=triton or --selftest loads it)"}
_TL = None
_FA_FWD = None
# triton costs a second or two to import, so only when it is actually being used or measured
_WANT_TRITON = os.environ.get("MMH3_TP_ATTN", "").strip().lower() in ("triton", "int8") or "--selftest" in sys.argv
if _WANT_TRITON:
    try:
        import triton as _triton_mod
        import triton.language as _tl

        @_triton_mod.jit
        def _FA_FWD(Q, K, V, Out, scale,
                    stride_qz, stride_qm, stride_qd,
                    stride_kz, stride_kn, stride_kd,
                    stride_vz, stride_vn, stride_vd,
                    stride_oz, stride_om, stride_od,
                    N, BLOCK_M: _tl.constexpr, BLOCK_N: _tl.constexpr, D: _tl.constexpr):
            """Flash-attention forward, online softmax, no mask (H3 attention is full over the
            packed sequence). One program per (row band, batch*head plane)."""
            pid_m = _triton_mod.program_id(0)
            pid_z = _triton_mod.program_id(1)
            offs_m = pid_m * BLOCK_M + _tl.arange(0, BLOCK_M)
            offs_d = _tl.arange(0, D)
            mask_m = offs_m < N
            q = _tl.load(Q + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                         mask=mask_m[:, None], other=0.0)
            m_i = _tl.full([BLOCK_M], float("-inf"), _tl.float32)
            l_i = _tl.zeros([BLOCK_M], _tl.float32)
            acc = _tl.zeros([BLOCK_M, D], _tl.float32)
            for start_n in range(0, N, BLOCK_N):
                offs_n = start_n + _tl.arange(0, BLOCK_N)
                mask_n = offs_n < N
                k = _tl.load(K + pid_z * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                             mask=mask_n[:, None], other=0.0)
                scores = _tl.dot(q, _tl.trans(k)) * scale
                scores = _tl.where(mask_n[None, :], scores, float("-inf"))
                m_new = _tl.maximum(m_i, _tl.max(scores, 1))
                alpha = _tl.exp(m_i - m_new)
                p = _tl.exp(scores - m_new[:, None])
                l_i = l_i * alpha + _tl.sum(p, 1)
                acc = acc * alpha[:, None]
                v = _tl.load(V + pid_z * stride_vz + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                             mask=mask_n[:, None], other=0.0)
                acc = _tl.dot(p.to(_tl.float16), v, acc)
                m_i = m_new
            acc = acc / l_i[:, None]
            _tl.store(Out + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                      acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

        _TL = _tl
        _TRITON["loaded"] = True
        _TRITON["reason"] = "ok"
    except Exception as exc:  # pragma: no cover - depends on the box
        _TRITON["reason"] = f"{type(exc).__name__}: {exc}"


def triton_attention(q, k, v, scale=None, block_m=64, block_n=64, num_warps=4, num_stages=2):
    """Triton flash-attention forward. Raises if triton or the kernel is unavailable."""
    if not _TRITON["loaded"]:
        raise RuntimeError(f"triton backend unavailable ({_TRITON['reason']})")
    import triton
    torch = _torch()
    scale = _scale_for(q, scale)
    heads = q.shape[1]
    length = q.shape[-2]
    dim = q.shape[-1]
    if dim > 128 or (dim & (dim - 1)) != 0:
        raise ValueError(f"triton kernel needs a power-of-two head dim <= 128, got {dim}")
    qc = q.contiguous().reshape(-1, length, dim)
    kc = k.contiguous().reshape(-1, length, dim)
    vc = v.contiguous().reshape(-1, length, dim)
    out = _like((qc.shape[0], length, dim), q.dtype, q.device)
    grid = (triton.cdiv(length, block_m), qc.shape[0])
    _FA_FWD[grid](qc, kc, vc, out, scale,
                  qc.stride(0), qc.stride(1), qc.stride(2),
                  kc.stride(0), kc.stride(1), kc.stride(2),
                  vc.stride(0), vc.stride(1), vc.stride(2),
                  out.stride(0), out.stride(1), out.stride(2),
                  length, BLOCK_M=block_m, BLOCK_N=block_n, D=dim,
                  num_warps=num_warps, num_stages=num_stages)
    return out.reshape(q.shape[0], heads, length, dim)


# --- triton int8 (the one that can actually beat qk_int_sv_i8 on Turing) -----------------------
#
# Turing has int8 tensor cores at 2x the fp16 rate (65e12 int8 MAC/s per T4, 32.5e12 fp16 MAC/s), but
# the measured kitchen kernel runs at 17.5e12 MAC/s (35 TOPS) = 27% of the int8 peak, i.e. about the
# fp16 peak. So the headroom is a scheduler/quantization problem, not an instruction problem.
#
# Scheme (SageAttention-1 style, the design that is known to work on sm_75):
#   * Q, K: per-row symmetric int8.  score = (q8 @ k8^T) * (sq[:, None] * sk[None, :]) * softmax_scale
#   * P:    after the running-max subtraction every p is in (0, 1], so it quantizes with a FIXED
#           scale of 127. That fixed scale is what keeps the online-softmax accumulator consistent
#           while every tile rescales it by alpha.
#   * V:    per-channel (head-dim) symmetric int8, so the dequant factor multiplies the *output* of
#           P @ V per tile: acc += (p8 @ v8) * (sv[None, :] / 127). Per-tile channel scales are exact,
#           not an approximation, because the factor is applied before the accumulation.
# Every dequant factor is therefore exact; the only error is int8 rounding where the fp16 path would
# also round (scores in fp32 accumulate). Compare the two kernels head to head in --selftest.

_QUANT_ROW_K = None   # per-row symmetric int8 (Q, K)
_QUANT_COL_K = None   # per-channel symmetric int8 (V)
_FA_I8_K = None


if _WANT_TRITON:
    try:
        @_triton_mod.jit
        def _QUANT_ROW(X, X8, S, N,
                       stride_xz, stride_xn, stride_xd,
                       stride_qz, stride_qn, stride_qd,
                       stride_sz, stride_sn,
                       BLOCK_N: _tl.constexpr, D: _tl.constexpr):
            """[*, N, D] float -> int8 per row, scale = max|x| / 127 (symmetric, clamped to 127)."""
            pz = _triton_mod.program_id(0)
            pn = _triton_mod.program_id(1)
            offs_n = pn * BLOCK_N + _tl.arange(0, BLOCK_N)
            offs_d = _tl.arange(0, D)
            mask_n = offs_n < N
            x = _tl.load(X + pz * stride_xz + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                         mask=mask_n[:, None], other=0.0).to(_tl.float32)
            amax = _tl.max(_tl.abs(x), 1)
            scale = _tl.maximum(amax / 127.0, 1e-8)
            q = x / scale[:, None]
            q = _tl.where(q >= 0, q + 0.5, q - 0.5)              # round half away from zero
            q = _tl.minimum(_tl.maximum(q, -127.0), 127.0)
            _tl.store(X8 + pz * stride_qz + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                      q.to(_tl.int8), mask=mask_n[:, None])
            _tl.store(S + pz * stride_sz + offs_n * stride_sn, scale, mask=mask_n)

        @_triton_mod.jit
        def _QUANT_COL(X, X8, S, N,
                       stride_xz, stride_xn, stride_xd,
                       stride_qz, stride_qn, stride_qd,
                       stride_sz, stride_sd,
                       BLOCK_N: _tl.constexpr, D: _tl.constexpr):
            """[*, N, D] float -> int8 per channel: scale[d] = max_n|x[n, d]| / 127."""
            pz = _triton_mod.program_id(0)
            offs_d = _tl.arange(0, D)
            amax = _tl.zeros([D], _tl.float32)
            for start in range(0, N, BLOCK_N):
                offs_n = start + _tl.arange(0, BLOCK_N)
                x = _tl.load(X + pz * stride_xz + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                             mask=(offs_n < N)[:, None], other=0.0).to(_tl.float32)
                amax = _tl.maximum(amax, _tl.max(_tl.abs(x), 0))
            scale = _tl.maximum(amax / 127.0, 1e-8)
            _tl.store(S + pz * stride_sz + offs_d * stride_sd, scale)
            for start in range(0, N, BLOCK_N):
                offs_n = start + _tl.arange(0, BLOCK_N)
                mask_n = offs_n < N
                x = _tl.load(X + pz * stride_xz + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                             mask=mask_n[:, None], other=0.0).to(_tl.float32) / scale[None, :]
                x = _tl.where(x >= 0, x + 0.5, x - 0.5)
                x = _tl.minimum(_tl.maximum(x, -127.0), 127.0)
                _tl.store(X8 + pz * stride_qz + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                          x.to(_tl.int8), mask=mask_n[:, None])

        @_triton_mod.jit
        def _FA_I8(Q8, K8, V8, SQ, SK, SV, Out, scale, N,
                   stride_qz, stride_qm, stride_qd,
                   stride_kz, stride_kn, stride_kd,
                   stride_vz, stride_vn, stride_vd,
                   stride_oz, stride_om, stride_od,
                   stride_sqz, stride_sqm,
                   stride_skz, stride_skn,
                   stride_svz, stride_svd,
                   BLOCK_M: _tl.constexpr, BLOCK_N: _tl.constexpr, D: _tl.constexpr):
            """Int8 flash attention. P is quantized with the fixed 1/127 scale of the running-max
            softmax output, so the int8 PV accumulator stays consistent across the alpha rescaling."""
            pid_m = _triton_mod.program_id(0)
            pid_z = _triton_mod.program_id(1)
            offs_m = pid_m * BLOCK_M + _tl.arange(0, BLOCK_M)
            offs_d = _tl.arange(0, D)
            mask_m = offs_m < N
            q = _tl.load(Q8 + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                         mask=mask_m[:, None], other=0)          # stay int8: the dot wants int8 operands
            sq = _tl.load(SQ + pid_z * stride_sqz + offs_m * stride_sqm, mask=mask_m, other=1.0)
            sv = _tl.load(SV + pid_z * stride_svz + offs_d * stride_svd).to(_tl.float32)
            m_i = _tl.full([BLOCK_M], float("-inf"), _tl.float32)
            l_i = _tl.zeros([BLOCK_M], _tl.float32)
            acc = _tl.zeros([BLOCK_M, D], _tl.float32)
            for start_n in range(0, N, BLOCK_N):
                offs_n = start_n + _tl.arange(0, BLOCK_N)
                mask_n = offs_n < N
                k = _tl.load(K8 + pid_z * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                             mask=mask_n[:, None], other=0)      # int8, no fp32 round trip
                sk = _tl.load(SK + pid_z * stride_skz + offs_n * stride_skn, mask=mask_n, other=1.0)
                s = _tl.dot(q, _tl.trans(k), out_dtype=_tl.int32)
                s = s.to(_tl.float32) * (sq[:, None] * sk[None, :]) * scale
                s = _tl.where(mask_n[None, :], s, float("-inf"))
                m_new = _tl.maximum(m_i, _tl.max(s, 1))
                alpha = _tl.exp(m_i - m_new)
                p = _tl.exp(s - m_new[:, None])                              # in (0, 1]
                l_i = l_i * alpha + _tl.sum(p, 1)
                p8 = _tl.where(p >= 0, p * 127.0 + 0.5, p * 127.0 - 0.5)     # p >= 0, so plain rounding
                p8 = _tl.minimum(p8, 127.0).to(_tl.int8)
                v = _tl.load(V8 + pid_z * stride_vz + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                             mask=mask_n[:, None], other=0)      # int8: second dot is int8 x int8 -> int32
                pv = _tl.dot(p8, v, out_dtype=_tl.int32).to(_tl.float32)
                acc = acc * alpha[:, None] + pv * (sv / 127.0)[None, :]
                m_i = m_new
            acc = acc / l_i[:, None]
            _tl.store(Out + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                      acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

        _QUANT_ROW_K = _QUANT_ROW
        _QUANT_COL_K = _QUANT_COL
        _FA_I8_K = _FA_I8
    except Exception as exc:  # pragma: no cover
        _TRITON["reason"] = f"{type(exc).__name__}: {exc}"


def int8_attention(q, k, v, scale=None, block_m=128, block_n=64, num_warps=4, num_stages=2,
                   quant_block=128):
    """Triton int8 flash attention: per-row int8 Q/K, per-channel int8 V, fixed-scale int8 P.

    Same (B, H, S, D) contract as the other backends. Numerics are int8-level on the score and PV
    paths, which is what the kitchen int8 kernel this is meant to replace already does; `--selftest`
    prints the max-abs difference against both the fp32 reference and the ComfyUI kernel.
    """
    if _FA_I8_K is None:
        raise RuntimeError(f"triton int8 backend unavailable ({_TRITON['reason']})")
    import triton
    torch = _torch()
    scale = _scale_for(q, scale)
    heads = q.shape[1]
    length = q.shape[-2]
    dim = q.shape[-1]
    if dim > 128 or (dim & (dim - 1)) != 0:
        raise ValueError(f"triton int8 kernel needs a power-of-two head dim <= 128, got {dim}")
    qc = q.contiguous().reshape(-1, length, dim)
    kc = k.contiguous().reshape(-1, length, dim)
    vc = v.contiguous().reshape(-1, length, dim)
    planes = qc.shape[0]
    q8 = torch.empty_like(qc, dtype=torch.int8)
    k8 = torch.empty_like(kc, dtype=torch.int8)
    v8 = torch.empty_like(vc, dtype=torch.int8)
    sq = torch.empty((planes, length), dtype=torch.float32, device=q.device)
    sk = torch.empty((planes, length), dtype=torch.float32, device=q.device)
    sv = torch.empty((planes, dim), dtype=torch.float32, device=q.device)
    qb = max(1, min(quant_block, 128))
    for src, dst, sca in ((qc, q8, sq), (kc, k8, sk)):
        _QUANT_ROW_K[(planes, triton.cdiv(length, qb))](
            src, dst, sca, length,
            src.stride(0), src.stride(1), src.stride(2),
            dst.stride(0), dst.stride(1), dst.stride(2),
            sca.stride(0), sca.stride(1),
            BLOCK_N=qb, D=dim, num_warps=4)
    _QUANT_COL_K[(planes,)](
        vc, v8, sv, length,
        vc.stride(0), vc.stride(1), vc.stride(2),
        v8.stride(0), v8.stride(1), v8.stride(2),
        sv.stride(0), sv.stride(1),
        BLOCK_N=64, D=dim, num_warps=4)
    out = _like((planes, length, dim), q.dtype, q.device)
    grid = (triton.cdiv(length, block_m), planes)
    _FA_I8_K[grid](q8, k8, v8, sq, sk, sv, out, scale, length,
                   q8.stride(0), q8.stride(1), q8.stride(2),
                   k8.stride(0), k8.stride(1), k8.stride(2),
                   v8.stride(0), v8.stride(1), v8.stride(2),
                   out.stride(0), out.stride(1), out.stride(2),
                   sq.stride(0), sq.stride(1),
                   sk.stride(0), sk.stride(1),
                   sv.stride(0), sv.stride(1),
                   BLOCK_M=block_m, BLOCK_N=block_n, D=dim,
                   num_warps=num_warps, num_stages=num_stages)
    return out.reshape(q.shape[0], heads, length, dim)


BACKENDS = {
    "comfy": comfy_attention,
    "sdpa": sdpa_attention,
    "math": sdpa_math_attention,
    "triton": triton_attention,
    "int8": int8_attention,
    "ref": reference_attention,
}

_TRITON_CONFIGS = [(64, 64, 4), (128, 64, 4), (64, 128, 4), (128, 64, 8), (64, 64, 8)]
# int8 wants a bigger M tile: the K/V re-read per row band is what limits this kernel, and P @ V is
# int8 so the fp16-rated register pressure is lower per row of M.
# Turing int8 tensor core peak and HBM bandwidth, for the roofline printout. 65e12 MAC/s is
# 130 TOPS: every MAC is two ops, and confusing the two is a factor-of-two error in a floor.
T4_PEAK_MACS = 65e12
T4_PEAK_BYTES = 320e9
# BLOCK_M is the query-tile height; it sets how many times K and V are re-read (traffic ~ 1/BLOCK_M)
# while leaving the MAC count fixed, so sweeping it separates the two rooflines. 256-row tiles need
# more warps to fill: a 256x128 score tile is 32k fp32 accumulators.
_INT8_CONFIGS = [(128, 64, 4), (128, 128, 4), (64, 64, 4), (64, 128, 4), (128, 64, 8), (128, 128, 8),
                 (256, 64, 8), (256, 128, 8)]


def _roofline(length, planes, dim, ms, block_m):
    """Where a config sits against the two rooflines that could bind it.

    macs, TOPS and % of the int8 peak are independent of the tile; the K/V bytes scale with
    1/BLOCK_M (each query tile re-reads the whole K and V). If ms tracks the byte column, the
    kernel is traffic-bound and a bigger tile is the fix; if ms stays flat while bytes fall, it is
    not traffic, and the remaining distance to the compute floor is scheduling.
    """
    macs = 2 * planes * length * length * dim
    traffic = 2 * planes * (-(-length // block_m)) * length * dim      # int8 K/V: one byte each
    return {"ms": ms, "tops": 2 * macs / ms / 1e12, "pct_peak": macs / ms / T4_PEAK_MACS,
            "gbs": traffic / ms / 1e9, "pct_bw": traffic / ms / T4_PEAK_BYTES,
            "compute_floor_ms": macs / T4_PEAK_MACS * 1e3, "mem_floor_ms": traffic / T4_PEAK_BYTES * 1e3}


# ---------------------------------------------------------------------------
# selection used by the TP path
# ---------------------------------------------------------------------------

_BACKEND = None
_RESOLVED = False


def selected_backend():
    """Backend named by MMH3_TP_ATTN that passes a smoke test on this GPU, else None.

    None means `_attn` keeps calling ComfyUI's optimized_attention, i.e. today's behaviour.
    """
    global _BACKEND, _RESOLVED
    if _RESOLVED:
        return _BACKEND
    _RESOLVED = True
    name = os.environ.get("MMH3_TP_ATTN", "").strip().lower()
    if not name or name == "comfy":
        return None
    fn = BACKENDS.get(name)
    if fn is None:
        logger.warning("[MultiGPU] MMH3_TP_ATTN=%s is not a known backend (have: %s); using the ComfyUI path",
                       name, ", ".join(k for k in BACKENDS if k != "ref"))
        return None
    ok, detail = smoke_test(fn)
    if not ok:
        logger.warning("[MultiGPU] MMH3_TP_ATTN=%s failed its smoke test (%s); using the ComfyUI path", name, detail)
        return None
    logger.info("[MultiGPU] tensor-parallel attention backend: %s", name)
    _BACKEND = fn
    return _BACKEND


def smoke_test(fn, heads=4, length=256, dim=DEFAULT_DIM, device=None, tol=0.25):
    """Tiny correctness check so a broken backend falls back instead of corrupting a run."""
    torch = _torch()
    try:
        device = _device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return False, "no cuda"
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        torch.manual_seed(0)
        q = torch.randn(1, heads, length, dim, device=device, dtype=dtype)
        k = torch.randn(1, heads, length, dim, device=device, dtype=dtype)
        v = torch.randn(1, heads, length, dim, device=device, dtype=dtype)
        got = fn(q, k, v)
        want = reference_attention(q, k, v)
        if got.shape != want.shape:
            return False, f"shape {tuple(got.shape)} != {tuple(want.shape)}"
        error = (got.float() - want.float()).abs().max().item()
        if not torch.isfinite(got.float()).all():
            return False, "non-finite output"
        return (error <= tol), f"max abs err {error:.2e}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# self test / benchmark
# ---------------------------------------------------------------------------

def _time_ms(fn, iters=5, warmup=2):
    torch = _torch()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def _kernels_of(fn, iters=1):
    """Top CUDA kernel names `fn` launches; [] when there is no GPU or the profiler fails.

    Used to prove *which* kernel the ComfyUI baseline ran: the TP path calls optimized_attention
    without preferred_attention, so it is worth knowing whether `comfy` here is the kitchen int8
    kernel (qk_int_sv_i8) or a fallback.
    """
    torch = _torch()
    try:
        if not torch.cuda.is_available():
            return []
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        timed = sorted(((e.self_device_time_total, e.key) for e in prof.key_averages()),
                       key=lambda kv: -kv[0])
        return [name for total, name in timed if total > 0][:5]
    except Exception:
        return []


def selftest(n=DEFAULT_N, heads=DEFAULT_HEADS, dim=DEFAULT_DIM, device=None, dtype=None,
             iters=5, block_m=64, names=None):
    """Correctness + speed of every backend at the TP path's real shapes."""
    torch = _torch()
    device = _device(device)
    if device.type != "cuda":
        logger.warning("[MultiGPU] no CUDA device: running a tiny CPU sanity pass only")
        n, heads, dim = 512, 8, 64
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    lines = [f"turing_attention selftest — {time.strftime('%Y-%m-%dT%H:%M:%S')}",
             f"device {device} {device_name}, dtype {str(dtype).split('.')[-1]}, "
             f"shape (1, {heads}, {n}, {dim})"]
    try:
        lines.append(f"capability sm_{''.join(str(x) for x in torch.cuda.get_device_capability(device))}")
    except Exception:
        pass
    lines.append(f"triton: {'available' if _TRITON['loaded'] else 'unavailable — ' + _TRITON['reason']}"
                 f"{'' if not _TRITON['loaded'] else (', int8 kernels ok' if _FA_I8_K is not None else ', int8 kernels failed to build')}")

    torch.manual_seed(0)
    dtype = torch.float16 if device.type == "cuda" else dtype
    q = torch.randn(1, heads, n, dim, device=device, dtype=dtype)
    k = torch.randn(1, heads, n, dim, device=device, dtype=dtype)
    v = torch.randn(1, heads, n, dim, device=device, dtype=dtype)

    verify = min(n, 2048)
    qv, kv, vv = q[..., :verify, :], k[..., :verify, :], v[..., :verify, :]
    want = reference_attention(qv, kv, vv)

    candidates = names or ["comfy", "sdpa", "triton", "int8", "math"]
    rows = []
    for name in candidates:
        fn = BACKENDS.get(name)
        if fn is None:
            continue
        try:
            got = fn(qv, kv, vv)
            error = (got.float() - want.float()).abs().max().item() * 1000.0
            if name in ("math", "ref") and n > 4096:
                # fp32 scores at this length would need heads*verify*n*4 bytes of VRAM
                rows.append((float("inf"), name, f"{error:.2f}e-3", "timing skipped (fp32 scores too large)"))
                continue
            ms = _time_ms(lambda: fn(q, k, v), iters=iters)
            rows.append((ms, name, f"{error:.2f}e-3", ""))
        except Exception as exc:
            rows.append((float("inf"), name, "n/a", f"{type(exc).__name__}: {exc}"[:70]))

    sweets = []
    sweeps = []
    for kernel, configs in (("triton", _TRITON_CONFIGS), ("int8", _INT8_CONFIGS)):
        if not any(c == kernel for c in candidates) or not _TRITON["loaded"]:
            continue
        runner = BACKENDS[kernel]
        best = None
        sweep = []
        for bm, bn, nw in configs:
            try:
                fn = lambda: runner(q, k, v, block_m=bm, block_n=bn, num_warps=nw)
                ms = _time_ms(fn, iters=iters)
                sweep.append((ms, bm, bn, nw))
                if best is None or ms < best[0]:
                    best = (ms, bm, bn, nw)
            except Exception as exc:
                rows.append((float("inf"), f"{kernel} bm{bm} bn{bn} w{nw}", "n/a", f"{type(exc).__name__}: {exc}"[:70]))
        sweeps.append((kernel, sweep))
        if best is not None:
            ms, bm, bn, nw = best
            got = runner(qv, kv, vv, block_m=bm, block_n=bn, num_warps=nw)
            error = (got.float() - want.float()).abs().max().item() * 1000.0
            rows = [r for r in rows if not r[1].startswith(kernel)]
            rows.append((ms, f"{kernel} bm{bm} bn{bn} w{nw}", f"{error:.2f}e-3", ""))
            sweets.append((kernel, bm, bn, nw, got))

    # the number that decides a drop-in swap: how far is the new kernel from the one it replaces
    notes = []
    if any(r[1].startswith("int8") for r in rows):
        try:
            base = comfy_attention(qv, kv, vv).float()
            for kernel, bm, bn, nw, got in sweets:
                if kernel != "int8":
                    continue
                delta = (got.float() - base).abs().max().item()
                rel = (got.float() - base).abs().max().item() / base.abs().max().clamp(min=1e-6).item()
                notes.append(f"int8 vs the ComfyUI int8 kernel it would replace: max abs {delta:.4f} "
                             f"({rel * 100:.2f}% of the output scale)")
        except Exception as exc:
            notes.append(f"int8 vs comfy comparison failed: {type(exc).__name__}: {exc}")
    # roofline section: the numbers that decide what to do next about the kernel that owns 35% of
    # a step. Print every config, not just the winner, because the *shape* of the sweep is the answer.
    for kernel, sweep in sweeps:
        if not sweep:
            continue
        notes.append("")
        notes.append(f"{kernel} sweep at (batch 1, {heads} heads, {n} tokens, {dim} dim), "
                     f"int8 K/V")
        notes.append(f"  {'bm':>4s}{'bn':>5s}{'w':>3s}{'ms':>9s}{'TOPS':>8s}{'%peak':>7s}"
                     f"{'GB/s':>8s}{'%bw':>6s}{'floor ms':>10s}  (compute/memory floors)")
        for ms, bm, bn, nw in sorted(sweep, key=lambda r: (r[1], r[2], r[3])):
            r = _roofline(n, heads, dim, ms, bm)
            notes.append(f"  {bm:4d}{bn:5d}{nw:3d}{ms * 1e3:9.2f}{r['tops']:8.1f}"
                         f"{r['pct_peak'] * 100:6.0f}%{r['gbs']:8.0f}{r['pct_bw'] * 100:5.0f}%"
                         f"{r['compute_floor_ms']:5.0f}/{r['mem_floor_ms']:.0f}")
        # verdict from the tile sensitivity: traffic falls as 1/BLOCK_M, MACs do not move at all
        by_m = {}
        for ms, bm, bn, nw in sorted(sweep, key=lambda r: r[0]):
            by_m.setdefault(bm, ms)
        if len(by_m) >= 2:
            lo, hi = min(by_m), max(by_m)
            best_lo, best_hi = by_m[lo], by_m[hi]
            bytes_ratio = hi / lo
            notes.append(f"  tile sensitivity: best ms at BLOCK_M {lo} is {best_lo * 1e3:.1f}, at {hi} is "
                         f"{best_hi * 1e3:.1f} -> {best_hi / best_lo:.2f}x slower with {bytes_ratio:.0f}x the K/V traffic")
            if best_hi / best_lo > 0.6 * bytes_ratio:
                notes.append("  verdict: ms tracks the byte column -> traffic-bound; raise BLOCK_M further")
            elif best_hi / best_lo < 0.25 * bytes_ratio:
                notes.append("  verdict: ms is nearly flat in tile size while traffic falls -> NOT traffic-bound;"
                             " the distance to the compute floor is scheduling/issue, not bandwidth")
            else:
                notes.append("  verdict: mixed -- some traffic sensitivity, well short of the 1/BLOCK_M slope")
            notes.append(f"  (the kitchen kernel's own 64->128 tile gain was 10-12%, while its traffic halves:"
                         f" that measurement already argues against traffic being what binds)")

    if device.type == "cuda":
        # which kernel did the baseline actually run, and is the new one a drop-in for the 335 ms
        # you measured for qk_int_sv_i8 at 28.5k tokens?
        try:
            base_k = _kernels_of(lambda: comfy_attention(qv, kv, vv))
            if base_k:
                notes.append("comfy ran: " + ", ".join(k[:44] for k in base_k[:3]))
            if any(r[1].startswith("int8") for r in rows):
                new_k = _kernels_of(lambda: BACKENDS["int8"](qv, kv, vv))
                if new_k:
                    notes.append("int8 ran: " + ", ".join(k[:44] for k in new_k[:3]))
        except Exception:
            pass
        notes.append("sanity: your profiled kitchen int8 attention was 335 ms/call at 28.5k tokens "
                     "x 28 heads; a `comfy` row far above that means this harness is not reaching it, "
                     "so compare the int8 row against 335 ms directly")

    rows.sort(key=lambda row: row[0])
    baseline = next((ms for ms, name, _, _ in rows if name == "comfy"), None)
    lines.append("")
    lines.append(f"{'backend':28s}{'ms/call':>10s}{'maxerr':>10s}{'vs comfy':>10s}  note")
    for ms, name, error, note in rows:
        speedup = "" if baseline in (None, 0) or ms in (float("inf"),) else f"{baseline / ms:9.2f}x"
        ms_text = "  failed" if ms == float("inf") else f"{ms:10.2f}"
        lines.append(f"{name:28s}{ms_text:>10s}{error:>10s}{speedup:>10s}  {note}")
    lines.extend(notes)
    best = rows[0] if rows else None
    if best:
        per_layer = best[0] / 1000.0
        lines.append("")
        lines.append(f"fastest: {best[1]} at {best[0]:.1f} ms/call for {heads} heads x {n} tokens "
                     f"(50 DiT blocks -> {per_layer * 50:.1f} s/step of attention)")
        lines.append(f"use it with: MMH3_TP_ATTN={best[1].split()[0]}")
    text = "\n".join(lines)
    print(text)
    try:
        with open(REPORT, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"\nwritten to {os.path.abspath(REPORT)}")
    except OSError:
        pass
    return text


def _ensure_comfy_importable():
    """Make `import comfy...` work when this script is run from the custom node directory."""
    for candidate in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        path = candidate
        for _ in range(4):
            path = os.path.dirname(path)
            if os.path.isdir(os.path.join(path, "comfy", "ldm", "modules")):
                if path not in sys.path:
                    sys.path.insert(0, path)
                return path
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="attention backends for the H3 tensor-parallel path")
    parser.add_argument("--selftest", action="store_true", help="correctness + speed on this GPU")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="sequence length (packed H3 tokens)")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS, help="heads per GPU (56 split over 2)")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--backends", default="comfy,int8,triton,sdpa,math")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.selftest:
        parser.print_help()
        return 1
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not importable here — run this with the same python that runs ComfyUI "
              "(on Kaggle, the notebook environment).")
        return 2
    _ensure_comfy_importable()
    selftest(n=args.n, heads=args.heads, dim=args.dim, device=args.device, iters=args.iters,
             names=[name.strip() for name in args.backends.split(",") if name.strip()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
