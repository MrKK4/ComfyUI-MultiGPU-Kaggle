"""comfy-kitchen 0.2.35 runs its SM80-only CUTLASS fp16 conv3d/linear kernels on
Turing (T4, SM75): they print "Mma<16,8,16> not implemented", decline or fault, and
leave a CUDA error that the next kitchen launch reports ("CUDA INT8 rowwise
quantization failed: invalid argument"). MiniMax H3's VAE encoder hits it on
single-frame reference images with --fast fp16_accumulation. Below Ampere both
kernels fall back to the torch ops kitchen already uses for declined shapes."""
import importlib
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("MultiGPU")

_pre_ampere = {}


def _is_pre_ampere(t):
    if not t.is_cuda:
        return False
    index = t.device.index if t.device.index is not None else torch.cuda.current_device()
    if index not in _pre_ampere:
        _pre_ampere[index] = torch.cuda.get_device_capability(index)[0] < 8
    return _pre_ampere[index]


def patch_comfy_kitchen_turing():
    try:
        ck_cuda = importlib.import_module("comfy_kitchen.backends.cuda")
        from comfy_kitchen.backends._activations import apply_residual
    except ImportError:
        return False
    if getattr(ck_cuda.fp16_linear, "_mmh3_turing_fix", False):
        return True

    cutlass_fp16_conv3d = ck_cuda._cutlass_fp16_conv3d
    fp16_linear = ck_cuda.fp16_linear

    def cutlass_fp16_conv3d_guarded(x, weight, bias, residual, stride, config=-1):
        if _is_pre_ampere(x):
            return None
        return cutlass_fp16_conv3d(x, weight, bias, residual, stride, config)

    def fp16_linear_guarded(x, weight, bias=None, residual=None, residual_scale=None):
        if _is_pre_ampere(x):
            return apply_residual(F.linear(x, weight, bias), residual, residual_scale)
        return fp16_linear(x, weight, bias, residual, residual_scale)

    fp16_linear_guarded._mmh3_turing_fix = True
    ck_cuda._cutlass_fp16_conv3d = cutlass_fp16_conv3d_guarded
    ck_cuda.fp16_linear = fp16_linear_guarded
    logger.info("[MultiGPU] SM80-only comfy_kitchen CUTLASS fp16 conv3d/linear routed to torch below Ampere")
    return True
