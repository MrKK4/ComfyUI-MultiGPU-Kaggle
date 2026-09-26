"""fp16 compute for LTX 2.x on GPUs without bf16 (Turing / T4).

ComfyUI lists LTXV/LTXAV inference dtypes as bf16/fp32, so a T4 runs the DiT in fp32.
Measured on LTX 2.5 22B distilled (2x T4, 896x512x121): largest activation anywhere in the
blocks is the video residual at ~1.6e4, 4x below the fp16 limit, and plain fp16 produced no
inf/NaN, so no mixed-precision rewrites are needed (unlike H3). Stage-2 steps ~13.5 -> ~10.4 s.
Disable with LTX_FP16=0."""
import logging
import os

import torch

logger = logging.getLogger("MultiGPU")


def patch_ltx_fp16():
    if os.environ.get("LTX_FP16", "1") == "0" or not torch.cuda.is_available():
        return False
    if any(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count())):
        return False
    import comfy.supported_models
    for cls in (comfy.supported_models.LTXV, comfy.supported_models.LTXAV):
        cls.supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    logger.info("[MultiGPU] LTX fp16 compute enabled (no bf16 GPU); LTX_FP16=0 disables")
    return True
