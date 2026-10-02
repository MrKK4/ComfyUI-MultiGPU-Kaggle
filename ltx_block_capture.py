"""Opt-in capture of one LTX audio-video transformer block call, for the tensor-parallel block test.

LTX_BLOCK_CAPTURE=<file>: the first BasicAVTransformerBlock call whose video stream has at least
LTX_BLOCK_CAPTURE_MIN_TOKENS tokens (default 8000: the 1 MP stage-2 pass, block 0) is saved with torch.save:
its inputs (copied before the in-place forward), its output, and the single-GPU time of the call plus two
repeats on copies. Tensors go to the CPU; transformer_options keeps only plain values. With
LTX_BLOCK_CAPTURE_STOP=1 the job is stopped right after the capture. Does nothing unless the variable is set.
"""
import logging
import os
import time

import torch

from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock, CompressedTimestep

logger = logging.getLogger("MultiGPU")

_PLAIN = (bool, int, float, str, type(None))


def _cpu(o):
    if torch.is_tensor(o):
        return o.detach().to("cpu").clone()
    if isinstance(o, CompressedTimestep):
        c = CompressedTimestep.__new__(CompressedTimestep)
        for k in CompressedTimestep.__slots__:
            setattr(c, k, _cpu(getattr(o, k)))
        return c
    if isinstance(o, (list, tuple)):
        return type(o)(_cpu(v) for v in o)
    if isinstance(o, dict):
        return {k: _cpu(v) for k, v in o.items()}
    return o if isinstance(o, _PLAIN) else "<dropped %s>" % type(o).__name__


def _gpu_copy(o, dev):
    if torch.is_tensor(o):
        return o.clone()
    if isinstance(o, (list, tuple)):
        return type(o)(_gpu_copy(v, dev) for v in o)
    return o


def _plain_options(topts):
    return {k: v for k, v in (topts or {}).items()
            if isinstance(v, _PLAIN) or (isinstance(v, (tuple, list)) and all(isinstance(x, _PLAIN) for x in v))}


def install_block_capture():
    path = os.environ.get("LTX_BLOCK_CAPTURE")
    if not path or getattr(BasicAVTransformerBlock.forward, "_mgpu_capture", False):
        return
    min_tokens = int(os.environ.get("LTX_BLOCK_CAPTURE_MIN_TOKENS", "8000"))
    stop = os.environ.get("LTX_BLOCK_CAPTURE_STOP", "0") == "1"
    orig = BasicAVTransformerBlock.forward
    state = {"done": False}

    def forward(self, x, **kwargs):
        if state["done"] or x[0].shape[1] < min_tokens:
            return orig(self, x, **kwargs)
        state["done"] = True
        dev = x[0].device
        kw_cpu = {k: (_plain_options(v) if k == "transformer_options" else _cpu(v)) for k, v in kwargs.items()}
        x_cpu = _cpu(x)
        spare = [_gpu_copy(x, dev) for _ in range(2)]
        times = []
        for xi in [x] + spare:
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            out = orig(self, xi, **kwargs)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t0)
            if xi is x:
                result = out
        del spare
        dropped = sorted(k for k, v in (kwargs.get("transformer_options") or {}).items() if k not in kw_cpu["transformer_options"])
        torch.save({"x": x_cpu, "kwargs": kw_cpu, "out": _cpu(result), "time_s": times,
                    "device": str(dev), "dropped_transformer_options": dropped,
                    "weight_devices": sorted({str(p.device) for p in self.parameters()})}, path)
        logger.info("[MultiGPU] LTX block captured to %s: video %s audio %s, single-GPU %s s, weights on %s",
                    path, tuple(x_cpu[0].shape), tuple(x_cpu[1].shape), ", ".join("%.3f" % t for t in times),
                    sorted({str(p.device) for p in self.parameters()}))
        if stop:
            raise RuntimeError("LTX block captured (LTX_BLOCK_CAPTURE_STOP=1): stopping the job")
        return result

    forward._mgpu_capture = True
    BasicAVTransformerBlock.forward = forward
    logger.info("[MultiGPU] LTX block capture armed: %s (video tokens >= %d)", path, min_tokens)
