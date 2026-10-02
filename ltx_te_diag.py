"""Text-encoder timing diagnostics, and a log of ComfyUI RAM-cache evictions.

Wraps comfy.sd.CLIP.load_model and encode_from_tokens and logs, per encode: load time and encode time, bytes this
process read from disk during each (/proc/self/io read_bytes: page-cache hits do not count), how much of the
encoder was already on its GPU before loading and after, and free GPU memory / host RAM. That tells a disk
re-read from a host-RAM upload from plain compute. LTX_TE_DIAG=0 turns it off.
"""
import logging
import os
import time

import torch

import comfy.sd

logger = logging.getLogger("MultiGPU")


def _disk_read():
    try:
        for line in open("/proc/self/io"):
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def _ram():
    try:
        mi = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        return int(mi["MemAvailable"].split()[0]) / 2**20
    except Exception:
        return float("nan")


def _loaded(patcher):
    try:
        return patcher.loaded_size() / 2**30
    except Exception:
        return float("nan")


def _log_cache_evictions():
    """ComfyUI's RAM-pressure output cache drops node outputs (text encodes, loaded VAEs) when host RAM runs low, so
    the next prompt re-runs them. Log each eviction pass that removed something (ComfyUI logs it below INFO)."""
    try:
        import comfy_execution.caching as caching
    except ImportError:
        return
    orig = caching.RAMPressureCache.ram_release

    def ram_release(self, target, *args, **kwargs):
        n, ram = len(self.cache), _ram()
        freed = orig(self, target, *args, **kwargs)
        if len(self.cache) < n:
            logger.info("[MultiGPU TE] ComfyUI RAM cache evicted %d node outputs (target %.1f GB available, had %.1f GB, free_active=%s)",
                        n - len(self.cache), target / 2**30, ram, kwargs.get("free_active", args[0] if args else False))
        return freed
    caching.RAMPressureCache.ram_release = ram_release


def install_te_diag():
    if os.environ.get("LTX_TE_DIAG", "1") == "0" or getattr(comfy.sd.CLIP.load_model, "_mgpu_te_diag", False):
        return
    orig_load, orig_encode = comfy.sd.CLIP.load_model, comfy.sd.CLIP.encode_from_tokens
    _log_cache_evictions()
    state = {}

    def load_model(self, *args, **kwargs):
        dev = self.patcher.load_device
        before, disk0, t0 = _loaded(self.patcher), _disk_read(), time.perf_counter()
        out = orig_load(self, *args, **kwargs)
        if isinstance(dev, torch.device) and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        state["load"] = (time.perf_counter() - t0, (_disk_read() - disk0) / 2**30, before, _loaded(self.patcher))
        return out

    def encode_from_tokens(self, *args, **kwargs):
        dev = self.patcher.load_device
        state.pop("load", None)
        disk0, t0 = _disk_read(), time.perf_counter()
        out = orig_encode(self, *args, **kwargs)
        if isinstance(dev, torch.device) and dev.type == "cuda":
            torch.cuda.synchronize(dev)
            free = torch.cuda.mem_get_info(dev)[0] / 2**30
        else:
            free = float("nan")
        total, disk = time.perf_counter() - t0, (_disk_read() - disk0) / 2**30
        lt, ld, before, after = state.get("load", (float("nan"),) * 4)
        logger.info("[MultiGPU TE] %s on %s: %.1fs total = load %.1fs (disk read %.2f GB, on GPU %.2f -> %.2f GB) + encode %.1fs "
                    "(disk read %.2f GB) | %s %.1f GB free | host RAM available %.1f GB",
                    type(self.cond_stage_model).__name__, dev, total, lt, ld, before, after, total - lt, disk - ld,
                    dev, free, _ram())
        return out

    load_model._mgpu_te_diag = True
    comfy.sd.CLIP.load_model = load_model
    comfy.sd.CLIP.encode_from_tokens = encode_from_tokens
