"""Text-encoder timing diagnostics, and a page-cache prefetch of the text encoder files after a large VAE decode.

Wraps comfy.sd.CLIP.load_model and encode_from_tokens and logs, per encode: load time and encode time, bytes this
process read from disk during each (/proc/self/io read_bytes: page-cache hits do not count), how much of the
encoder was already on its GPU before loading and after, and free GPU memory / host RAM. That tells a disk
re-read from a host-RAM upload from plain compute. LTX_TE_DIAG=0 turns it off.
"""
import logging
import os
import threading
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


_TE_FILES = []  # text encoder checkpoint files ComfyUI has loaded this session
_PREFETCH = {"thread": None}


def prefetch_text_encoders():
    """Read the loaded text encoder files back into the page cache on a background thread (called after a large
    VAE decode): the encoder's dynamic weights are paged from its file, and by the next prompt the OS has
    usually dropped those pages (ok-37: 7.5-8.5 GB re-read from disk, ~26 s per new prompt). Skipped when host
    RAM is too low to keep them. LTX_TE_PREFETCH=0 turns it off."""
    if os.environ.get("LTX_TE_PREFETCH", "1") == "0" or not _TE_FILES:
        return
    t = _PREFETCH["thread"]
    if t is not None and t.is_alive():
        return
    files = [f for f in _TE_FILES if os.path.isfile(f)]
    size = sum(os.path.getsize(f) for f in files) / 2**30
    if _ram() < size + 2:
        logger.info("[MultiGPU TE] prefetch skipped: %.1f GB of text encoder files, host RAM available %.1f GB", size, _ram())
        return

    def run():
        t0, done = time.perf_counter(), 0
        for f in files:
            with open(f, "rb", buffering=0) as fh:
                try:
                    os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
                except (AttributeError, OSError):
                    pass
                while True:
                    b = fh.read(64 << 20)
                    if not b:
                        break
                    done += len(b)
        logger.info("[MultiGPU TE] prefetched %.1f GB of text encoder files into the page cache in %.1fs | host RAM available %.1f GB",
                    done / 2**30, time.perf_counter() - t0, _ram())

    _PREFETCH["thread"] = threading.Thread(target=run, name="te-prefetch", daemon=True)
    _PREFETCH["thread"].start()
    logger.info("[MultiGPU TE] prefetching %.1f GB of text encoder files in the background: %s", size,
                ", ".join(os.path.basename(f) for f in files))


def install_te_diag():
    if os.environ.get("LTX_TE_DIAG", "1") == "0" or getattr(comfy.sd.CLIP.load_model, "_mgpu_te_diag", False):
        return
    orig_load, orig_encode, orig_load_clip = comfy.sd.CLIP.load_model, comfy.sd.CLIP.encode_from_tokens, comfy.sd.load_clip

    def load_clip(ckpt_paths, *args, **kwargs):
        for p in ckpt_paths or []:
            if p not in _TE_FILES:
                _TE_FILES.append(p)
        return orig_load_clip(ckpt_paths, *args, **kwargs)
    comfy.sd.load_clip = load_clip
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
