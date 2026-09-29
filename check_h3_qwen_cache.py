"""Offline checks for the H3 Qwen3-VL vision cache (h3_qwen_cache.py).

The claim that needs proving is narrow but load-bearing: a prompt-only change must reuse the vision
tower *without* any way for a second job to read corrupted tensors. So this checks, with a fake
torch and a fake MiniMaxQwen3VL:

  1. same pixels -> the tower runs once, the second call is served from cache and returns equal values
  2. different pixels -> the tower runs; different device / video-block flag -> separate entries
  3. aliasing: mutating anything a call returned, on a hit *or* a miss, leaves the cache intact
     (the failure mode would be a silently wrong face on the next job)
  4. the byte limit evicts oldest-first and never exceeds itself
  5. non-image entries bypass the cache entirely, and MMH3_QWEN_CACHE=0 turns it off

    python check_h3_qwen_cache.py        # exits 1 on any failure
"""
import importlib
import sys
import types

import numpy as np


class Tensor:
    """A numpy-backed tensor with just the torch surface h3_qwen_cache uses."""

    def __init__(self, array, device="cuda:0"):
        self.a = np.asarray(array, dtype="float32")
        self.device = device

    def clone(self):
        return Tensor(self.a.copy(), self.device)

    def detach(self):
        return self

    def to(self, device):
        dev = self.device if device == self.device else device
        return Tensor(self.a.copy(), dev)

    def contiguous(self):
        return Tensor(self.a, self.device)

    def numpy(self):
        return self.a

    @property
    def shape(self):
        return self.a.shape

    def numel(self):
        return self.a.size

    def element_size(self):
        return self.a.dtype.itemsize

    def __eq__(self, other):
        return isinstance(other, Tensor) and np.array_equal(self.a, other.a)


def install_fakes(cache_mb="768", enabled="1"):
    import os
    os.environ["MMH3_QWEN_CACHE"] = enabled
    os.environ["MMH3_QWEN_CACHE_MB"] = cache_mb

    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.is_tensor = lambda x: isinstance(x, Tensor)
    sys.modules["torch"] = torch

    calls = {"n": 0}

    class Vision:                            # weak-referenceable, like the nn.Module it stands for
        pass

    class MiniMaxQwen3VL:
        def __init__(self):
            self.visual = Vision()           # identity is what the key checks

        def preprocess_embed(self, embed, device):
            calls["n"] += 1
            n_tok = int(embed["data"].a.shape[0])
            merged = Tensor(np.full((n_tok, 8), float(n_tok)), device)
            deepstack = [Tensor(np.full((n_tok, 8), i + 1.0), device) for i in range(3)]
            return merged, {"grid": Tensor(np.array([[1, n_tok, 1]]), device), "deepstack": deepstack}

    te = types.ModuleType("comfy.text_encoders.minimax")
    te.MiniMaxQwen3VL = MiniMaxQwen3VL
    for name in ("comfy", "comfy.text_encoders"):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    sys.modules["comfy.text_encoders.minimax"] = te
    return MiniMaxQwen3VL, calls


def reload_module():
    for name in list(sys.modules):
        if name == "h3_qwen_cache":
            del sys.modules[name]
    return importlib.import_module("h3_qwen_cache")


def entry(seed, device="cuda:0", video_block=False):
    data = Tensor(np.random.default_rng(seed).random((16, 4, 4)).astype("float32"), device)
    return {"type": "image", "data": data, "original_image": data, "minimax_video_block": video_block}


def main():
    checks = []
    cls, calls = install_fakes()
    sf = reload_module()
    assert sf.patch_h3_qwen_cache(), "patch did not install"
    model = cls()

    # 1. same pixels: one tower run, second call from cache
    a = entry(1)
    first = sf.__dict__  # noqa - keep the module reachable for the reader
    merged1, extra1 = model.preprocess_embed(entry(1), "cuda:0")
    merged2, extra2 = model.preprocess_embed(entry(1), "cuda:0")
    checks.append(("same pixels: tower ran once (%d)" % calls["n"], calls["n"] == 1))
    checks.append(("cached values equal", merged1 == merged2 and extra1["deepstack"][0] == extra2["deepstack"][0]))
    checks.append(("grid preserved", extra1["grid"] == extra2["grid"]))

    # 2. different pixels / device / flag -> separate entries
    model.preprocess_embed(entry(2), "cuda:0")
    checks.append(("different pixels: tower ran again (%d)" % calls["n"], calls["n"] == 2))
    model.preprocess_embed(entry(2), "cuda:1")
    checks.append(("different device is a separate entry (%d)" % calls["n"], calls["n"] == 3))
    model.preprocess_embed(entry(2, video_block=True), "cuda:0")
    checks.append(("video-block flag is part of the key (%d)" % calls["n"], calls["n"] == 4))
    model.preprocess_embed(entry(2), "cuda:0")
    checks.append(("the plain entry is still cached (%d)" % calls["n"], calls["n"] == 4))

    # 3. aliasing: writing through a returned tensor must not reach the cache
    hit_a, hit_extra = model.preprocess_embed(entry(1), "cuda:0")
    hit_a.a[:] = -12345.0
    hit_extra["deepstack"][0].a[:] = -12345.0
    again, again_extra = model.preprocess_embed(entry(1), "cuda:0")
    checks.append(("mutating a hit's tensors does not corrupt the cache",
                   float(again.a[0, 0]) != -12345.0 and float(again_extra["deepstack"][0].a[0, 0]) != -12345.0))
    before = calls["n"]
    miss_a, miss_extra = model.preprocess_embed(entry(3), "cuda:0")
    miss_a.a[:] = -999.0
    miss_extra["deepstack"][0].a[:] = -999.0
    checks.append(("mutating a miss's tensors does not corrupt the cache", calls["n"] == before + 1))
    cached3, cached3_extra = model.preprocess_embed(entry(3), "cuda:0")
    checks.append(("the next job reads the correct values",
                   float(cached3.a[0, 0]) != -999.0 and float(cached3_extra["deepstack"][0].a[0, 0]) > 0))

    # 4. non-image entries bypass
    before = calls["n"]
    out = model.preprocess_embed({"type": "audio", "data": Tensor(np.zeros((4, 4)))}, "cuda:0")
    checks.append(("non-image entries bypass the cache", calls["n"] == before + 1 and out[0] is not None))

    # 5. the byte limit evicts and holds
    cls, calls = install_fakes(cache_mb="0.003")            # one entry is ~2.1 kB: one fits, four do not
    sf = reload_module()
    cls._mmh3_qwen_cache = False        # a fresh process would not carry the previous install
    assert sf.patch_h3_qwen_cache(), "second install failed"
    model = cls()
    for seed in range(4):
        model.preprocess_embed(entry(100 + seed), "cuda:0")
    checks.append(("byte limit holds (%.0f B <= 3000 B)" % sf._STATS["bytes"], sf._STATS["bytes"] <= 3000))
    checks.append(("eviction kept the newest entry (%d entries)" % sf._STATS["entries"], sf._STATS["entries"] == 1))
    before = calls["n"]
    model.preprocess_embed(entry(103), "cuda:0")
    checks.append(("the newest entry is still a hit", calls["n"] == before))
    model.preprocess_embed(entry(100), "cuda:0")
    checks.append(("the evicted entry recomputes", calls["n"] == before + 1))

    # 6. disabling
    cls, calls = install_fakes(enabled="0")
    sf = reload_module()
    installed = sf.patch_h3_qwen_cache()
    model = cls()
    model.preprocess_embed(entry(1), "cuda:0")
    model.preprocess_embed(entry(1), "cuda:0")
    checks.append(("MMH3_QWEN_CACHE=0 leaves the node unpatched", installed is False and calls["n"] == 2))

    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
