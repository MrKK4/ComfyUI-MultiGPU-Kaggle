"""Offline check for h3_ref_cache.py's key: it must hit when nothing changed and never otherwise.

The failure modes worth pinning are not "does the LRU evict" but the ones that would silently hand a
job someone else's face:

  * two encoders with an identical weight *layout* but different weights (the collision the old
    name/shape/dtype fingerprint could not see) -> different keys
  * two faces whose tensors share their first row of pixels (the old key hashed `img[:1]`) ->
    different keys
  * the same face encoded by an encoder another node has patched -> different keys
  * anything else unchanged -> the same key, and the second call must not run the encode

    python check_h3_ref_cache.py        # exits 1 on any failure
"""
import os
import sys
import types

import numpy as np


# --------------------------------------------------------------------------- fake torch

class _Bool:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=bool)

    def sum(self):
        return int(self.array.sum())

    def any(self):
        return bool(self.array.any())


class _T:
    def __init__(self, array, dtype=None):
        self.array = np.asarray(array)
        self.dtype = dtype or str(self.array.dtype)

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return _T(self.array.astype("float32"))

    def numpy(self):
        return self.array

    def numel(self):
        return int(self.array.size)

    def reshape(self, *shape):
        return _T(self.array.reshape(shape if len(shape) > 1 else shape[0]))

    def __getitem__(self, idx):
        return _T(self.array[idx])

    def __ne__(self, other):
        return _Bool(self.array != getattr(other, "array", other))


def install_fakes():
    torch = types.ModuleType("torch")
    torch.Tensor = _T
    torch.is_tensor = lambda x: isinstance(x, _T)
    torch.equal = lambda a, b: np.array_equal(a.array, b.array)
    sys.modules["torch"] = torch
    return torch


# --------------------------------------------------------------------------- the node under test

ENCODED = []          # every real encode the patched node performed
EMPTY_LATENTS = []
MUTATE = [False]      # makes the fake encoder return a different conditioning, to test VERIFY


def _empty_av_latent(width, height, length):       # stands in for the node module's helper
    EMPTY_LATENTS.append((width, height, length))
    return ("latent", None)


class _NodeOutput:
    def __init__(self, *args):
        self.args = tuple(args)


def make_node():
    class FakeNode:
        @classmethod
        def execute(cls, *args, **kwargs):
            prompt = args[1]                     # cls is bound, so args[0] is the clip
            ENCODED.append(prompt)
            value = float(sum(map(ord, str(prompt))) % 997)     # deterministic per prompt ...
            if MUTATE[0]:
                value += 1.0                                    # ... except when the test asks
            return _NodeOutput(_T([value], dtype="float16"), "latent")

    FakeNode.__module__ = "__main__"          # so the patcher finds _empty_av_latent here
    return FakeNode


def install_node():
    import nodes
    from comfy_api.latest import io
    node = make_node()
    nodes.NODE_CLASS_MAPPINGS = {"MiniMaxH3ReferenceToVideo": node}
    io.NodeOutput = _NodeOutput
    return node


def model(tag, **weights):
    """A fake encoder: state_dict of hand-picked tensors, so layout and content are controllable."""
    class _Model:
        def state_dict(self):
            return {k: _T(np.asarray(v)) for k, v in weights.items()}

    return _Model()


def encoder(tag="layout", patch_keys=()):
    class _Patcher:
        def __init__(self):
            self.patches = {k: None for k in patch_keys}

    class _Clip:
        def __init__(self):
            self.cond_stage_model = model(tag, a=np.ones((4, 8)), b=np.full((2, 2), tag == "layout"))
            self.patcher = _Patcher()

    return _Clip()


def face(seed, first_row_same=False):
    arr = np.random.default_rng(seed).random((8, 8, 3)).astype("float32")
    if first_row_same:
        arr[0] = 0.25
    return _T(arr)


def run(node, prompt="a face", images=None, width=768, height=1344, length=124, clip=None, vae=None,
        size_mode="match"):
    images = {"ref": face(1)} if images is None else images
    return node.execute(clip or encoder(), prompt, width, height, length, size_mode,
                        vae or encoder("vae"), None, images, None, None, None)


def main():
    install_fakes()
    nodes_mod = types.ModuleType("nodes")
    sys.modules["nodes"] = nodes_mod
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    io_mod = types.ModuleType("comfy_api.latest.io")
    comfy_api.latest, latest.io = latest, io_mod
    latest.io = io_mod
    for name, mod in (("comfy_api", comfy_api), ("comfy_api.latest", latest), ("comfy_api.latest.io", io_mod)):
        sys.modules[name] = mod

    failures = []

    def check(name, condition, detail=""):
        print("  [%s] %s%s" % ("ok" if condition else "FAIL", name, (" -- " + detail) if (detail and not condition) else ""))
        if not condition:
            failures.append(name)

    import h3_ref_cache as rc
    node = install_node()
    check("patcher installed", rc.patch_minimax_h3_ref_cache() is True)

    # unchanged inputs -> one encode, then a hit that does not encode
    run(node)
    run(node)
    check("same face + prompt reuses the encode (2 calls, 1 encode)", len(ENCODED) == 1)
    check("the empty AV latent is rebuilt for every hit, and only then",
          len(EMPTY_LATENTS) == rc._STATS["hits"] and rc._STATS["hits"] == 1)

    # a different prompt is a miss
    run(node, prompt="a different face")
    check("prompt change misses", len(ENCODED) == 2)

    # two faces sharing their first pixel row must not collide (the old `img[:1]` hash)
    rng = np.random.default_rng(7)
    a = rng.random((8, 8, 3)).astype("float32")
    b = a.copy()
    b[0] = 0.25
    run(node, images={"ref": _T(a)})
    before = len(ENCODED)
    run(node, images={"ref": _T(b)})
    check("faces sharing a first row do not collide", len(ENCODED) == before + 1)

    # identical layout, different weights -> different key (the collision the old fingerprint had)
    run(node, clip=encoder())
    before = len(ENCODED)
    run(node, clip=encoder("other-layout"))
    check("same layout, different weights -> miss", len(ENCODED) == before + 1)

    # a patch another node put on the encoder invalidates the entry
    run(node, clip=encoder("layout"))
    before = len(ENCODED)
    run(node, clip=encoder("layout", patch_keys=("mixed_precision",)))
    check("encoder patches are part of the key", len(ENCODED) == before + 1)

    # LRU: with two entries, the third key evicts the first
    rc.MAX_ENTRIES = 2
    for prompt in ("p1", "p2", "p3"):
        run(node, prompt=prompt)
    before = len(ENCODED)
    run(node, prompt="p1")
    check("LRU evicts the oldest entry", len(ENCODED) == before + 1)
    run(node, prompt="p3")
    check("the most recent entry survives", len(ENCODED) == before + 1)
    rc.MAX_ENTRIES = 4

    # verify mode: an identical re-encode passes, an entry that no longer matches is dropped
    rc.VERIFY = True
    run(node, prompt="verify-me")                      # miss, cached
    before = len(ENCODED)
    run(node, prompt="verify-me")                      # hit, re-encoded, identical
    check("VERIFY re-encodes a hit and confirms it", len(ENCODED) == before + 1 and rc._STATS["verify_ok"] == 1)
    MUTATE[0] = True
    before_bad, before = rc._STATS["verify_bad"], len(ENCODED)
    out = run(node, prompt="verify-me")                # hit, re-encode differs -> upstream result wins
    check("VERIFY catches a stale entry, returns the fresh encode and drops it",
          rc._STATS["verify_bad"] == before_bad + 1 and len(ENCODED) == before + 1
          and float(out.args[0].array[0]) == 1.0 + float(sum(map(ord, "verify-me")) % 997))
    MUTATE[0] = False
    before = len(ENCODED)
    run(node, prompt="verify-me")                      # the stale entry was dropped -> a real miss
    check("the dropped entry is really gone", len(ENCODED) == before + 1)
    rc.VERIFY = False

    # report
    lines = rc._report("h3_ref_cache_test.txt")
    text = open("h3_ref_cache_test.txt").read()
    os.remove("h3_ref_cache_test.txt")
    check("report has a hit rate and a saving line",
          any("hit rate" in ln for ln in lines) and any("saving per hit" in ln for ln in lines))
    check("report counts the encodes it avoided", rc._STATS["hits"] >= 3 and rc._STATS["misses"] >= 5)
    check("report mentions the key cost", "key cost" in text)

    print("\n%s" % ("all checks passed" if not failures else "FAILURES: %s" % ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
