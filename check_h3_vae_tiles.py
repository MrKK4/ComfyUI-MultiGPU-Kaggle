"""Offline check for h3_vae_tiles.py: the tiles, their order and their values must not change.

Batching changes only the size of the decoder call, so the things worth pinning are the ones that
would corrupt a frame silently:

  * every tile comes out exactly once, in ComfyUI's order, with its own values -- a dropped or
    duplicated tile is a seam or a repeated strip in the decoded video;
  * an out-of-memory batch falls back to one tile at a time and still yields all of them;
  * the reservation asks the planner for (batch - 1) tiles and for nothing when batching is off;
  * the report says what the run actually did.

    python check_h3_vae_tiles.py        # exits 1 on any failure
"""
import sys
import types

import numpy as np


class _T:
    """Just enough tensor: slicing, cat, chunk, shape -- what _decode_tile_row touches."""

    def __init__(self, array, tag=None):
        self.array = np.asarray(array)
        self.tag = tag
        self.device = "cuda:0"

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, idx):
        return _T(self.array[idx])

    def numel(self):
        return int(self.array.size)

    def chunk(self, n):
        return [_T(part) for part in np.array_split(self.array, n, axis=0)]


def cat(tensors, dim=0):
    return _T(np.concatenate([t.array for t in tensors], axis=dim))


def install_fakes():
    torch = types.ModuleType("torch")
    torch.cat = cat
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=lambda: None)
    sys.modules["torch"] = torch
    return torch


class FakeTileVAE:
    """Decodes by tagging every element of a tile with the tile's own x offset."""

    vae_ratio = 8

    def __init__(self, fail_over_batch=1):
        self.calls = []
        self.fail_over_batch = fail_over_batch

    def _decode_pixels(self, stacked):
        n = stacked.shape[0]
        self.calls.append(n)
        if n > self.fail_over_batch:
            raise MemoryError("out of memory")
        return _T(stacked.array.copy())          # identity: the caller can see the tile values


def main():
    install_fakes()
    asked = []
    mm = types.ModuleType("comfy.model_management")
    mm.free_memory = lambda need, device: asked.append((need, str(device)))
    free_mb = [2048]
    mm.get_free_memory = lambda device: free_mb[0] * 2 ** 20
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    import h3_vae_tiles as vt

    failures = []

    def check(name, condition, detail=""):
        print("  [%s] %s%s" % ("ok" if condition else "FAIL", name, (" -- " + str(detail)) if detail and not condition else ""))
        if not condition:
            failures.append(name)

    check("requested_batch: explicit", vt.requested_batch(1) == 2 and vt.requested_batch(4) == 2)
    check("requested_batch: 'auto' is at least 2", vt.requested_batch(1) == 2)
    vt.REQUESTED = "0"
    check("requested_batch: '0' is ComfyUI's own answer", vt.requested_batch(1) == 1 and vt.requested_batch(4) == 4)
    vt.REQUESTED = "3"
    check("requested_batch: 3", vt.requested_batch(1) == 3)
    vt.REQUESTED = "auto"
    check("requested_batch: auto with a big card", vt.requested_batch(4) == 4)
    vt.REQUESTED = "2"

    # tiles: latent [B, C, T, H, W] with W = 3 tile columns; every tile carries its own value, so a
    # duplicate or a reorder is visible in the yielded order
    z = np.zeros((1, 4, 1, 2, 3), dtype="float32")
    for j in range(3):
        z[0, :, :, :, j] = j + 1
    x3, l3 = [0, 8, 16], [8, 8, 8]
    vae = FakeTileVAE(fail_over_batch=99)
    tiles = list(vt._decode_tile_row(vae, _T(z), [0, 8], [8, 8]))
    check("one tile out per tile in, in order", [float(t.array.flat[0]) for t in tiles] == [1.0, 2.0],
          [float(t.array.flat[0]) for t in tiles])
    check("two tiles go through the decoder in one call", vae.calls == [2], vae.calls)

    vae = FakeTileVAE(fail_over_batch=99)
    tiles3 = list(vt._decode_tile_row(vae, _T(z), x3, l3))
    check("a row longer than the batch is split without losing a tile",
          [float(t.array.flat[0]) for t in tiles3] == [1.0, 2.0, 3.0] and vae.calls == [2, 1], vae.calls)

    # out of memory at batch 2 -> falls back to one at a time, all tiles still come out
    vae = FakeTileVAE(fail_over_batch=1)
    tiles_oom = list(vt._decode_tile_row(vae, _T(z), x3, l3))
    check("OOM falls back to single tiles",
          [float(t.array.flat[0]) for t in tiles_oom] == [1.0, 2.0, 3.0] and vae.calls == [2, 1, 1, 1], vae.calls)
    check("OOM is counted", vt._STATS["oom"] == 1)

    # the case this module exists for: a card that looks full, where ComfyUI's own formula yields 1
    free_mb[0] = 100
    vt.REQUESTED = "0"
    vae = FakeTileVAE(fail_over_batch=99)
    tiles_seq = list(vt._decode_tile_row(vae, _T(z), x3, l3))
    check("ComfyUI's own formula batches 1 tile when the card looks full", vae.calls == [1, 1, 1], vae.calls)
    vt.REQUESTED = "2"
    vt._STATS["batch0"] = None
    vae = FakeTileVAE(fail_over_batch=99)
    tiles2 = list(vt._decode_tile_row(vae, _T(z), x3, l3))
    check("the module batches 2 anyway, and says so", vae.calls == [2, 1]
          and vt._STATS["batch0"]["comfy_auto"] == 1 and vt._STATS["batch0"]["used"] == 2,
          (vae.calls, vt._STATS["batch0"]))
    check("the tiles are identical either way",
          [t.shape for t in tiles2] == [t.shape for t in tiles_seq]
          and [float(t.array.flat[0]) for t in tiles2] == [float(t.array.flat[0]) for t in tiles_seq])
    free_mb[0] = 2048

    # reservation: (batch - 1) tiles, and nothing at all when the batch is ComfyUI's
    vt.reserve_extra_tiles("cuda:1")
    check("reserves one extra tile at batch 2", asked and asked[-1][0] == int(vt.PER_TILE_MB * 2 ** 20)
          and asked[-1][1] == "cuda:1", asked)
    vt.REQUESTED = "0"
    before = len(asked)
    vt.reserve_extra_tiles("cuda:1")
    check("reserves nothing when batching is ComfyUI's", len(asked) == before)
    vt.REQUESTED = "2"

    # report
    lines = vt._report("h3_vae_tiles_test.txt")
    text = open("h3_vae_tiles_test.txt").read()
    import os
    os.remove("h3_vae_tiles_test.txt")
    check("report says what ComfyUI would have done and what was used",
          any("ComfyUI's own auto" in ln for ln in lines) and any("mean tiles/call" in ln for ln in lines))
    check("report counts decoder calls and tiles", "decoder calls" in text and "out-of-memory" in text)

    print("\n%s" % ("all checks passed" if not failures else "FAILURES: %s" % ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
