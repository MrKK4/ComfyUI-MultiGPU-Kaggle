"""Decode MiniMax H3 spatial tiles in batches of a known size, on every VAE copy there is.

ComfyUI v0.37.0 already batches decoder calls, but only as many tiles as it thinks will fit right
now (`_decode_tile_row`: `free // (128 MB * batch)` capped at 4). On a T4 that is running the
tensor-parallel shards and, in a face swap, a second VAE copy on the other card, free VRAM is small
by design -- so the formula lands on 1 tile per call and every tile pays its own kernel launches.
This module does three things about that, and nothing else:

  * a batch size you choose (`MMH3_VAE_TILE_BATCH`, default 2; "0" restores ComfyUI's own auto,
    "auto" means at least 2 and more if ComfyUI's formula says so);
  * a reservation before the decode (`reserve_for_tiles`, called from h3_dual_vae.py for both
    devices) asking ComfyUI's planner to make room, so the batch does not collapse for lack of free
    VRAM -- this is the same lever the TeaCache H3 optimizer pulls through `memory_used_decode`;
  * an out-of-memory fallback that drops to one tile at a time and keeps going, counting the event
    in `h3_vae_tiles.txt`.

Tiles, their order, the blend and the canvas are ComfyUI's: only the size of the decoder call
changes. The author of the same technique measured it bit-identical on an RTX 5090 and up to
2.4e-4 (well below one 8-bit pixel step) on a 3090, where the GEMM kernel selection changes with
batch size -- so expect rounding-level differences on Turing, not a different picture.

Disable with MMH3_VAE_TILES=0.
"""
import logging
import os
import time

import torch

logger = logging.getLogger("MultiGPU")
ENABLED = os.environ.get("MMH3_VAE_TILES", "1") != "0"
REQUESTED = os.environ.get("MMH3_VAE_TILE_BATCH", "2").strip().lower()
PER_TILE_MB = float(os.environ.get("MMH3_VAE_TILE_MB", "300"))
REPORT = "h3_vae_tiles.txt"
_OOM = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", RuntimeError)
_STATS = {"calls": 0, "tiles": 0, "seq_calls": 0, "oom": 0, "batch0": None, "wall_s": 0.0,
          "devices": {}, "requested": REQUESTED}
_WARNED = set()


def _once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        logger.info(message)


def requested_batch(auto):
    """The decoder-call batch size to use, given ComfyUI's own memory-based answer."""
    if REQUESTED == "0":
        return max(1, auto)
    if REQUESTED in ("auto", "at-least-auto", "max"):
        return max(2, auto)
    try:
        return max(1, int(REQUESTED))
    except ValueError:
        _once("bad_batch", "[MultiGPU] MMH3_VAE_TILE_BATCH=%r is not a number, '0' or 'auto'; "
                           "using 2" % REQUESTED)
        return 2


def reserve_for_tiles(device, tiles):
    """Ask ComfyUI's planner for room for `tiles` extra decoder tiles on `device`."""
    if not ENABLED or tiles <= 0:
        return
    try:
        import comfy.model_management as mm
        mm.free_memory(int(tiles * PER_TILE_MB * 2 ** 20), device)
    except Exception as exc:
        _once("reserve", "[MultiGPU] VAE tile reservation skipped: %s: %s" % (type(exc).__name__, exc))


def reserve_extra_tiles(device):
    """Room for the extra tiles one batched decoder call uses (batch - 1). 0 in ComfyUI's auto."""
    if REQUESTED == "0":
        return
    reserve_for_tiles(device, max(0, requested_batch(1) - 1))


def _decode_tile_row(self, z_row, x_idx, x_len):
    """ComfyUI's own tile-row batching with a chosen batch size and an OOM fallback."""
    import comfy.model_management
    free = comfy.model_management.get_free_memory(z_row.device)
    auto = int(max(1, min(4, free // (128 * 2 ** 20 * max(1, z_row.shape[0])))))
    batch = requested_batch(auto)
    if _STATS["batch0"] is None:
        _STATS["batch0"] = {"device": str(z_row.device), "comfy_auto": auto, "used": batch,
                            "free_mb": free / 2 ** 20}
        logger.info("[MultiGPU] VAE tiles: %.1f GB free on %s, ComfyUI would batch %d, using %d",
                    free / 2 ** 30, z_row.device, auto, batch)
    slices = [z_row[..., j_pos // self.vae_ratio:(j_pos + j_len) // self.vae_ratio]
              for j_pos, j_len in zip(x_idx, x_len)]
    k = 0
    while k < len(slices):
        group = slices[k:k + batch]
        tg = time.perf_counter()
        try:
            decoded = self._decode_pixels(torch.cat(group))
        except _OOM:
            if len(group) == 1:
                raise
            _STATS["oom"] += 1
            _once("oom", "[MultiGPU] VAE tile batch %d ran out of memory; decoding one tile at a "
                         "time from here" % len(group))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            batch = 1
            continue
        if len(group) == 1:
            _STATS["seq_calls"] += 1
        _STATS["calls"] += 1
        _STATS["tiles"] += len(group)
        _STATS["wall_s"] += time.perf_counter() - tg
        yield from decoded.chunk(len(group))
        k += len(group)
    _report()


def _report(path=REPORT):
    first = _STATS["batch0"] or {}
    mean = (_STATS["tiles"] / _STATS["calls"]) if _STATS["calls"] else 0.0
    lines = [
        "MiniMax H3 VAE spatial tiles (MMH3_VAE_TILES=%s, MMH3_VAE_TILE_BATCH=%s)"
        % ("1" if ENABLED else "0", _STATS["requested"]),
        "  first row: %s, ComfyUI's own auto would have been %s, used %s (%s free)"
        % (first.get("device", "n/a"), first.get("comfy_auto", "n/a"), first.get("used", "n/a"),
           ("%.2f GB" % (first["free_mb"] / 1024.0)) if first else "n/a"),
        "  decoder calls %d  tiles %d  mean tiles/call %.2f  single-tile calls %d"
        % (_STATS["calls"], _STATS["tiles"], mean, _STATS["seq_calls"]),
        "  decoder-call time %.2f s   out-of-memory fallbacks %d" % (_STATS["wall_s"], _STATS["oom"]),
    ]
    if _STATS["tiles"] <= 1:
        lines.append("  note: this run decoded at most one tile, so the batch size never mattered")
    else:
        lines.append("  tiles, their order and the blend are ComfyUI's; only the call size differs")
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
    return lines


def patch_h3_vae_tiles():
    if not ENABLED:
        logger.info("[MultiGPU] VAE tile batching disabled (MMH3_VAE_TILES=0)")
        return False
    try:
        import comfy.ldm.minimax.vae as hv
    except Exception as exc:
        logger.info("[MultiGPU] VAE tile batching unavailable: %s: %s", type(exc).__name__, exc)
        return False
    cls = getattr(hv, "MiniMaxH3VideoVAE", None)
    if cls is None or getattr(cls, "_mmh3_tiles", False):
        return bool(getattr(cls, "_mmh3_tiles", False))
    original = getattr(cls, "_decode_tile_row", None)
    if original is None:                       # older/newer core without the batched row: leave it
        logger.info("[MultiGPU] VAE tile batching: core has no _decode_tile_row, nothing to change")
        return False
    def patched(self, z_row, x_idx, x_len):
        return _decode_tile_row(self, z_row, x_idx, x_len)

    cls._decode_tile_row = patched
    cls._mmh3_tiles = True
    cls._mmh3_tile_original = original        # kept for reference, not called
    logger.info("[MultiGPU] VAE tile batching: %s tiles per decoder call on every VAE copy "
                "(MMH3_VAE_TILES=0 disables, MMH3_VAE_TILE_BATCH=0 restores ComfyUI's auto)",
                "2" if REQUESTED == "2" else REQUESTED)
    return True


_GLOBALS = {}
