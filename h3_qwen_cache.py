"""Split the H3 Qwen3-VL encode at the ViT/LLM boundary.

`MiniMaxH3ReferenceToVideo` re-encodes Qwen3-VL for every job. The existing ref cache
(`h3_ref_cache.py`) short-circuits the whole node when the prompt *and* the face are unchanged, but
changing a word in the prompt re-runs the entire 32B encode -- including the vision tower, whose
output depends only on the pixels.

The boundary is `MiniMaxQwen3VL.preprocess_embed`, the one method `SDClipModel.encode_token_weights`
calls to turn an image entry into sequence embeddings (sd1_clip.py: `self.transformer.preprocess_embed(emb, device)`).
Everything downstream of it -- the token embedding lookup, the splice, the 50 LLM layers, the
DeepStack injections -- depends on the prompt. So caching that method's `(merged, {"grid", "deepstack"})`
means a prompt tweak re-runs only the LLM.

Both routes funnel through it: the plain face image (`<Picture i>: <vision block>`) and reference
video frames (`MiniMaxQwen3VL.preprocess_embed` handles those with `minimax_video_block`). The
cache key is the image content (shape + SHA-1 of the pixels), the video-block flag, the target
device and the identity of the loaded vision tower, so nothing can collide across images, sizes,
devices or checkpoints.

Safety around aliasing: the cached tensors are clones and every return is a clone, so neither an
in-place consumer on a miss nor one on a hit can corrupt what the next job reads. Consumers today
are read-only anyway (`torch.cat` for the splice at sd1_clip.py:240, `x[mask] + deepstack[i]` at
llama.py:1031).

`h3_qwen_cache.txt` reports hits, misses, the vision-tower seconds skipped and the VRAM held.
Disable with MMH3_QWEN_CACHE=0; bound the held bytes with MMH3_QWEN_CACHE_MB (default 768).
"""
import hashlib
import logging
import os
import time
import weakref

import torch

logger = logging.getLogger("MultiGPU")
REPORT = "h3_qwen_cache.txt"
ENABLED = os.environ.get("MMH3_QWEN_CACHE", "1") != "0"
LIMIT_MB = float(os.environ.get("MMH3_QWEN_CACHE_MB", "768"))
_STATS = {"hit": 0, "miss": 0, "skip": 0, "uncacheable": 0, "skipped_ms": 0.0,
          "bytes": 0, "entries": 0}


def _clone(obj):
    """Deep-copy the vision output so nothing shared with the cache can be written through."""
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone(v) for v in obj)
    return obj


def _bytes_of(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_bytes_of(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes_of(v) for v in obj)
    return 0


_DIGEST_WARNED = []


def _digest(embed):
    """(shape, sha1 of the pixels) for one image entry; None when it cannot be hashed safely.

    Returning None means "do not cache this entry", which is the safe outcome but a silent one: a
    broken digest would look exactly like a cache that never hits. So the reason is logged once and
    counted in the report.
    """
    data = embed.get("data")
    if not torch.is_tensor(data):
        _no_digest("entry data is not a tensor")
        return None
    try:
        host = data.detach().to("cpu").contiguous()
        return (tuple(host.shape), hashlib.sha1(host.numpy().tobytes()).hexdigest())
    except Exception as exc:
        _no_digest(f"{type(exc).__name__}: {exc}")
        return None


def _weak_holder(obj):
    """A weakref to the vision tower, or None if it cannot be weakly referenced.

    Comparing identity through a weakref is what stops a cache entry from being served to a
    different checkpoint that happens to reuse the same id and the same image.
    """
    try:
        return weakref.ref(obj)
    except TypeError:
        _no_digest(f"{type(obj).__name__} is not weak-referenceable")
        return None


def _no_digest(reason):
    _STATS["uncacheable"] += 1
    if not _DIGEST_WARNED:
        _DIGEST_WARNED.append(reason)
        logger.warning("[MultiGPU] H3 Qwen3-VL vision cache cannot hash an image entry (%s); "
                       "that entry is recomputed every time", reason)


def _report():
    if not _STATS["hit"] and not _STATS["miss"]:
        return
    lines = [
        "H3 Qwen3-VL vision cache (MMH3_QWEN_CACHE=%s)" % ("1" if ENABLED else "0"),
        "vision-tower calls served from cache: %d, computed: %d, entries skipped: %d"
        % (_STATS["hit"], _STATS["miss"], _STATS["skip"]),
        "entries that could not be hashed (so never cached): %d%s"
        % (_STATS["uncacheable"], "" if not _DIGEST_WARNED else " -- " + _DIGEST_WARNED[0]),
        "vision-tower time skipped: %.1f s" % (_STATS["skipped_ms"] / 1000.0),
        "held: %.1f MB (limit %.0f MB), entries: %d" % (_STATS["bytes"] / 1e6, LIMIT_MB, _STATS["entries"]),
        "",
        "a prompt change now re-runs only the LLM: the vision tower is keyed on the pixels.",
    ]
    try:
        with open(REPORT, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def patch_h3_qwen_cache():
    global _STATS
    if not ENABLED:
        logger.info("[MultiGPU] H3 Qwen3-VL vision cache disabled (MMH3_QWEN_CACHE=0)")
        return False
    try:
        import comfy.text_encoders.minimax as te_minimax
    except Exception as exc:
        logger.info("[MultiGPU] H3 Qwen3-VL vision cache unavailable: %s: %s", type(exc).__name__, exc)
        return False
    cls = getattr(te_minimax, "MiniMaxQwen3VL", None)
    if cls is None:
        return False
    if getattr(cls, "_mmh3_qwen_cache", False):
        return True
    original = cls.preprocess_embed
    cache = {}
    order = []
    limit = int(LIMIT_MB * 1e6)

    def cached_preprocess_embed(self, embed, device):
        if embed.get("type") != "image":
            return original(self, embed, device)
        key = _digest(embed)
        if key is None:
            _STATS["skip"] += 1
            return original(self, embed, device)
        full = (key, bool(embed.get("minimax_video_block", False)), str(device), id(self.visual))
        entry = cache.get(full)
        if entry is not None and entry[0]() is self.visual:
            _STATS["hit"] += 1
            _STATS["skipped_ms"] += entry[2] * 1000.0
            _report()
            return _clone(entry[1][0]), _clone(entry[1][1])
        t0 = time.perf_counter()
        out = original(self, embed, device)
        wall = time.perf_counter() - t0
        # a new image can evict; the limit keeps a long session from pinning VRAM
        holder = _weak_holder(self.visual)
        if out[0] is not None and holder is not None:
            size = _bytes_of(out)
            if size <= limit:
                cache[full] = (holder, (_clone(out[0]), _clone(out[1])), wall)
                order.append(full)
                _STATS["bytes"] += size
                while _STATS["bytes"] > limit and order:
                    old = order.pop(0)
                    gone = cache.pop(old, None)
                    if gone is not None:
                        _STATS["bytes"] -= _bytes_of(gone[1])
        _STATS["miss"] += 1
        _STATS["entries"] = len(cache)
        _report()
        logger.info("[MultiGPU] H3 Qwen3-VL vision tower: %.2f s (%s)", wall,
                    "cached, a prompt change will reuse it" if out[0] is not None else "not cacheable")
        return out

    cls.preprocess_embed = cached_preprocess_embed
    cls._mmh3_qwen_cache = True
    logger.info("[MultiGPU] H3 Qwen3-VL vision cache enabled: prompt-only changes reuse the vision "
                "tower; MMH3_QWEN_CACHE=0 disables, MMH3_QWEN_CACHE_MB=%.0f bounds it", LIMIT_MB)
    return True
