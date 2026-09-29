"""Reuse the MiniMax H3 Ref2VA prompt + reference-image encode across jobs.

MiniMaxH3ReferenceToVideo also takes the output width/height/length, so ComfyUI re-runs the
Qwen3-VL encode (~60 s on 2x T4) for every new driving video in a face swap even when the prompt
and face are unchanged. The conditioning only depends on the prompt, the reference images, the
ref_image_size mode (and the output size for "match"), so it is cached on exactly those plus the
loaded text encoder / VAE; the empty AV latent is rebuilt for the new size. Jobs with reference
videos or audio are not cached. Disable with MMH3_REF_CACHE=0.

What the key is, and why:

  prompt, ref_image_size, output size    the conditioning arguments themselves
  every reference image, in full         sha1 of the whole tensor, not of its first row: two faces
                                         shot against the same background share those bytes
  text encoder                           ComfyUI re-runs the loaders whenever another workflow
                                         (face swap step 1) ran in between, so the same file comes
                                         back as a new object. Keyed on the weights' names/shapes/
                                         dtypes *plus* the first 4 kB of four spread-out tensors, so
                                         two fine-tunes with an identical layout cannot collide
                                         without hashing 14 GB per lookup. The digest is computed
                                         once per encoder object and reused.
  encoder patches                        the patch keys another node put on the patcher (mixed
                                         precision, LoRA, a cache in front of the tower): a patched
                                         encoder must not reuse an unpatched encode
  VAE                                    same treatment, for the empty-latent path

MMH3_REF_CACHE_ENTRIES (default 4) keeps the most recent encodes, LRU, so alternating faces do not
thrash. `h3_ref_cache.txt` is written after every call: hits, misses, the time actually spent on
each, and the saving that follows from the difference. MMH3_REF_CACHE_VERIFY=1 re-encodes on the
first hit and compares the conditioning exactly (one extra ~60 s encode, then it trusts the cache):
the only test that proves a hit returns what a miss would have returned.
"""
import collections
import hashlib
import logging
import os
import time
import weakref

import torch

logger = logging.getLogger("MultiGPU")

REPORT = "h3_ref_cache.txt"
MAX_ENTRIES = max(1, int(os.environ.get("MMH3_REF_CACHE_ENTRIES", "4")))
VERIFY = os.environ.get("MMH3_REF_CACHE_VERIFY", "0") != "0"
_SAMPLE_TENSORS = 4
_SAMPLE_BYTES = 4096
_STATS = collections.OrderedDict([
    ("calls", 0), ("hits", 0), ("misses", 0), ("hit_s", 0.0), ("miss_s", 0.0),
    ("fp_calls", 0), ("fp_s", 0.0), ("fp_cached", 0),
    ("verify_ok", 0), ("verify_bad", 0), ("verify_detail", ""),
])


def _weak_holder(obj):
    try:
        return weakref.ref(obj)
    except TypeError:
        return None


def _sample(sd, key):
    """sha1 of the first 4 kB of one weight: cheap content identity for a layout fingerprint."""
    v = sd[key]
    if not torch.is_tensor(v):
        return key
    flat = v.detach().reshape(-1)[:_SAMPLE_BYTES].float().cpu().numpy().tobytes()
    return "%s:%s" % (key, hashlib.sha1(flat).hexdigest()[:16])


def _fingerprint(obj, attr):
    """Layout + a content sample of the weights, plus whatever another node patched onto them.

    Cost is ~1000 dict entries of shape/dtype plus four 4 kB reads, not a 14 GB hash, and it is
    computed once per model object (the encoder survives across jobs only as long as ComfyUI keeps
    it loaded, which is exactly when the digest is worth keeping).
    """
    model = getattr(obj, attr, None) if obj is not None else None
    if model is None:
        return None
    holder = getattr(model, "_mmh3_ref_fp", None)
    if holder is not None and holder[0] is not None and holder[0]() is model:
        _STATS["fp_cached"] += 1
        return holder[1]
    t0 = time.perf_counter()
    sd = model.state_dict()
    keys = list(sd)
    picks = sorted({0, len(keys) // 3, (2 * len(keys)) // 3, len(keys) - 1}) if keys else []
    layout = hashlib.sha1(repr([(k, tuple(v.shape), str(v.dtype)) for k, v in sd.items()]).encode()).hexdigest()[:32]
    value = (layout, tuple(_sample(sd, keys[i]) for i in picks if i < len(keys)))
    _STATS["fp_calls"] += 1
    _STATS["fp_s"] += time.perf_counter() - t0
    try:
        model._mmh3_ref_fp = (_weak_holder(model), value)
    except Exception:
        pass
    return value


def _patch_key(clip):
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        return None
    names = []
    for attr in ("patches", "object_patches"):
        d = getattr(patcher, attr, None)
        if isinstance(d, dict):
            names += [str(k) for k in d.keys()]
    return hashlib.sha1(repr(sorted(names)).encode()).hexdigest()[:16]


def _image_key(images):
    out = []
    for img in images:
        t0 = time.perf_counter()
        digest = hashlib.sha1(img.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        _STATS["fp_s"] += time.perf_counter() - t0
        out.append((tuple(img.shape), str(img.dtype), digest))
    return tuple(out)


def _equal(a, b):
    """Exact equality over the conditioning structure. Returns (ok, detail)."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return False, "one side is not a tensor"
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, "shape/dtype %s %s vs %s %s" % (tuple(a.shape), a.dtype, tuple(b.shape), b.dtype)
        if torch.equal(a, b):
            return True, ""
        diff = a != b
        return False, "%d of %d values differ" % (int(diff.sum()), a.numel())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, "length %d vs %d" % (len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            ok, detail = _equal(x, y)
            if not ok:
                return False, "[%d] %s" % (i, detail)
        return True, ""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, "keys differ"
        for k in a:
            ok, detail = _equal(a[k], b[k])
            if not ok:
                return False, "[%s] %s" % (k, detail)
        return True, ""
    return (a == b), "" if a == b else "%r vs %r" % (a, b)


def _report(path=REPORT):
    hits = _STATS["hits"]
    mean_hit = (_STATS["hit_s"] / hits) if hits else 0.0
    mean_miss = (_STATS["miss_s"] / _STATS["misses"]) if _STATS["misses"] else 0.0
    per_hit = max(0.0, mean_miss - mean_hit) if (hits and _STATS["misses"]) else 0.0
    lines = [
        "MiniMax H3 Ref2VA cache (MMH3_REF_CACHE=%s, entries=%d, verify=%s)"
        % (os.environ.get("MMH3_REF_CACHE", "1"), MAX_ENTRIES, "1" if VERIFY else "0"),
        "  calls %d   hits %d   misses %d   hit rate %s"
        % (_STATS["calls"], hits, _STATS["misses"],
           ("%.0f%%" % (100.0 * hits / _STATS["calls"])) if _STATS["calls"] else "n/a"),
        "  time on hits: %s avg   |   time on misses: %s avg"
        % (("%.2f s" % mean_hit) if hits else "n/a", ("%.2f s" % mean_miss) if _STATS["misses"] else "n/a"),
        "  observed saving per hit vs the miss path: %s   (this session: %.1f s)"
        % (("%.2f s" % per_hit) if per_hit else "n/a (needs one hit and one miss)", per_hit * hits),
        "  key cost: %d fingerprint(s) in %.1f ms (%d reused); image hashing included above"
        % (_STATS["fp_calls"], _STATS["fp_s"] * 1000.0, _STATS["fp_cached"]),
    ]
    if VERIFY:
        lines.append("  VERIFY: %d hit(s) re-encoded and compared exactly, %d differed%s"
                     % (_STATS["verify_ok"], _STATS["verify_bad"],
                        (": " + _STATS["verify_detail"]) if _STATS["verify_detail"] else ""))
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
    return lines


def patch_minimax_h3_ref_cache():
    if os.environ.get("MMH3_REF_CACHE", "1") == "0":
        return False
    import sys
    import nodes
    from comfy_api.latest import io
    # ComfyUI loads its built-in node files under an internal module name, so patch the registered class
    node = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ReferenceToVideo")
    if node is None:
        return False
    nm = sys.modules[node.__module__]
    if getattr(node, "_mmh3_ref_cache", False):
        return True
    execute = node.execute.__func__
    cache = collections.OrderedDict()          # key -> conditioning; LRU across the last few faces

    def cached_execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                       ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        args = (cls, clip, prompt, width, height, length, ref_image_size, vae, audio_vae,
                ref_images, ref_videos, ref_video_audios, ref_audios)
        if ref_videos or ref_video_audios or ref_audios:
            return execute(*args)
        images = [img for img in (ref_images or {}).values() if img is not None]
        key = (prompt, ref_image_size, (width, height) if ref_image_size == "match" else None,
               _image_key(images), _fingerprint(clip, "cond_stage_model"), _fingerprint(vae, "first_stage_model"),
               _patch_key(clip))
        _STATS["calls"] += 1
        t_call = time.perf_counter()
        cached = cache.get(key)
        if cached is not None:
            latent, _ = nm._empty_av_latent(width, height, length)
            out = io.NodeOutput(cached, latent)
            _STATS["hits"] += 1
            _STATS["hit_s"] += time.perf_counter() - t_call    # before any verification re-encode
            if VERIFY:
                fresh = execute(*args)
                ok, detail = _equal(fresh.args[0], cached)
                if ok:
                    _STATS["verify_ok"] += 1
                else:
                    _STATS["verify_bad"] += 1
                    if not _STATS["verify_detail"]:
                        _STATS["verify_detail"] = detail
                    cache.pop(key, None)
                    logger.warning("[MultiGPU] H3 Ref2VA cache: a hit did NOT match a fresh encode "
                                   "(%s); dropping it", detail)
                    _report()
                    return fresh
            cache.move_to_end(key)
            logger.info("[MultiGPU] H3 Ref2VA: reusing the cached prompt + reference encode "
                        "(%d/%d hits)", _STATS["hits"], _STATS["calls"])
            _report()
            return out
        t0 = time.perf_counter()
        out = execute(*args)
        _STATS["misses"] += 1
        _STATS["miss_s"] += time.perf_counter() - t0
        cache[key] = out.args[0]
        cache.move_to_end(key)
        while len(cache) > MAX_ENTRIES:
            cache.popitem(last=False)
        _report()
        return out

    node.execute = classmethod(cached_execute)
    node._mmh3_ref_cache = True
    logger.info("[MultiGPU] H3 Ref2VA prompt + reference encode cache enabled (%d entries, LRU, "
                "verify=%s); MMH3_REF_CACHE=0 disables", MAX_ENTRIES, "on" if VERIFY else "off")
    return True
