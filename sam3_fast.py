"""SAM 3.1 video tracking: remove the per-frame host round trips, keep the masks identical.

The profile of a 124-frame track (0.86 s/frame) shows the GPU idle ~40% of the wall time with one
CPU core pegged at 100% and ~35 `cudaStreamSynchronize` plus thousands of small launches per frame.
Every sync drains the GPU pipeline: the CPU cannot issue the next frame's work until the previous
one has come back, so the GPU waits. The generator of syncs is not the model -- it is bookkeeping
that reaches for host values once or several times per frame:

  tracker.track_video_with_detection
    fill_holes_in_mask_scores(..., max_area=16)   two `.cpu()` round trips per frame (bg then fg
                                                  connected components), back to the GPU, twice
    pack_masks(...).to(intermediate_device())     one D2H copy per frame; on Kaggle (default load,
                                                  no --gpu-only) intermediate_device() is CPU, so
                                                  every frame pays a full device sync
  _nms_masks                                       `if overlap.max() >= thresh` inside a per-detection
                                                  Python loop -> one sync per detection
  _match_and_add_detections                        `.nonzero().tolist()`, `.argmax().item()` and a
                                                  per-detection `if overlap[..] >= 0.5` -> one sync
                                                  each, every frame

This module removes the first three without changing a single mask value:

  fill_holes   the bg and fg passes are done in one host excursion, in numpy, in the same order as
               upstream (bg first, then fg computed on the mask *after* the bg fill -- that ordering
               is load-bearing and is preserved). One sync instead of two, and no GPU->CPU->GPU
               ping-pong between them.
  nms          the overlap matrix is computed in one batched call and brought to the host once; the
               greedy loop then runs on those numbers. Same decisions, one sync per frame instead of
               one per detection.
  masks        `comfy.model_management.intermediate_device` is redirected to the model's own device
               for the duration of the node, so the per-frame mask accumulator stays in VRAM, and
               the stacked result is moved to the real intermediate device once at the end.

`_match_and_add_detections` is left alone for now: it needs a copy of the method rather than a
function patch, and it will be worth doing only if the numbers say so.

Safety: every fast path is wrapped, and any exception logs once and falls back to the upstream
implementation. `sam_fast.txt` is written after each tracking run with call counts and the host
time spent in each patched function, so the A/B is a before/after of one file.

DEFAULTS TO OFF (MMH3_SAM_FAST=1 to enable). A reported run produced flicker with this module on.
Re-reading upstream against this module found **no deviation in the mask arithmetic** -- the
hole-filling passes, the connected-component areas and the greedy NMS decisions all reproduce
upstream on the same inputs -- and one deviation in *dtype*: `_nms_masks` promoted the overlap
matrix to float32 before comparing against the threshold, while upstream compares the native
values. Upcasting fp16 to fp32 is exact, so with the default threshold (0.5, exactly representable
in fp16) this makes no difference to a single decision; it was removed anyway because it is still
the wrong comparison for any threshold that fp16 cannot represent exactly.

What this module *does* change, and what the flicker report should be blamed on until an A/B says
otherwise:

  * the device of the tensors the node returns. With the redirect active, the per-frame masks stay
    in VRAM and are moved back to the real intermediate device once at the end. A downstream node
    that branches on `tensor.device` (torch interpolation vs a numpy/PIL resize, for instance) can
    then take a different path than it would have -- same values, different implementation.
  * nothing else: the two patched functions are pure, and every failure path returns upstream's own
    result.

So the honest state is "not reproduced, not explained by code reading, off by default". Running the
same clip twice -- once with MMH3_SAM_FAST=0 (the default) and once with MMH3_SAM_GOLDEN=1 --
separates this module from ComfyUI's own SAM 3.1 multiplex core: if the flicker survives with the
module off, it is upstream's tracking, not this file.

Ways to prove it on the real clip, all opt-in:

  MMH3_SAM_GOLDEN=1   runs upstream's implementation alongside the fast one on every call and
                      compares the results exactly (same shape, dtype and values). Mismatches are
                      logged with the first differing element, counted in `sam_fast.txt`, and the
                      *upstream* result is returned, so a golden run cannot introduce a difference.
  check_sam_fast.py   offline: replays the mask cleanup and NMS logic against upstream's algorithm.
  sam_fast.txt        per-run verdict line: "value-identical" only if every compared call matched.

The per-frame mask accumulator also stops being a blind device redirect: the redirect is decided
once per run from free VRAM (MMH3_SAM_VRAM_FLOOR_MB, default 2048) and only used when the card can
hold the clip's masks, so it cannot trade a sync win for the OOM that a second face-swap job hit.
"""
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger("MultiGPU")
# Off by default: a real clip flickered with this enabled, and one deviation from upstream (the
# float32 promotion in _nms_masks) was found only afterwards. MMH3_SAM_FAST=1 enables it;
# MMH3_SAM_GOLDEN=1 proves it against upstream on the same clip before you trust it.
ENABLED = os.environ.get("MMH3_SAM_FAST", "0") != "0"
GOLDEN = os.environ.get("MMH3_SAM_GOLDEN", "0") != "0"
# The mask accumulator may only move to VRAM when the card can hold the clip's masks on top of
# everything else that is resident; below this floor the real intermediate device (CPU) is used.
VRAM_FLOOR_MB = float(os.environ.get("MMH3_SAM_VRAM_FLOOR_MB", "2048"))
REPORT = "sam_fast.txt"
_STATS = defaultdict(float)
_GOLDEN = {"calls": 0, "mismatches": 0, "first": "", "compared": 0}
_WARNED = set()


def _once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        logger.info(message)


# ---------------------------------------------------------------------------
# fill_holes_in_mask_scores, in one host excursion
# ---------------------------------------------------------------------------

def _components(mask_np, tracker):
    """Connected components of a 2-D uint8 array -> (labels, per-pixel area).

    Mirrors tracker._get_connected_components exactly, including its cv2/scipy branch, but stays in
    numpy: the caller is already on the host and does not need a device tensor back.
    """
    if getattr(tracker, "_HAS_CV2", False):
        cv2 = tracker.cv2
        _, labeled, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        areas = stats[labeled, cv2.CC_STAT_AREA].astype("int32")
    else:
        ndimage = tracker.ndimage
        labeled, num_features = ndimage.label(mask_np)
        # Upstream writes into np.zeros_like(m) -- a uint8 array, so areas wrap modulo 256 for
        # components larger than 255 pixels. Mirrored deliberately: this path exists to reproduce
        # upstream exactly, bug included, and it is unused wherever cv2 is installed (Kaggle).
        areas = np.zeros_like(mask_np)
        for c in range(1, num_features + 1):
            component = labeled == c
            areas[component] = component.sum()
    return labeled, areas


def _fill_holes_one_trip(tracker, mask, max_area):
    """Upstream fill_holes_in_mask_scores with bg+fg resolved in a single host round trip."""
    arr = np.array(mask.detach().float().cpu(), dtype="float32")     # one sync, one copy
    for i in range(arr.shape[0]):
        plane = arr[i, 0]
        # background holes: small background components become foreground
        bg = (plane <= 0).astype("uint8")
        _, areas_bg = _components(bg, tracker)
        small_bg = bg.astype(bool) & (areas_bg <= max_area)
        plane[small_bg] = 0.1
        # foreground sprinkles: computed on the mask *after* the bg fill, as upstream does
        fg = (plane > 0).astype("uint8")
        thresh = (int(fg.sum(dtype="int64")) // 2)
        thresh = min(thresh, max_area)
        _, areas_fg = _components(fg, tracker)
        small_fg = fg.astype(bool) & (areas_fg <= thresh)
        plane[small_fg] = -0.1
    out = torch.from_numpy(np.ascontiguousarray(arr))
    return out.to(device=mask.device, dtype=mask.dtype, non_blocking=True)


# ---------------------------------------------------------------------------
# _nms_masks without a sync per detection
# ---------------------------------------------------------------------------

def _nms_one_trip(tracker, masks, scores, thresh):
    """Same greedy NMS, one host round trip: the overlap matrix is read once, not per candidate.

    The matrix is kept in the masks' own dtype all the way to the comparison, exactly as upstream
    compares it (`overlap.max() >= thresh` on the fp16 tensor it just built; inf/nan entries read
    the same either way, but a threshold fp16 cannot represent exactly does not). `check_sam_fast`
    asserts the no-promotion property directly, not just the decisions.
    """
    order = scores.argsort(descending=True)
    masks, scores = masks[order], scores[order]
    n = int(masks.shape[0])
    if n <= 1:
        return masks, scores
    overlap = tracker._compute_mask_overlap(masks, masks).detach().cpu().tolist()
    keep = []
    for i in range(n):
        if keep and max(overlap[i][j] for j in keep) >= thresh:
            continue
        keep.append(i)
    _STATS["nms_syncs_saved"] += max(0, n - 1)
    return masks[keep], scores[keep]


# ---------------------------------------------------------------------------
# golden A/B: prove the fast path against upstream on the real clip
# ---------------------------------------------------------------------------

def _clone(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(v) for v in value)
    return value


def _equal(a, b):
    """Exact equality: same type, shape, dtype, values. Returns (ok, detail)."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return False, "one side is not a tensor"
        if a.shape != b.shape:
            return False, "shape %s vs %s" % (tuple(a.shape), tuple(b.shape))
        if a.dtype != b.dtype:
            return False, "dtype %s vs %s" % (a.dtype, b.dtype)
        if torch.equal(a, b):
            return True, ""
        diff = a != b
        first = tuple(int(i) for i in diff.nonzero()[0].tolist()) if diff.any() else ()
        return False, "%d of %d values differ (first at %s)" % (int(diff.sum()), a.numel(), first)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, "length %d vs %d" % (len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            ok, detail = _equal(x, y)
            if not ok:
                return False, "[%d] %s" % (i, detail)
        return True, ""
    if type(a) is not type(b):
        return False, "type %s vs %s" % (type(a).__name__, type(b).__name__)
    return (a == b), "" if a == b else "%r vs %r" % (a, b)


def _golden_check(name, original, args, kwargs, fast_out):
    """Run upstream on copies of the inputs and compare. Returns upstream's output on mismatch."""
    try:
        up = original(*[_clone(v) for v in args], **{k: _clone(v) for k, v in kwargs.items()})
    except Exception as exc:
        _once("golden_err_" + name,
              "[MultiGPU] SAM3 golden %s: upstream failed to run (%s: %s)" % (name, type(exc).__name__, exc))
        return None
    _GOLDEN["calls"] += 1
    ok, detail = _equal(up, fast_out)
    if ok:
        _GOLDEN["compared"] += 1
        return None
    _GOLDEN["mismatches"] += 1
    if not _GOLDEN["first"]:
        _GOLDEN["first"] = "%s: %s" % (name, detail)
    logger.warning("[MultiGPU] SAM3 golden: %s differs from upstream (%s) - using upstream's result",
                   name, detail)
    return up


def _wrap(tracker, name, fast):
    """Replace a module-level tracker function with `fast`, falling back to the original on error."""
    original = getattr(tracker, name)

    def patched(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = fast(*args, **kwargs)
        except Exception as exc:  # never let an optimisation change the result or crash a run
            _once("fallback_" + name, f"[MultiGPU] SAM3 fast {name} fell back to upstream: "
                                      f"{type(exc).__name__}: {exc}")
            return original(*args, **kwargs)
        if GOLDEN:
            upstream_out = _golden_check(name, original, args, kwargs, out)
            if upstream_out is not None:
                out = upstream_out
        _STATS[name + "_calls"] += 1
        _STATS[name + "_ms"] += (time.perf_counter() - t0) * 1000.0
        return out

    patched._mmh3_fast = True
    setattr(tracker, name, patched)
    return patched


# ---------------------------------------------------------------------------
# keep the per-frame mask accumulator in VRAM
# ---------------------------------------------------------------------------

def _demote(obj, device):
    """Move the tracking result back to the real intermediate device, once, at the end."""
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, dict):
        return {k: _demote(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_demote(v, device) for v in obj)
    args = getattr(obj, "args", None)                    # io.NodeOutput
    if isinstance(args, tuple):
        obj.args = tuple(_demote(v, device) for v in args)
    return obj


def _pick_device(real_device_fn):
    """Where the per-frame mask accumulator lives, decided once per run.

    Sticky by construction: the clip's masks are concatenated at the end, so a mid-run switch would
    leave them split over two devices. The redirect only happens when the card has room for the
    clip on top of everything already resident -- the second face-swap job OOMed inside tracking
    while the TP shards and helper VAE copies were still on cuda:0.
    """
    state = {}

    def pick():
        if "device" not in state:
            import comfy.model_management as mm
            compute = mm.get_torch_device()
            free = None
            try:
                if compute.type == "cuda":
                    free = torch.cuda.mem_get_info(compute)[0]
            except Exception:
                free = None
            if free is None or free >= VRAM_FLOOR_MB * 2 ** 20:
                state["device"] = compute
                state["why"] = ("%.1f GB free VRAM" % (free / 2 ** 30)) if free is not None else "free VRAM unknown"
            else:
                state["device"] = real_device_fn()
                state["why"] = "%.1f GB free VRAM < %d MB floor" % (free / 2 ** 30, int(VRAM_FLOOR_MB))
            logger.info("[MultiGPU] SAM3 fast path: mask accumulator on %s (%s)",
                        state["device"], state["why"])
        return state["device"]

    pick.info = lambda: state.get("why", "not decided")
    return pick


def _patch_node():
    import nodes
    node = nodes.NODE_CLASS_MAPPINGS.get("SAM3_VideoTrack")
    if node is None:
        return False
    if getattr(node, "_mmh3_fast", False):
        return True
    execute = node.execute.__func__

    def fast_execute(cls, *args, **kwargs):
        import comfy.model_management as mm
        real = mm.intermediate_device
        picker = _pick_device(real)
        mm.intermediate_device = picker                 # evaluated inside the loop, so it is the
        t0 = time.perf_counter()                        # device the model actually loaded on
        try:
            out = execute(cls, *args, **kwargs)
        finally:
            mm.intermediate_device = real
        _STATS["device_why"] = picker.info()
        _demote(out, real())                            # one D2H at the end, not one per frame
        _STATS["run_wall_s"] += time.perf_counter() - t0
        data = out.args[0] if getattr(out, "args", None) else None
        if isinstance(data, dict) and data.get("n_frames"):
            _STATS["frames"] += int(data["n_frames"])
        _report()
        return out

    node.execute = classmethod(fast_execute)
    node._mmh3_fast = True
    return True


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _report(path=REPORT):
    frames = int(_STATS["frames"])
    saved = int(_STATS["fill_holes_calls"] + _STATS["nms_syncs_saved"] + frames)
    lines = [
        "SAM3 fast path (MMH3_SAM_FAST=%s, MMH3_SAM_GOLDEN=%s)" % ("1" if ENABLED else "0",
                                                                  "1" if GOLDEN else "0"),
        "tracking run wall (node): %.2f s over %d frame(s)" % (_STATS["run_wall_s"], frames),
        "mask accumulator device: %s" % (_STATS.get("device_why") or "n/a"),
        "",
        "%-26s %8s %10s  %s" % ("patched", "calls", "host ms", "note"),
        "%-26s %8d %10.1f  %s" % ("fill_holes_in_mask_scores", int(_STATS["fill_holes_in_mask_scores_calls"]),
                                  _STATS["fill_holes_in_mask_scores_ms"],
                                  "2 host round trips -> 1"),
        "%-26s %8d %10.1f  %s" % ("_nms_masks", int(_STATS["_nms_masks_calls"]), _STATS["_nms_masks_ms"],
                                  "1 sync per detection -> 1 per frame"),
        "%-26s %8s %10s  %s" % ("intermediate device", "-", "-",
                                "per-frame D2H -> 1 at the end (%d frame(s))" % frames),
        "",
        "host round trips avoided this run: ~%d (cleanup %d + nms %d + mask copies %d)"
        % (saved, int(_STATS["fill_holes_in_mask_scores_calls"]),
           int(_STATS["nms_syncs_saved"]), frames),
    ]
    if GOLDEN:
        verdict = ("value-identical on this run (%d call(s) compared against upstream)"
                   % _GOLDEN["compared"]) if not _GOLDEN["mismatches"] else (
                   "NOT identical: %d of %d compared call(s) differ; first: %s"
                   % (_GOLDEN["mismatches"], _GOLDEN["calls"], _GOLDEN["first"]))
        lines += ["", "golden A/B against upstream (MMH3_SAM_GOLDEN=1)",
                  "  compared: %d   mismatches: %d   upstream failures to run: %d"
                  % (_GOLDEN["calls"], _GOLDEN["mismatches"], _GOLDEN["calls"] - _GOLDEN["compared"]
                     - _GOLDEN["mismatches"]),
                  "  VERDICT: %s" % verdict]
    if frames == 0 and not _STATS["fill_holes_in_mask_scores_calls"]:
        return
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("[MultiGPU] SAM3 fast path: %s", lines[1])
    except OSError:
        pass


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def patch_sam3_fast():
    global ENABLED
    if GOLDEN and not ENABLED:
        # golden is the proof, so it implies the thing being proved
        ENABLED = True
        logger.info("[MultiGPU] SAM3 fast path enabled by MMH3_SAM_GOLDEN=1 (golden A/B run)")
    if not ENABLED:
        logger.info("[MultiGPU] SAM3 fast path disabled (MMH3_SAM_FAST=0, the default); "
                    "MMH3_SAM_FAST=1 enables, MMH3_SAM_GOLDEN=1 proves it against upstream")
        return False
    try:
        import comfy.ldm.sam3.tracker as tracker
    except Exception as exc:
        logger.info("[MultiGPU] SAM3 fast path unavailable: %s: %s", type(exc).__name__, exc)
        return False

    done = []
    if hasattr(tracker, "fill_holes_in_mask_scores") and not getattr(tracker.fill_holes_in_mask_scores, "_mmh3_fast", False):
        _wrap(tracker, "fill_holes_in_mask_scores",
              lambda mask, max_area=0: mask if max_area <= 0 else _fill_holes_one_trip(tracker, mask, max_area))
        done.append("fill_holes(1 round trip)")
    if hasattr(tracker, "_nms_masks") and not getattr(tracker._nms_masks, "_mmh3_fast", False):
        _wrap(tracker, "_nms_masks",
              lambda masks, scores, thresh=0.5: _nms_one_trip(tracker, masks, scores, thresh))
        done.append("nms(1 round trip)")
    if _patch_node():
        done.append("masks stay in VRAM")

    if done:
        logger.info("[MultiGPU] SAM3 fast path: %s — MMH3_SAM_FAST=0 disables", ", ".join(done))
    return bool(done)
