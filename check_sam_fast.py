"""Offline equivalence check for sam3_fast.py's mask cleanup and NMS.

The GPU paths cannot run here (no torch), but the two things that could silently change a mask are
pure logic: the order of the background/foreground passes and the greedy NMS decisions. Both are
reproduced against upstream's algorithm on random masks with injected sprinkles and holes. The
fake torch only needs the four methods sam3_fast touches, and the stub cv2 carries a straightforward
BFS labeller as the oracle.

    python check_sam_fast.py        # exits 1 on any mismatch
"""
import sys
from collections import deque

import numpy as np

CC_STAT_AREA = 4


class _Stats:
    """Area per label, indexing-compatible with cv2.connectedComponentsWithStats' stats array."""

    def __init__(self, labeled, n):
        self._areas = np.bincount(np.asarray(labeled).ravel(), minlength=n)

    def __getitem__(self, idx):
        labels, col = idx          # cv2 semantics: stats[labeled, CC_STAT_AREA] -> per-pixel area
        assert col == CC_STAT_AREA
        return self._areas[labels]


class fake_cv2:
    CC_STAT_AREA = CC_STAT_AREA

    @staticmethod
    def connectedComponentsWithStats(m, connectivity=8):
        m = np.asarray(m)
        h, w = m.shape
        labeled = np.zeros((h, w), dtype="int32")
        n = 0
        if connectivity == 8:
            nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for y0 in range(h):
            for x0 in range(w):
                if m[y0, x0] and labeled[y0, x0] == 0:
                    n += 1
                    q = deque([(y0, x0)])
                    labeled[y0, x0] = n
                    while q:
                        y, x = q.popleft()
                        for dy, dx in nbrs:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and labeled[ny, nx] == 0:
                                labeled[ny, nx] = n
                                q.append((ny, nx))
        return n + 1, labeled, _Stats(labeled, n + 1), None


class _FakeTensor:
    """Just enough tensor for sam3_fast: detach/float/cpu/to, numpy interop, shape and device."""

    def __init__(self, array, device="cuda:0", dtype="float16"):
        self.array = np.asarray(array)
        self.device = device
        self.dtype = dtype

    def detach(self):
        return self

    def float(self):
        return _FakeTensor(self.array.astype("float32"), self.device, "float32")

    def cpu(self):
        return _FakeTensor(self.array, "cpu", self.dtype)

    def to(self, device=None, dtype=None, non_blocking=False):
        arr = self.array
        if dtype is not None and dtype != "same":
            arr = arr.astype("float32" if str(dtype).endswith("32") else arr.dtype)
        return _FakeTensor(arr, str(device) if device is not None else self.device, self.dtype)

    def __array__(self, dtype=None):
        return self.array.astype(dtype) if dtype is not None else self.array

    # the NMS path uses these
    def __len__(self):
        return len(self.array)

    @property
    def shape(self):
        return self.array.shape

    def argsort(self, descending=False):
        # torch semantics: descending=True sorts values from largest to smallest
        order = np.argsort(-self.array, kind="stable") if descending else np.argsort(self.array, kind="stable")
        return _FakeTensor(order, self.device, "int64")

    def __getitem__(self, idx):
        if isinstance(idx, _FakeTensor):
            idx = idx.array
        return _FakeTensor(self.array[idx], self.device, self.dtype)

    def tolist(self):
        return self.array.tolist()

    def max(self):
        return float(self.array.max())          # what upstream's `if ... >= thresh` reads


def install_fakes():
    import types
    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor

    def from_numpy(arr):
        return _FakeTensor(np.asarray(arr), "cpu", "float32")

    torch.from_numpy = from_numpy
    sys.modules["torch"] = torch
    sys.modules["numpy"] = np
    return torch


# --------------------------------------------------------------------------- upstream algorithms

def upstream_fill_holes(cv2, mask, max_area):
    """Verbatim port of tracker.fill_holes_in_mask_scores (device ops -> numpy)."""
    if max_area <= 0:
        return mask
    arr = np.array(mask, dtype="float32")

    def cc(mask_bin):
        _, labeled, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        return stats[labeled, CC_STAT_AREA].astype("int32")

    for i in range(arr.shape[0]):
        plane = arr[i, 0]
        mask_bg = (plane <= 0).astype("uint8")
        areas_bg = cc(mask_bg)
        small_bg = mask_bg.astype(bool) & (areas_bg <= max_area)
        plane[small_bg] = 0.1

        mask_fg = (plane > 0).astype("uint8")
        thresh = int(mask_fg.sum(dtype="int64")) // 2
        thresh = min(thresh, max_area)
        areas_fg = cc(mask_fg)
        small_fg = mask_fg.astype(bool) & (areas_fg <= thresh)
        plane[small_fg] = -0.1
    return arr


def upstream_nms(compute_overlap, masks, scores, thresh=0.5):
    order = np.argsort(-np.asarray(scores))
    masks, scores = masks[order], np.asarray(scores)[order]
    keep = []
    for i in range(masks.shape[0]):
        if keep:
            if compute_overlap(masks[i:i + 1], masks[keep]).max() >= thresh:
                continue
        keep.append(i)
    return masks[keep], scores[keep]


def compute_overlap(a, b):
    """Same as tracker._compute_mask_overlap, in numpy."""
    a_flat = (np.asarray(a) > 0).astype("float32").reshape(len(a), -1)
    b_flat = (np.asarray(b) > 0).astype("float32").reshape(len(b), -1)
    inter = a_flat @ b_flat.T
    area_a = a_flat.sum(1, keepdims=True)
    area_b = b_flat.sum(1, keepdims=True).T
    iou = inter / np.clip(area_a + area_b - inter, 1, None)
    iom = inter / np.clip(np.minimum(np.broadcast_to(area_a, iou.shape), np.broadcast_to(area_b, iou.shape)), 1, None)
    return _FakeTensor(np.maximum(iou, iom), "cuda:0", "float32")   # a device tensor, like upstream


def main():
    install_fakes()
    import sam3_fast as sf

    tracker = type("Tracker", (), {"_HAS_CV2": True, "cv2": fake_cv2,
                                   "_compute_mask_overlap": staticmethod(compute_overlap)})()

    rng = np.random.default_rng(0)
    bad = 0
    print("fill_holes_in_mask_scores: fast vs upstream")
    for trial in range(40):
        n_obj = int(rng.integers(1, 4))
        h = w = 48
        mask = (rng.random((n_obj, 1, h, w)) > 0.55).astype("float32") * 10.0 - 5.0
        # inject sprinkles (1..9 px fg blobs) and holes (1..9 px bg blobs) in known spots
        for _ in range(12):
            cy, cx = int(rng.integers(1, h - 4)), int(rng.integers(1, w - 4))
            size = int(rng.integers(1, 4))
            target = rng.random() > 0.5
            mask[0, 0, cy:cy + size, cx:cx + size] = 7.0 if target else -7.0
        max_area = int(rng.choice([0, 4, 16, 64]))
        want = upstream_fill_holes(fake_cv2, np.array(mask), max_area)
        got = np.asarray(sf._fill_holes_one_trip(tracker, _FakeTensor(mask), max_area).array)
        if not np.array_equal(want, got):
            bad += 1
            diff = np.argwhere(want != got)[:3]
            print(f"  MISMATCH trial {trial} max_area {max_area}: {len(np.argwhere(want != got))} px, first {diff.tolist()}")
    print(f"  40 random masks x max_area in (0, 4, 16, 64): {'identical' if bad == 0 else f'{bad} mismatches'}")

    print("_nms_masks: fast vs upstream (greedy decisions)")
    bad = 0
    for trial in range(40):
        n = int(rng.integers(2, 8))
        masks = np.zeros((n, 1, 24, 24), dtype="float32")
        for i in range(n):
            y, x = int(rng.integers(0, 16)), int(rng.integers(0, 16))
            masks[i, 0, y:y + 8, x:x + 8] = 1.0
            if i > 1:                                         # overlapping blobs, like real detections
                masks[i, 0, y + 2:y + 6, x + 2:x + 6] = 1.0
                z = int(rng.integers(0, 14))
                masks[i, 0, z:z + 6, z:z + 6] = 1.0
        scores = rng.random(n).astype("float32")
        want_m, want_s = upstream_nms(compute_overlap, masks, scores)
        got_m, got_s = sf._nms_one_trip(tracker, _FakeTensor(masks), _FakeTensor(scores), 0.5)
        got_m = np.asarray(getattr(got_m, "array", got_m))
        got_s = np.asarray(getattr(got_s, "array", got_s))
        if not (np.array_equal(np.asarray(want_m), got_m) and np.array_equal(np.asarray(want_s), got_s)):
            bad += 1
            print(f"  MISMATCH trial {trial}: kept {len(want_s)} vs {len(got_s)}")
    print(f"  40 random detection sets: {'identical' if bad == 0 else f'{bad} mismatches'}")

    return 1 if bad else 0


def install_test():
    """Wire patch_sam3_fast() against fake modules and confirm it patches, routes and reports."""
    import os
    import types
    install_fakes()
    import sam3_fast as sf

    tracker = types.ModuleType("comfy.ldm.sam3.tracker")
    tracker._HAS_CV2 = True
    tracker.cv2 = fake_cv2
    tracker._compute_mask_overlap = compute_overlap
    tracker.fill_holes_in_mask_scores = lambda mask, max_area=0: mask       # upstream stand-in
    tracker._nms_masks = lambda masks, scores, thresh=0.5: (masks, scores)
    node = types.SimpleNamespace()
    node.execute = classmethod(lambda cls, *a, **k: "ran")
    nodes_mod = types.ModuleType("nodes")
    nodes_mod.NODE_CLASS_MAPPINGS = {"SAM3_VideoTrack": node}
    mm = types.ModuleType("comfy.model_management")
    mm.intermediate_device = lambda: "cpu"
    mm.get_torch_device = lambda: "cuda:0"
    modules = {"comfy": types.ModuleType("comfy"), "comfy.ldm": types.ModuleType("comfy.ldm"),
               "comfy.ldm.sam3": types.ModuleType("comfy.ldm.sam3"),
               "comfy.ldm.sam3.tracker": tracker, "nodes": nodes_mod,
               "comfy.model_management": mm}
    for name, mod in modules.items():
        if name in ("comfy",):
            mod.ldm = modules["comfy.ldm"]
        if name in ("comfy", "comfy.ldm"):
            mod.sam3 = modules["comfy.ldm.sam3"]
        if name in ("comfy", "comfy.ldm", "comfy.ldm.sam3"):
            mod.tracker = tracker
        if name == "comfy":
            mod.model_management = mm
        sys.modules[name] = mod

    ok = sf.patch_sam3_fast()
    checks = [("patch_sam3_fast returned True", ok is True),
              ("fill_holes patched", getattr(tracker.fill_holes_in_mask_scores, "_mmh3_fast", False)),
              ("nms patched", getattr(tracker._nms_masks, "_mmh3_fast", False)),
              ("node patched", getattr(node, "_mmh3_fast", False))]
    # route a call through the patched entry point and confirm the fast path really ran
    mask = _FakeTensor((np.random.default_rng(1).random((1, 1, 32, 32)) > 0.6).astype("float32") * 5.0)
    out = tracker.fill_holes_in_mask_scores(mask, 16)
    checks.append(("fast fill ran", int(sf._STATS["fill_holes_in_mask_scores_calls"]) == 1))
    checks.append(("output is a tensor", isinstance(out, _FakeTensor)))
    out2 = tracker._nms_masks(_FakeTensor(np.ones((3, 1, 8, 8), "float32")),
                              _FakeTensor(np.array([0.9, 0.5, 0.1], "float32")), 0.5)
    checks.append(("fast nms ran", int(sf._STATS["_nms_masks_calls"]) == 1))
    # a raising fast path must fall back to the working upstream, not propagate
    upstream_marker = object()
    tracker.fill_holes_in_mask_scores = lambda *a, **k: upstream_marker
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    sf._wrap(tracker, "fill_holes_in_mask_scores", boom)
    sf._WARNED.clear()
    fell_back = tracker.fill_holes_in_mask_scores(mask, 16)
    checks.append(("fallback on error", fell_back is upstream_marker))
    sf._report("sam_fast_test.txt")
    checks.append(("report written", os.path.exists("sam_fast_test.txt")))
    if os.path.exists("sam_fast_test.txt"):
        print("--- sam_fast_test.txt ---")
        print(open("sam_fast_test.txt").read().strip())
        os.remove("sam_fast_test.txt")
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    return 0 if all(p for _, p in checks) else 1


if __name__ == "__main__":
    sys.exit(main() or install_test())
