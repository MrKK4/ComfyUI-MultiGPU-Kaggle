"""Diagnostics for the H3 tensor-parallel path (Kaggle 2x T4).

Two things the TP path currently guesses at, measured instead:

1. `probe()` -- what the chunked exchange can actually move: host-staging round trip
   (D2H + H2D through pinned memory, both GPUs at once, same chunking as
   `_ChunkedExchange`) vs direct peer copies when the driver allows them, plus the
   PCIe link each GPU sits on. This decides whether the exchange is a wall or a
   rounding error.
2. `analyze_trace()` -- one profiled step (the existing `tp_profile.request` hook)
   exported as a chrome trace and broken down per device into kernel classes
   (int8 GEMM / attention / norm+rope / elementwise / memcpy) with PCIe bytes and
   the share of the step each device was actually busy. That is the accounting of
   where a step's wall time goes.

Everything is inert unless MMH3_TP_DIAG=1 is set before ComfyUI starts.

Standalone use (same python as ComfyUI, from ComfyUI's working dir):
    python tp_diag.py probe            # topology + bandwidth, no ComfyUI needed
    python tp_diag.py analyze [trace]  # summarize an exported chrome trace

Env: MMH3_TP_DIAG=1 to install the hooks, MMH3_TP_DIAG_ROWS (default 7700),
MMH3_TP_DIAG_COLS (default 5376), MMH3_TP_DIAG_CHUNKS (default 4).
"""
import ctypes
import ctypes.util
import json
import logging
import os
import time
from collections import defaultdict

logger = logging.getLogger("MultiGPU")

REPORT = "tp_diag.txt"
TRACE = "tp_diag_trace.json"
ON = os.environ.get("MMH3_TP_DIAG", "0").strip().lower() not in ("", "0", "false", "no")

ROWS = int(os.environ.get("MMH3_TP_DIAG_ROWS", "7700"))   # tokens in one packed H3 sequence
COLS = int(os.environ.get("MMH3_TP_DIAG_COLS", "5376"))   # hidden size of the exchanged partial
CHUNKS = max(1, int(os.environ.get("MMH3_TP_DIAG_CHUNKS", "4")))


def _append(path, text):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")


def _gbps(nbytes, seconds):
    return 0.0 if seconds <= 0 else nbytes / seconds / 1e9


# ---------------------------------------------------------------------------
# topology / bandwidth probe
# ---------------------------------------------------------------------------

def _cudart():
    names = ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
    try:
        found = ctypes.util.find_library("cudart")
        if found:
            names.append(found)
    except Exception:
        pass
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _peer_matrix(count):
    """cudaDeviceCanAccessPeer for every ordered pair, or None where unavailable."""
    lib = _cudart()
    matrix = {}
    for a in range(count):
        for b in range(count):
            if a == b:
                matrix[(a, b)] = True
                continue
            matrix[(a, b)] = None
            if lib is None:
                continue
            value = ctypes.c_int(0)
            try:
                if lib.cudaDeviceCanAccessPeer(ctypes.byref(value), a, b) == 0:
                    matrix[(a, b)] = bool(value.value)
            except Exception:
                pass
    return matrix


def _enable_peer(src, dst):
    """Best effort: let src reach dst's memory. Returns True if a peer copy is plausible."""
    try:
        import torch
        for name in ("enable_peer_access", "enable_peer_memory"):
            fn = getattr(torch.cuda, name, None)
            if callable(fn):
                try:
                    fn(src)
                    return True
                except Exception:
                    pass
    except Exception:
        pass
    lib = _cudart()
    if lib is None:
        return False
    try:
        return lib.cudaDeviceEnablePeerAccess(dst, 0) == 0
    except Exception:
        return False


def _pcie_link(index):
    """PCIe generation/width for one GPU, from nvidia-smi (no NVML dependency)."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and parts[0] == str(index):
                return f"{parts[1]} gen{parts[2]} x{parts[3]}"
    except Exception:
        pass
    return "unknown"


def _staged_roundtrip(torch, rows, cols, chunks, dtype):
    """Mimic _ChunkedExchange: both ranks push their partial to the other through pinned host
    buffers, in row chunks, on a per-device copy stream. Returns (bytes, seconds)."""
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    shape = (rows, cols)
    item = torch.tensor([], dtype=dtype).element_size()
    src = [torch.empty(shape, dtype=dtype, device=d) for d in devs]
    dst = [torch.empty(shape, dtype=dtype, device=d) for d in devs]
    host = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in devs]
    streams = [torch.cuda.Stream(d) for d in devs]
    step = -(-rows // chunks)
    for d in devs:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for a in range(0, rows, step):
        b = min(a + step, rows)
        for r, d in enumerate(devs):
            with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                host[r][a:b].copy_(src[r][a:b], non_blocking=True)
                dst[1 - r][a:b].copy_(host[r][a:b], non_blocking=True)
    for d in devs:
        torch.cuda.synchronize(d)
    seconds = time.perf_counter() - t0
    payload = rows * cols * item
    del src, dst, host
    return 2 * payload * 2, seconds  # both ranks, D2H + H2D


def _peer_roundtrip(torch, rows, cols, dtype):
    """Direct device-to-device copies between the two GPUs (one PCIe crossing per byte)."""
    if not (_enable_peer(0, 1) and _enable_peer(1, 0)):
        return None
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    shape = (rows, cols)
    item = torch.tensor([], dtype=dtype).element_size()
    src = [torch.empty(shape, dtype=dtype, device=d) for d in devs]
    dst = [torch.empty(shape, dtype=dtype, device=d) for d in devs]
    streams = [torch.cuda.Stream(d) for d in devs]
    for d in devs:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for r, d in enumerate(devs):
        with torch.cuda.device(d), torch.cuda.stream(streams[r]):
            dst[1 - r].copy_(src[r], non_blocking=True)
    for d in devs:
        torch.cuda.synchronize(d)
    seconds = time.perf_counter() - t0
    payload = rows * cols * item
    del src, dst
    return 2 * payload, seconds


def probe(rows=None, cols=None, chunks=None):
    """Measure what the exchange can move. Returns the report as text; writes tp_diag.txt too."""
    rows = ROWS if rows is None else rows
    cols = COLS if cols is None else cols
    chunks = CHUNKS if chunks is None else chunks
    lines = [f"tp_diag probe — {time.strftime('%Y-%m-%dT%H:%M:%S')}"]
    try:
        import torch
    except ImportError:
        lines.append("torch unavailable")
        text = "\n".join(lines)
        _append(REPORT, text)
        return text
    if not torch.cuda.is_available():
        lines.append("cuda unavailable")
        text = "\n".join(lines)
        _append(REPORT, text)
        return text

    count = torch.cuda.device_count()
    lines.append(f"devices: {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        lines.append(f"  cuda:{i} {props.name} sm_{props.major}{props.minor} "
                     f"{props.total_memory / 1024 ** 3:.1f}GiB pcie[{_pcie_link(i)}]")
    matrix = _peer_matrix(count)
    if count > 1:
        pairs = " ".join(f"cuda:{a}->cuda:{b}:{'Y' if matrix.get((a, b)) else ('N' if matrix[(a, b)] is not None else '?')}"
                         for a in range(count) for b in range(count) if a != b)
        lines.append(f"peer access: {pairs}")
    if count < 2:
        lines.append("single device — exchange probe skipped")
        text = "\n".join(lines)
        _append(REPORT, text)
        return text

    dtype = torch.float16
    payload_gb = rows * cols * 2 / 1024 ** 3
    lines.append(f"exchange payload: {rows}x{cols} {str(dtype).split('.')[-1]} "
                 f"= {payload_gb:.2f}GiB per rank, {chunks} chunks")

    # single direction reference numbers
    try:
        big = torch.empty(rows, cols, dtype=dtype, device="cuda:0")
        host = torch.empty(rows, cols, dtype=dtype, pin_memory=True)
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        host.copy_(big, non_blocking=True)
        torch.cuda.synchronize(0)
        d2h = time.perf_counter() - t0
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        big.copy_(host, non_blocking=True)
        torch.cuda.synchronize(0)
        h2d = time.perf_counter() - t0
        lines.append(f"cuda:0 pinned D2H {_gbps(payload_gb * 1024 ** 3, d2h):.2f} GB/s, "
                     f"H2D {_gbps(payload_gb * 1024 ** 3, h2d):.2f} GB/s")
        del big, host
    except Exception as exc:
        lines.append(f"single-direction reference failed: {exc}")

    try:
        moved, seconds = _staged_roundtrip(torch, rows, cols, chunks, dtype)
    except Exception as exc:
        moved, seconds = 0, 0.0
        lines.append(f"staged exchange failed: {exc}")
    if seconds:
        lines.append(f"staged exchange (both ranks, {chunks} chunks): {seconds * 1000:.1f} ms, "
                     f"{_gbps(moved, seconds):.2f} GB/s aggregate, "
                     f"{_gbps(moved / 2, seconds):.2f} GB/s per direction")
        lines.append(f"  -> a 50-block step moves 100 exchanges: {50 * seconds:.2f} s of PCIe if not overlapped")

    try:
        peer = _peer_roundtrip(torch, rows, cols, dtype)
    except Exception as exc:
        peer = None
        lines.append(f"peer exchange failed: {exc}")
    if peer is None:
        lines.append("peer exchange: unavailable (no cudaDeviceCanAccessPeer both ways)")
    else:
        moved, seconds = peer
        lines.append(f"peer exchange (both ranks): {seconds * 1000:.1f} ms, "
                     f"{_gbps(moved, seconds):.2f} GB/s aggregate"
                     f" -> {50 * seconds:.2f} s per 50-block step")
        try:
            staged = _staged_roundtrip(torch, rows, cols, chunks, dtype)
            if staged[1] > 0:
                lines.append(f"peer speedup over staging: {staged[1] / seconds:.2f}x")
        except Exception:
            pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    text = "\n".join(lines)
    _append(REPORT, text)
    logger.info("[MultiGPU TP diag] probe written to %s", os.path.abspath(REPORT))
    return text


# ---------------------------------------------------------------------------
# chrome trace breakdown
# ---------------------------------------------------------------------------

def classify(name, category=""):
    """Bucket one GPU kernel/memcpy event into a coarse class."""
    low = name.lower()
    if category in ("gpu_memcpy", "Memcpy") or low.startswith("memcpy"):
        return "memcpy"
    if "int8" in low or "convrot" in low or "s8_" in low or "quant" in low:
        return "gemm_int8"
    if any(k in low for k in ("fmha", "attention", "flash", "efficient_attention", "sdpa")):
        return "attention"
    if any(k in low for k in ("rms", "rope", "norm")):
        return "norm_rope"
    if any(k in low for k in ("gemm", "cublas", "nvjet", "cutlass", "wgrad", "sgemm", "hgmma", "igemm")):
        return "gemm_fp"
    if any(k in low for k in ("elementwise", "vectorized", "silu", "sigmoid", "gemv", "reduce",
                              "convert", "cast", "_to_copy", "fill", "mul", "add", "sub", "div",
                              "cat", "copy", "index", "gather", "scatter", "embedding", "softmax")):
        return "elementwise"
    return "other"


def _memcpy_bytes(event):
    args = event.get("args") or {}
    for key in ("bytes", "Bytes", "size", "num_bytes"):
        value = args.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _event_device(event):
    """Device index of one GPU event, tolerant of the shapes torch's chrome trace has used."""
    args = event.get("args") or {}
    for key in ("device", "device_id", "deviceId", "Device"):
        value = args.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if ":" in text and text.startswith(("cuda", "xpu")):
                try:
                    return int(text.split(":")[-1])
                except ValueError:
                    pass
    pid = event.get("pid")
    if isinstance(pid, int) and pid >= 0:
        return pid
    return None


def _event_stream(event):
    args = event.get("args") or {}
    for key in ("stream", "Stream", "stream_id"):
        value = args.get(key)
        if isinstance(value, int):
            return value
    tid = event.get("tid")
    return tid if isinstance(tid, int) else -1


def _union_span(intervals):
    """Wall time (same unit as the intervals, i.e. us) with at least one interval active."""
    merged = 0.0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                merged += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        merged += cur_end - cur_start
    return merged


def analyze_trace(path=TRACE):
    """Summarize one exported chrome trace: per-device busy time by class, PCIe bytes, top kernels."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data

    per_dev_class = defaultdict(float)          # (device, class) -> us
    per_dev_total = defaultdict(float)
    per_dev_span = {}                           # device -> (first_ts, last_end)
    per_dev_intervals = defaultdict(list)       # device -> [(start, end)]
    per_dev_streams = defaultdict(lambda: defaultdict(float))
    bytes_by_dir = defaultdict(int)
    by_name = defaultdict(float)                # name -> total gpu us
    by_name_dev = defaultdict(lambda: defaultdict(float))

    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category not in ("kernel", "gpu_memcpy", "Memcpy", "gpu_user_annotation"):
            continue
        name = event.get("name", "")
        duration = float(event.get("dur", 0.0))
        device = _event_device(event)
        if device is None:
            continue
        klass = classify(name, category)
        per_dev_class[(device, klass)] += duration
        per_dev_total[device] += duration
        per_dev_streams[device][_event_stream(event)] += duration
        if category in ("gpu_memcpy", "Memcpy") or name.lower().startswith("memcpy"):
            direction = "other"
            low = name.lower()
            if "dtoh" in low or "device -> pinned" in low or "device to host" in low:
                direction = "DtoH"
            elif "htod" in low or "pinned -> device" in low or "host to device" in low:
                direction = "HtoD"
            elif "dtod" in low or "device to device" in low:
                direction = "DtoD"
            bytes_by_dir[direction] += _memcpy_bytes(event)
        ts = float(event.get("ts", 0.0))
        span = per_dev_span.get(device)
        per_dev_span[device] = (min(span[0], ts), max(span[1], ts + duration)) if span else (ts, ts + duration)
        per_dev_intervals[device].append((ts, ts + duration))
        by_name[name] += duration
        by_name_dev[name][device] += duration

    return {
        "path": path,
        "per_dev_class": per_dev_class,
        "per_dev_total": per_dev_total,
        "per_dev_span": per_dev_span,
        "per_dev_occupancy": {device: _union_span(iv) for device, iv in per_dev_intervals.items()},
        "per_dev_streams": per_dev_streams,
        "bytes_by_dir": bytes_by_dir,
        "by_name": by_name,
        "by_name_dev": by_name_dev,
    }


def format_report(summary, top=12):
    """Render analyze_trace() output as the accounting table."""
    lines = [f"tp_diag — profiled step breakdown — {time.strftime('%Y-%m-%dT%H:%M:%S')}"]
    spans = summary["per_dev_span"]
    wall = max((end - start for start, end in spans.values()), default=0.0) / 1e6
    lines.append(f"step wall (device timelines): {wall:.2f} s")
    used_devices = sorted(summary["per_dev_total"])
    lines.append(f"devices in trace: {used_devices}")
    for device in used_devices:
        busy = summary["per_dev_total"][device] / 1e6
        resident = summary.get("per_dev_occupancy", {}).get(device, 0.0) / 1e6
        share = (resident / wall * 100) if wall else 0.0
        streams = sorted(summary.get("per_dev_streams", {}).get(device, {}).items(), key=lambda kv: -kv[1])
        stream_text = " ".join(f"s{sid}={sec / 1e6:.2f}s" for sid, sec in streams[:4])
        lines.append(f"  cuda:{device}: resident {resident:.2f} s ({share:.0f}% of step), "
                     f"kernel-time {busy:.2f} s, gap {max(0.0, wall - resident):.2f} s | streams: {stream_text}")

    classes = sorted({klass for _, klass in summary["per_dev_class"]})
    devices = sorted(summary["per_dev_total"])
    lines.append("  per-class GPU time (s):")
    lines.append("    " + "class".ljust(14) + "".join(f"cuda:{d}".rjust(12) for d in devices))
    for klass in classes:
        row = "    " + klass.ljust(14)
        for device in devices:
            row += f"{summary['per_dev_class'].get((device, klass), 0.0) / 1e6:12.2f}"
        lines.append(row)

    moved = summary["bytes_by_dir"]
    if moved:
        lines.append("  PCIe traffic: " + ", ".join(
            f"{direction} {value / 1024 ** 3:.2f} GiB" for direction, value in sorted(moved.items())))
    else:
        lines.append("  PCIe traffic: not in trace (bytes arg missing)")

    ranked = sorted(summary["by_name"].items(), key=lambda kv: -kv[1])[:top]
    if ranked:
        lines.append(f"  top {len(ranked)} kernels by GPU time:")
        for name, total in ranked:
            per_device = summary["by_name_dev"][name]
            detail = " ".join(f"cuda:{d}={per_device[d] / 1e6:.2f}s" for d in sorted(per_device))
            lines.append(f"    {total / 1e6:8.2f}s  {name[:64]:64s} {detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------

def _install_trace_capture(tp_module):
    """Keep the TP profiler and add trace export + breakdown behind it."""
    original = getattr(tp_module, "_profiled", None)
    if original is None or getattr(original, "_mmh3_diag", False):
        return False
    try:
        import torch
    except ImportError:
        return False
    real_profile = torch.profiler.profile
    captured = []

    class CapturingProfile(real_profile):
        def __enter__(self):
            captured.append(self)
            return real_profile.__enter__(self)

    def profiled_with_diag(executor, *args, **kwargs):
        captured.clear()
        torch.profiler.profile = CapturingProfile
        try:
            out = original(executor, *args, **kwargs)
        finally:
            torch.profiler.profile = real_profile
        if captured:
            try:
                captured[-1].export_chrome_trace(TRACE)
                _append(REPORT, format_report(analyze_trace(TRACE)))
                logger.info("[MultiGPU TP diag] trace -> %s, breakdown -> %s",
                            os.path.abspath(TRACE), os.path.abspath(REPORT))
            except Exception as exc:
                logger.warning("[MultiGPU TP diag] trace breakdown failed: %s", exc)
        return out

    profiled_with_diag._mmh3_diag = True
    profiled_with_diag._mmh3_original = original
    tp_module._profiled = profiled_with_diag
    return True


def patch_tp_diag():
    """Install the diagnostics. No-op unless MMH3_TP_DIAG=1."""
    if not ON:
        return False
    installed = False
    try:
        from . import h3_tensor_parallel as tp_module
        installed = _install_trace_capture(tp_module)
    except Exception as exc:
        logger.debug("[MultiGPU TP diag] profiler hook unavailable: %s", exc)
    try:
        probe()
    except Exception as exc:
        logger.warning("[MultiGPU TP diag] probe failed: %s", exc)
    logger.info("[MultiGPU TP diag] enabled (report: %s)", os.path.abspath(REPORT))
    return installed or True


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    action = argv[0] if argv else "probe"
    if action == "probe":
        print(probe())
        return 0
    if action == "analyze":
        trace = argv[1] if len(argv) > 1 else TRACE
        text = format_report(analyze_trace(trace))
        print(text)
        _append(REPORT, text)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
