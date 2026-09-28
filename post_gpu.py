"""Keep the post-decode chain on the GPU instead of bouncing every frame through host memory.

Where the 40 s of "CPU blur / uncrop / MP4" comes from: ComfyUI's post nodes do their work on the
input's device and then hand the result to `comfy.model_management.intermediate_device()`, which is
**CPU** unless ComfyUI was started with `--gpu-only`. `VAE.decode` ends the same way, so the decoded
frames land on the host, and every node after it -- `ImageBlur` (a reflect-padded conv2d), composite,
uncrop, mask ops -- runs its kernel on the CPU over the whole 124-frame batch, then copies back to
the GPU for the next node that needs one.

`intermediate_device()` is the single junction. This patch makes it return the compute device while
a run is opted in, so:

  * `tensor.to(intermediate_device())` becomes a no-op for a tensor that is already on the GPU, and
    the D2H copy disappears rather than being paid once per node,
  * the convs and interpolations inside those nodes are issued on the GPU, where a 0.5 MP x 124
    frame batch is milliseconds instead of seconds,
  * the final video encode still reads host memory, which is one copy at the end instead of one per
    node.

Opt in with `MMH3_POST_GPU=1`, or by creating a file named `post_gpu.request` in ComfyUI's working
directory (delete it to turn the mode off without restarting). Default is off, because the mode
trades host memory for VRAM and a workflow tuned around CPU intermediates could otherwise run the
card out of memory.

The VRAM guard is what keeps that safe: before answering "use the GPU", the patched function checks
free memory on the compute device and falls back to the original CPU device when there is less than
`MMH3_POST_GPU_MB` (default 2048) left. The check is cached for `MMH3_POST_GPU_TTL` seconds
(default 0.5) because `intermediate_device()` is called on every node and a driver query per call
would show up in the profile.

`post_gpu.txt` records how often the mode redirected, how often the guard refused, and the lowest
free memory seen, so the A/B is: one run with `MMH3_POST_GPU=1`, one without, compare the post stage.
"""
import logging
import os
import time

logger = logging.getLogger("MultiGPU")
REQUEST = "post_gpu.request"
REPORT = "post_gpu.txt"
ENABLED = os.environ.get("MMH3_POST_GPU", "0") != "0"
LIMIT_MB = float(os.environ.get("MMH3_POST_GPU_MB", "2048"))
TTL = float(os.environ.get("MMH3_POST_GPU_TTL", "0.5"))
_STATS = {"redirect": 0, "guard": 0, "error": 0, "min_free": None}


def _wanted():
    return ENABLED or os.path.exists(REQUEST)


def patch_post_gpu():
    """Point intermediate_device() at the compute device while the run is opted in."""
    if not _wanted():
        logger.info("[MultiGPU] GPU post-processing intermediates off "
                    "(set MMH3_POST_GPU=1 or touch %s to enable)", REQUEST)
        return False
    try:
        import comfy.model_management as mm
        import torch
    except Exception as exc:
        logger.info("[MultiGPU] GPU post-processing intermediates unavailable: %s: %s",
                    type(exc).__name__, exc)
        return False
    if getattr(mm, "_mmh3_post_gpu", False):
        return True
    original = mm.intermediate_device
    state = {"open": True, "free": None, "at": 0.0}

    def gpu_if_room():
        """The compute device when the mode is on and there is VRAM to spare, else the original."""
        if not _wanted():
            return original()
        try:
            device = mm.get_torch_device()
        except Exception:
            _STATS["error"] += 1
            return original()
        if getattr(device, "type", "cpu") != "cuda":
            return original()
        now = time.perf_counter()
        if now - state["at"] >= TTL:
            # comfy's own helper: torch.cuda.mem_get_info based, but it also accounts for the
            # free-memory model ComfyUI keeps, which is what decides whether the next node can load
            try:
                free = mm.get_free_memory(device)
                state["free"] = int(free)
                state["at"] = now
                _STATS["min_free"] = free if _STATS["min_free"] is None else min(_STATS["min_free"], free)
            except Exception:
                _STATS["error"] += 1
                return original()
        if state["free"] is None or state["free"] < LIMIT_MB * 1e6:
            _STATS["guard"] += 1
            _write_report()
            return original()
        _STATS["redirect"] += 1
        _write_report()
        return device

    gpu_if_room._mmh3_original = original
    mm.intermediate_device = gpu_if_room
    mm._mmh3_post_gpu = True
    logger.info("[MultiGPU] GPU post-processing intermediates ON: frames stay on %s while free VRAM "
                "stays above %.0f MB; delete %s or set MMH3_POST_GPU=0 to stop",
                os.environ.get("CUDA_VISIBLE_DEVICES", "the compute device"), LIMIT_MB, REQUEST)
    return True


def _write_report(path=REPORT):
    if _STATS["redirect"] + _STATS["guard"] + _STATS["error"] == 0:
        return
    free = _STATS["min_free"]
    lines = [
        "GPU post-processing intermediates (MMH3_POST_GPU=%s, request file %s)"
        % ("1" if ENABLED else "0", "present" if os.path.exists(REQUEST) else "absent"),
        "intermediate_device() redirects to the GPU: %d" % _STATS["redirect"],
        "refused by the %.0f MB floor (fell back to CPU): %d" % (LIMIT_MB, _STATS["guard"]),
        "fell back on an error: %d" % _STATS["error"],
        "lowest free VRAM seen: %s" % ("n/a" if free is None else "%.0f MB" % (free / 1e6)),
        "",
        "Compare a run with MMH3_POST_GPU=1 against one without on the blur/uncrop/composite part of",
        "the job: the frames now stay on the GPU after VAE decode instead of being copied to host by",
        "every node. The final encode still reads host memory.",
    ]
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
