"""Profile one SAM3_VideoTrack run on request: create sam_profile.request in ComfyUI's working dir,
run a tracking job, read sam_profile.txt (wall time, per-device kernel busy time, top kernels)."""
import logging
import os
import time
from collections import defaultdict

import torch

logger = logging.getLogger("MultiGPU")
FLAG = "sam_profile.request"


def patch_sam3_profile():
    import nodes
    # ComfyUI loads its built-in node files under an internal module name, so patch the registered class
    node = nodes.NODE_CLASS_MAPPINGS.get("SAM3_VideoTrack")
    if node is None:
        return False
    if getattr(node, "_mmh3_profile", False):
        return True
    execute = node.execute.__func__

    def profiled(cls, *args, **kwargs):
        if not os.path.exists(FLAG):
            return execute(cls, *args, **kwargs)
        os.remove(FLAG)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = execute(cls, *args, **kwargs)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        busy = defaultdict(float)
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                busy[e.device_index] += e.self_device_time_total / 1e6
        report = ["SAM3_VideoTrack (profiler on): wall %.2fs | kernel busy %s" % (wall, "  ".join(
                  "cuda:%d %.2fs" % (d, t) for d, t in sorted(busy.items()))),
                  prof.key_averages().table(sort_by="self_device_time_total", row_limit=30),
                  prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)]
        with open("sam_profile.txt", "w") as f:
            f.write("\n".join(report))
        logger.info("[MultiGPU] SAM profile written to %s: %s", os.path.abspath("sam_profile.txt"), report[0])
        return out

    node.execute = classmethod(profiled)
    node._mmh3_profile = True
    return True
