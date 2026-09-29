"""Offline checks for post_gpu.py, the GPU-intermediates mode for the post-decode chain.

The patch answers a question ComfyUI asks on every node -- "where should intermediates live?" -- so
the failure modes worth proving are the boring ones: it must never answer "GPU" when the mode is
off, when there is no CUDA device, when VRAM is short, or when the query itself breaks; it must be
cheap enough to call thousands of times; and every fallback must be the untouched original.

    python check_post_gpu.py        # exits 1 on any failure
"""
import importlib
import os
import sys
import types


class Device(str):
    """A device that reports a .type like torch.device does."""

    @property
    def type(self):
        return "cuda" if self.startswith("cuda") else "cpu"


def install_fakes(free_mb=8000, free_raises=False):
    torch = types.ModuleType("torch")
    torch.device = Device
    sys.modules["torch"] = torch
    calls = {"free": 0}

    mm = types.ModuleType("comfy.model_management")
    mm.intermediate_device = lambda: Device("cpu")

    def get_torch_device():
        return Device("cuda:0")

    def get_free_memory(device=None):
        calls["free"] += 1
        if free_raises:
            raise RuntimeError("driver query failed")
        return free_mb * 1e6

    mm.get_torch_device = get_torch_device
    mm.get_free_memory = get_free_memory
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    return mm, calls


def reload_module(env=None, touch_request=False):
    for key in list(os.environ):
        if key.startswith("MMH3_POST_GPU"):
            del os.environ[key]
    for key, value in (env or {}).items():
        os.environ[key] = value
    path = "post_gpu.request"
    if touch_request:
        open(path, "w").close()
    elif os.path.exists(path):
        os.remove(path)
    for name in ("post_gpu",):
        sys.modules.pop(name, None)
    return importlib.import_module("post_gpu"), path


def main():
    checks = []

    # 1. off by default: the original answer survives, untouched
    mm, calls = install_fakes()
    org = mm.intermediate_device
    sf, req = reload_module()
    installed = sf.patch_post_gpu()
    for _ in range(100):
        assert mm.intermediate_device().type == "cpu"
    checks.append(("off by default: not installed, still CPU", installed is False and mm.intermediate_device is org))

    # 2. env opt-in redirects to the compute device
    mm, calls = install_fakes(free_mb=8000)
    sf, req = reload_module({"MMH3_POST_GPU": "1", "MMH3_POST_GPU_MB": "2048", "MMH3_POST_GPU_TTL": "0.5"})
    checks.append(("patch installs with MMH3_POST_GPU=1", sf.patch_post_gpu() is True))
    checks.append(("intermediates answer cuda:0", mm.intermediate_device().type == "cuda"))
    checks.append(("redirect counted", sf._STATS["redirect"] == 1))

    # 3. the VRAM floor refuses instead of OOMing
    mm, calls = install_fakes(free_mb=512)
    sf, req = reload_module({"MMH3_POST_GPU": "1", "MMH3_POST_GPU_MB": "2048"})
    sf.patch_post_gpu()
    checks.append(("below the floor: falls back to CPU", mm.intermediate_device().type == "cpu"))
    checks.append(("refusal counted", sf._STATS["guard"] == 1))
    checks.append(("nothing redirected", sf._STATS["redirect"] == 0))

    # 4. the TTL caches the driver query: intermediate_device() is called per node
    mm, calls = install_fakes(free_mb=8000)
    sf, req = reload_module({"MMH3_POST_GPU": "1", "MMH3_POST_GPU_TTL": "3600"})
    sf.patch_post_gpu()
    for _ in range(500):
        mm.intermediate_device()
    checks.append(("500 calls, %d memory queries (TTL caches it)" % calls["free"], calls["free"] <= 2))
    mm2, calls2 = install_fakes(free_mb=8000)
    sf, req = reload_module({"MMH3_POST_GPU": "1", "MMH3_POST_GPU_TTL": "0"})
    sf.patch_post_gpu()
    for _ in range(10):
        mm2.intermediate_device()
    checks.append(("TTL=0 queries every call (%d for 10 calls)" % calls2["free"], calls2["free"] == 10))

    # 5. a broken query falls back rather than propagating
    mm, calls = install_fakes(free_raises=True)
    sf, req = reload_module({"MMH3_POST_GPU": "1"})
    sf.patch_post_gpu()
    checks.append(("query error: falls back to CPU", mm.intermediate_device().type == "cpu"))
    checks.append(("error counted", sf._STATS["error"] == 1))

    # 6. the request file turns it on and off without a restart
    mm, calls = install_fakes(free_mb=8000)
    sf, req = reload_module(touch_request=True)
    checks.append(("request file installs the patch", sf.patch_post_gpu() is True))
    checks.append(("request file redirects", mm.intermediate_device().type == "cuda"))
    os.remove(req)
    checks.append(("deleting the file reverts to CPU without a restart", mm.intermediate_device().type == "cpu"))

    # 7. no CUDA device: nothing to redirect to
    mm, calls = install_fakes(free_mb=8000)
    mm.get_torch_device = lambda: Device("cpu")
    sf, req = reload_module({"MMH3_POST_GPU": "1"})
    sf.patch_post_gpu()
    checks.append(("no CUDA device: stays CPU", mm.intermediate_device().type == "cpu"))

    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
