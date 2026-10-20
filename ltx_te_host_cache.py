"""Opt-in host caching for large text encoders used with disk-backed models.

ComfyUI's global --fast-disk mode is required for LTX 2.5 on ~30 GB hosts,
but it also makes the Gemma text encoder fault its weights from the model file
again after GPU eviction.  This helper changes only the selected CLIP patcher
back to the normal RAM-pressure-managed host cache.
"""

import logging
import os


logger = logging.getLogger("MultiGPU")
_ENV = "LTX_TE_HOST_CACHE"


def prefer_clip_host_cache(clip):
    """Prefer host caching for one CLIP object when explicitly enabled."""
    if os.environ.get(_ENV, "0") != "1":
        return False

    patcher = getattr(clip, "patcher", None)
    if patcher is None or not getattr(patcher, "is_dynamic", lambda: False)():
        logger.warning("[MultiGPU] LTX TE host cache requested, but CLIP is not dynamically loaded")
        return False

    patcher.fast_disk = False
    model = getattr(patcher, "model", None)
    for pin_state in getattr(model, "dynamic_pins", {}).values():
        pin_state["fast_disk"] = False

    logger.info(
        "[MultiGPU] LTX TE host cache enabled: Gemma weights use ComfyUI's RAM-pressure-managed cache"
    )
    return True
