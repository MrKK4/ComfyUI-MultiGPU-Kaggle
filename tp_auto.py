"""Automatic tensor parallel: every sampling run on a supported model uses both GPUs, whatever loaded the model.

Hooks CFGGuider.sample (KSampler, SamplerCustom, SamplerCustomAdvanced and guiders built on CFGGuider all go through
it). Before ComfyUI collects the model's wrappers, the diffusion model's class picks the tensor-parallel module
(Qwen-Image 2.1, Krea 2, LTX 2.x, MiniMax H3); its shards come from the loaded model itself (tp_source.ModelWeights:
any loader, any weight format - int8 convrot passes through, others are re-quantized to int8 convrot). One automatic
model keeps shards at a time: another model's shards are dropped when a new one attaches.

MULTIGPU_TP_AUTO=0 turns it off. Models loaded by the UNETLoader*TensorParallel nodes are already attached and are
left as they are. Anything unsupported, or any failure while attaching, samples on one GPU as before (logged).
"""
import gc
import logging
import os

import torch

import comfy.samplers

from . import h3_tensor_parallel, krea2_tensor_parallel, ltx_tensor_parallel, qwen21_tensor_parallel
from .tp_source import ModelWeights

logger = logging.getLogger("MultiGPU")
# (name, module, wrapper key its attach() uses)
MODULES = (("Qwen-Image 2.1", qwen21_tensor_parallel, "qwen21_tp"), ("Krea 2", krea2_tensor_parallel, "krea2_tp"),
           ("LTX 2.x", ltx_tensor_parallel, "ltx_tp"), ("MiniMax H3", h3_tensor_parallel, "mmh3_tp"))
_logged = set()


def _release_others(keep):
    """Drops automatic shards of every other model (and the reference to that model)."""
    freed = []
    for _, mod, _ in MODULES:
        for key in [k for k in mod._INSTANCES if isinstance(k, tuple) and k[0] == "auto" and k != keep]:
            _drop(mod, key)
            freed.append(key)
    if freed:
        gc.collect()
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
    return freed


def _drop(mod, key):
    tp = mod._INSTANCES.pop(key)
    getattr(mod, "_SHARD_CACHE", {}).pop(tp.path, None)   # H3 caches shards by source
    for attr in ("gpu", "host", "shards"):
        if getattr(tp, attr, None) is not None:
            setattr(tp, attr, None)


def _build(tp, mod):
    """Shards now, before sampling, so a model the shard builder cannot read falls back to one GPU cleanly."""
    if hasattr(tp, "shards"):
        if tp.shards is None:
            tp.shards = mod._load_shards(tp.path, len(tp.blocks))
    elif tp.gpu is None:
        tp._load()


def _detach(patcher, dm, mod, key, wkey):
    for blk in list(getattr(dm, "blocks", [])) + list(getattr(dm, "transformer_blocks", [])):
        blk.__dict__.pop("forward", None)
    if hasattr(dm, "txtfusion"):
        dm.txtfusion.__dict__.pop("forward", None)
    dm._mgpu_tp = None
    dit = patcher.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    for k in [k for k in dit if k[0] == "double_block"]:
        dit.pop(k)
    for kind in ("diffusion_model", "outer_sample"):
        patcher.remove_wrappers_with_key(kind, wkey)
    if key in mod._INSTANCES:
        _drop(mod, key)


def attach(patcher):
    dm = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    if dm is None:
        return None
    for name, mod, wkey in MODULES:
        if not mod.supports(dm):
            continue
        tp = getattr(dm, "_mgpu_tp", None)
        key = ("auto", id(dm))
        if tp is not None and key not in mod._INSTANCES:   # attached by a UNETLoader*TensorParallel node
            return tp
        new = key not in mod._INSTANCES
        if new:
            _release_others(key)
            logger.info("[MultiGPU TP auto] %s (%s): both GPUs, shards from the loaded model", name, type(dm).__name__)
        try:
            tp = mod.attach(patcher, ModelWeights(dm, patcher) if new else mod._INSTANCES[key].path, key)
            _build(tp, mod)
            return tp
        except Exception:
            _detach(patcher, dm, mod, key, wkey)
            raise
    if type(dm).__name__ not in _logged:
        _logged.add(type(dm).__name__)
        logger.info("[MultiGPU TP auto] %s: no tensor-parallel support, one GPU", type(dm).__name__)
    return None


_sample = comfy.samplers.CFGGuider.sample


def sample(self, *args, **kwargs):
    try:
        attach(self.model_patcher)
    except Exception:
        logger.exception("[MultiGPU TP auto] could not attach tensor parallel; this run uses one GPU")
    return _sample(self, *args, **kwargs)


def install():
    if os.environ.get("MULTIGPU_TP_AUTO", "1") == "0" or torch.cuda.device_count() < 2:
        return False
    if comfy.samplers.CFGGuider.sample is not sample:
        comfy.samplers.CFGGuider.sample = sample
    logger.info("[MultiGPU TP auto] on: supported models (%s) sample on both GPUs", ", ".join(m[0] for m in MODULES))
    return True
