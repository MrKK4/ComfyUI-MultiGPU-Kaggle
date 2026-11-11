"""Notebook opt-in: expose one Olivia bundle choice and free managed VRAM."""

import logging
import math
import os
from types import SimpleNamespace

import torch

from .h3_bundle_loader import load_h3_bundle
from .h3_bundle_source import H3BundleSource
from . import h3_tensor_parallel

logger = logging.getLogger('MultiGPU')
_INSTALLED = False


def _bundle_args():
    names = ('MMH3_TP_BUNDLE_DIR', 'MMH3_TP_PIECES_DIR', 'MMH3_TP_BUNDLE_MANIFEST_SHA256')
    values = tuple(os.environ.get(name, '') for name in names)
    if not all(values):
        raise RuntimeError('H3 auto bundle needs bundle, pieces, and pinned manifest paths in the notebook environment')
    return values


def _active_bundle():
    active = [tp for tp in h3_tensor_parallel._INSTANCES.values()
              if isinstance(getattr(tp, 'path', None), H3BundleSource)
              and getattr(tp, 'shards', None) is not None]
    if len(active) > 1:
        raise RuntimeError('More than one H3 bundle is resident; automatic eviction is ambiguous')
    return active[0] if active else None


def install():
    """Install only in the H3 notebook after its bundle paths are configured."""
    global _INSTALLED
    if _INSTALLED or os.environ.get('MMH3_TP_BUNDLE_AUTO') != '1':
        return _INSTALLED
    _bundle_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError('H3 auto bundle needs two CUDA devices')

    import folder_paths
    import nodes
    import comfy.model_management as mm

    bundle_name = os.environ.get('MMH3_TP_BUNDLE_NAME', '').strip()
    if not bundle_name or '/' in bundle_name or '\\' in bundle_name or not bundle_name.endswith('.safetensors'):
        raise RuntimeError('H3 notebook needs one virtual Olivia bundle filename')
    def is_bundle_name(name):
        return name == bundle_name

    original_list = folder_paths.get_filename_list
    if bundle_name in original_list('diffusion_models'):
        raise RuntimeError('H3 bundle choice conflicts with an installed diffusion model: ' + bundle_name)
    def filename_list(category):
        names = original_list(category)
        if category != 'diffusion_models':
            return names
        return sorted(set(names).union((bundle_name,)))
    folder_paths.get_filename_list = filename_list

    original_unet_load = nodes.UNETLoader.load_unet
    def load_unet(self, unet_name, weight_dtype):
        if not is_bundle_name(unet_name):
            return original_unet_load(self, unet_name, weight_dtype)
        if weight_dtype != 'default':
            raise ValueError('H3 split bundle uses its pinned int8 dtype; set weight_dtype to default')
        logger.info('[TP-BUNDLE-AUTO] %s -> Olivia split bundle', unet_name)
        return (load_h3_bundle(*_bundle_args()),)
    # Comfy Remote wraps this loader later. Give it the actual routing predicate
    # and bundle identity so it can preserve only this bundle's warm shards.
    load_unet.__mmh3_bundle_router__ = SimpleNamespace(
        is_routed=is_bundle_name,
        bundle_key=lambda _name: H3BundleSource(*_bundle_args()),
    )
    nodes.UNETLoader.load_unet = load_unet

    # ComfyUI skips built-in combo membership validation for arguments named
    # by VALIDATE_INPUTS. This permits a saved H3 workflow whose checkpoint is
    # absent locally to reach the bundle route, while keeping ordinary names
    # subject to the normal file list.
    def validate_unet_name(unet_name):
        if is_bundle_name(unet_name):
            return True
        return unet_name in original_list('diffusion_models') or 'Model file is unavailable'
    nodes.UNETLoader.VALIDATE_INPUTS = staticmethod(validate_unet_name)

    original_tp_load = h3_tensor_parallel.UNETLoaderH3TensorParallel.load
    def load_tp(self, unet_name):
        if not is_bundle_name(unet_name):
            return original_tp_load(self, unet_name)
        logger.info('[TP-BUNDLE-AUTO] %s -> Olivia split bundle (TP loader)', unet_name)
        return (load_h3_bundle(*_bundle_args()),)
    h3_tensor_parallel.UNETLoaderH3TensorParallel.load = load_tp
    h3_tensor_parallel.UNETLoaderH3TensorParallel.VALIDATE_INPUTS = staticmethod(validate_unet_name)

    original_free = mm.free_memory
    def free_memory(memory_required, device, *args, **kwargs):
        result = original_free(memory_required, device, *args, **kwargs)
        target_device = torch.device(device) if device is not None else None
        if target_device is None or target_device.type != 'cuda':
            return result
        rank = target_device.index if target_device.index is not None else torch.cuda.current_device()
        if rank not in (0, 1) or not math.isfinite(memory_required) or memory_required <= 0:
            return result
        tp = _active_bundle()
        if tp is None or h3_tensor_parallel._BUNDLE_LEASES[tp.path]:
            return result
        free, total = torch.cuda.mem_get_info(rank)
        # Huge requests are ComfyUI's "unload everything" sentinel, not a
        # real allocation. Preserve that behavior without evicting the bundle.
        if memory_required > total:
            return result
        target = min(total, int(memory_required) + (256 << 20))
        if free < target:
            targets = [0.0, 0.0]
            targets[rank] = target / (1 << 30)
            released = h3_tensor_parallel.evict_bundle_tp(tp, targets)
            logger.info('[TP-BUNDLE-AUTO] GPU%d managed request %.2fGiB; evicted %d pieces',
                        rank, memory_required / (1 << 30), released[rank])
        return result
    mm.free_memory = free_memory

    _INSTALLED = True
    logger.info('[TP-BUNDLE-AUTO] Olivia bundle choice: %s; other checkpoints load their own weights', bundle_name)
    return True
