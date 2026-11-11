"""Expose source-free H3 projections to ComfyUI's ordinary LoRA loader."""
import logging
import uuid

import comfy.lora
import comfy.model_patcher
import comfy.weight_adapter


_TARGETS = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')
TP_KEYS = frozenset(
    f'diffusion_model.blocks.{block}.{target}.weight'
    for block in range(50) for target in _TARGETS
)
_ATTACHMENT = 'mmh3_bundle_tp_lora'
_TP_MODULES = frozenset(key[:-len('.weight')] for key in TP_KEYS)
_BRIDGE_INSTALLED = False
logger = logging.getLogger('MultiGPU')


class H3BundleLoraError(RuntimeError):
    pass


def _key_name(key):
    return key if isinstance(key, str) else key[0]


def _targets_tp_module(key):
    return isinstance(key, str) and key.rpartition('.')[0] in _TP_MODULES


def _validate_tp_patch(key, value, strength_model):
    if not isinstance(key, str) or key not in TP_KEYS:
        raise H3BundleLoraError(f'H3 bundle TP LoRA cannot patch {key!r}; only projection weights are supported')
    if strength_model != 1.0 or not isinstance(value, comfy.weight_adapter.LoRAAdapter):
        raise H3BundleLoraError(f'H3 bundle TP LoRA requires a plain LoRA and model strength 1: {key}')
    weights = value.weights
    if len(weights) != 6:
        raise H3BundleLoraError(f'H3 bundle TP LoRA has an unexpected weight format: {key}')
    up, down, _alpha, mid, dora, reshape = weights
    if mid is not None or dora is not None or reshape is not None:
        raise H3BundleLoraError(f'H3 bundle TP LoRA cannot apply LoCon, DoRA, or reshaped weights: {key}')
    if up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]:
        raise H3BundleLoraError(f'H3 bundle TP LoRA has incompatible matrix dimensions: {key}')


class _VirtualPatches:
    def __init__(self, patches=None):
        self.patches = patches or {}

    def on_model_patcher_clone(self):
        return _VirtualPatches({key: values[:] for key, values in self.patches.items()})


class H3BundleModelPatcher(comfy.model_patcher.CoreModelPatcher):
    """Keep TP LoRAs off ComfyUI's weight-patching path, including low-VRAM loads."""

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        ordinary, virtual = {}, {}
        for key, value in patches.items():
            if _targets_tp_module(_key_name(key)):
                _validate_tp_patch(key, value, strength_model)
                virtual[key] = value
            else:
                ordinary[key] = value
        accepted = super().add_patches(ordinary, strength_patch, strength_model)
        if virtual:
            attachment = self.get_attachment(_ATTACHMENT)
            if attachment is None:
                attachment = _VirtualPatches()
                self.set_attachments(_ATTACHMENT, attachment)
            for key, value in virtual.items():
                attachment.patches.setdefault(key, []).append(
                    (strength_patch, value, strength_model, None, None))
            self.patches_uuid = uuid.uuid4()
            accepted.extend(virtual)
            logger.info('[MultiGPU TP] H3 bundle LoRA accepted %d projection patches', len(virtual))
        return accepted

    def add_hook_patches(self, hook, patches, *args, **kwargs):
        tp = [key for key in patches if _targets_tp_module(_key_name(key))]
        if tp:
            raise H3BundleLoraError(
                f'H3 bundle TP does not support scheduled LoRA patches ({len(tp)} projection keys)')
        return super().add_hook_patches(hook, patches, *args, **kwargs)


def sampling_patches(patcher):
    """Combine ordinary companion patches and TP-only patches for the sampler."""
    merged = {key: values[:] for key, values in patcher.patches.items()}
    virtual = patcher.get_attachment(_ATTACHMENT)
    if virtual is None:
        return merged
    merged.update({key: values[:] for key, values in virtual.patches.items()})
    return merged


def install_lora_key_bridge():
    """Add aliases only for a source-free H3 model; other models keep ComfyUI behavior."""
    global _BRIDGE_INSTALLED
    if _BRIDGE_INSTALLED:
        return
    original = comfy.lora.model_lora_keys_unet

    def model_lora_keys_unet(model, key_map=None):
        if key_map is None:
            key_map = {}
        key_map = original(model, key_map)
        if getattr(model, '_mmh3_bundle_virtual_lora', False):
            for key in TP_KEYS:
                if key in key_map.values():
                    continue
                stem = key[len('diffusion_model.'):-len('.weight')]
                flat = stem.replace('.', '_')
                for alias in ('lora_unet_' + flat, 'lycoris_' + flat,
                              key[:-len('.weight')], 'model.diffusion_model.' + stem,
                              'transformer.' + stem, 'base_model.model.' + stem, stem):
                    key_map.setdefault(alias, key)
        return key_map

    model_lora_keys_unet.__mmh3_bundle_lora_bridge__ = True
    comfy.lora.model_lora_keys_unet = model_lora_keys_unet
    _BRIDGE_INSTALLED = True
