"""H3 TP bundle model loader. The 50 split projections are owned by the TP wrapper."""
import hashlib
import json
import logging
import math
import os

import torch
from safetensors.torch import load_file

import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.storage
import comfy.utils
import comfy.ldm.minimax.model as h3_model

from .h3_bundle_source import BundleFileError, H3BundleSource
from .h3_bundle_lora import H3BundleModelPatcher, install_lora_key_bridge
from .h3_tensor_parallel import attach, supports

logger = logging.getLogger('MultiGPU')
_DTYPES = {'U8':torch.uint8,'I8':torch.int8,'I16':torch.int16,'I32':torch.int32,
           'I64':torch.int64,'BF16':torch.bfloat16,'F16':torch.float16,
           'F32':torch.float32,'F64':torch.float64,'BOOL':torch.bool}


class _Shape:
    def __init__(self, shape):
        self.shape = shape


def _reference_sha(module):
    with open(module.__file__, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_h3_bundle(bundle_directory, pieces_directory, manifest_sha256):
    """Construct the ordinary H3 Comfy model from companion weights, then attach TP pieces."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError('H3 TP bundle needs two CUDA devices')
    source = H3BundleSource(bundle_directory, pieces_directory, manifest_sha256)
    manifest, companion_path, config = source.validate()
    if manifest['n_blocks'] != 50:
        raise BundleFileError('H3 generation loader expects 50 transformer blocks')
    if (_reference_sha(comfy.model_detection) != config['model_detection_sha256']
            or _reference_sha(h3_model) != config['model_constructor_sha256']):
        raise BundleFileError('ComfyUI H3 detector/constructor differs from the pinned bundle build')
    prefix = manifest['key_prefix']
    catalog = manifest['original_tensors']
    original_shapes = {row['name'][len(prefix):]:_Shape(row['shape']) for row in catalog}
    detected = comfy.model_detection.detect_unet_config(original_shapes, '', metadata=manifest['safetensors_metadata'])
    if detected != config['detected_unet_config']:
        raise BundleFileError('ComfyUI detected H3 configuration differs from bundle')
    model_config = comfy.model_detection.model_config_from_unet_config(detected, original_shapes, '')
    if model_config is None or model_config.__class__.__name__ != 'MiniMaxH3':
        raise BundleFileError('ComfyUI did not select MiniMaxH3')
    companion = load_file(str(companion_path), device='cpu')
    if prefix:
        companion = {name[len(prefix):]:value for name,value in companion.items()}
    tp_modules = {row['module'] for row in manifest['tp_modules']}
    tp_quant = {module+'.comfy_quant' for module in tp_modules}
    if not tp_quant <= set(companion):
        raise BundleFileError('H3 TP quant metadata is missing')
    # Original quant metadata remains in the bundle, but projection weights are
    # never passed to Comfy's module loader. Its quant loader requires a real
    # weight before consuming comfy_quant; these projections execute only in TP.
    load_dict = {name:value for name,value in companion.items() if name not in tp_quant}
    converted, metadata = comfy.utils.convert_old_quants(load_dict, '', metadata=manifest['safetensors_metadata'])
    if set(converted) != set(load_dict):
        raise BundleFileError('ComfyUI changed companion keys during quant conversion')
    model_config.quant_config = comfy.utils.detect_layer_quantization(companion, '')
    if model_config.quant_config != {'mixed_ops':True}:
        raise BundleFileError('ComfyUI did not detect H3 mixed quantization')
    parameters = sum(math.prod(row['shape']) for row in catalog)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    dtype = comfy.model_management.unet_dtype(model_params=parameters,
        supported_dtypes=list(model_config.supported_inference_dtypes),weight_dtype=None)
    manual_cast = comfy.model_management.unet_manual_cast(None,load_device,
        model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(dtype,manual_cast,device=load_device)
    model = model_config.get_model(converted, '')
    install_lora_key_bridge()
    model._mmh3_bundle_virtual_lora = True
    patcher = H3BundleModelPatcher(model,load_device=load_device,
        offload_device=offload_device,fast_disk=comfy.storage.state_dict_fast_disk(converted))
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    to_load = model_config.process_unet_state_dict(converted)
    missing,unexpected = model.diffusion_model.load_state_dict(to_load,strict=False,assign=patcher.is_dynamic())
    if missing or unexpected:
        raise BundleFileError(f'H3 companion load mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}')
    for module_name in tp_modules:
        module = model.diffusion_model.get_submodule(module_name)
        if getattr(module,'weight','absent') is not None or getattr(module,'bias',None) is not None:
            raise BundleFileError('H3 TP projection unexpectedly owns a full weight: '+module_name)
    if not supports(model.diffusion_model):
        raise BundleFileError('H3 bundle constructed an unexpected model')
    patcher.cached_patcher_init = (load_h3_bundle,(bundle_directory,pieces_directory,manifest_sha256))
    tp = attach(patcher,source,source)
    tp.bundle_manifest = manifest
    logger.info('[TP-BUNDLE] loaded %d companion tensors; %d TP modules; source SHA %s',
                len(companion),len(tp_modules),manifest['source']['sha256'])
    return patcher


class UNETLoaderH3TPBundle:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'bundle_directory': ('STRING', {'default':os.environ.get('MMH3_TP_BUNDLE_DIR','')}),
            'pieces_directory': ('STRING', {'default':os.environ.get('MMH3_TP_PIECES_DIR','')}),
            'bundle_manifest_sha256': ('STRING', {'default':os.environ.get('MMH3_TP_BUNDLE_MANIFEST_SHA256','')}),
        }}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'multigpu'

    def load(self,bundle_directory,pieces_directory,bundle_manifest_sha256):
        return (load_h3_bundle(bundle_directory,pieces_directory,bundle_manifest_sha256),)
