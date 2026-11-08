"""Experiment-only native Sol VAE backport; leaves installed ComfyUI and DiT classes unchanged."""
import copy
import importlib.util
import sys
from pathlib import Path
import torch
import comfy.ops
import comfy.model_management as mm
import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths


def load_sol(template,filename):
    name=__name__+'.native'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('native_sol.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    native=sys.modules[name]
    sd,metadata=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise('vae',filename),return_metadata=True)
    if 'blocks.0.scale1' in sd:
        top=('encoder','quant_conv','post_quant_conv','latents_mean','latents_std')
        sd={k if k.split('.')[0] in top else 'decoder.'+k:v for k,v in sd.items()}
        sd.setdefault('latents_mean',torch.tensor(native.LATENTS_MEAN));sd.setdefault('latents_std',torch.tensor(native.LATENTS_STD))
    assert 'decoder.blocks.0.scale1' in sd,'Not the noncausal Sol layout'
    quant=comfy.utils.detect_layer_quantization(sd,'')
    ops=comfy.ops.mixed_precision_ops(quant,torch.float16) if quant is not None else comfy.ops.disable_weight_init
    layers=sum(k.startswith('decoder.blocks.') and k.endswith('.scale1') for k in sd)
    model=native.MiniMaxH3VideoVAE(operations=ops,num_layers=layers,encoder='encoder.conv_in.weight' in sd,
                                 decoder=True,noncausal_decoder=True).eval().to(torch.float16)
    mm.archive_model_dtypes(model)
    result=copy.copy(template)
    result.first_stage_model=model;result.device=torch.device('cuda:1');result.vae_dtype=torch.float16
    result.disable_offload=False;result.size=None;result.output_device=mm.intermediate_device()
    result.patcher=comfy.model_patcher.CoreModelPatcher(model,load_device=result.device,offload_device=mm.vae_offload_device(),
                                                      fast_disk=comfy.storage.state_dict_fast_disk(sd))
    missing,left=model.load_state_dict(sd,strict=False,assign=result.patcher.is_dynamic())
    if missing:raise RuntimeError('Sol missing weights: '+str(missing))
    if any(not k.endswith('.comfy_quant') for k in left):raise RuntimeError('Sol unconsumed weights: '+str(left))
    if not result.patcher.is_dynamic():model.to(torch.float16)
    result.upscale_ratio=(lambda t:4*t-3*(-(-t//5)),16,16)
    result.downscale_ratio=(lambda t:t//17*5+-(-(t%17+3)//4) if t%17 else max(1,t//17*5),16,16)
    result._tile_bounded_shape=lambda shape,*tiles:tuple(shape)
    result.process_output=lambda x:x
    def memory(shape,dtype):
        t,h,w=shape[2:];dec=model.decoder
        block=min(t,dec.block)*h*w+dec.num_global
        return int((t*h*w+block*(dec.ring*4+20))*dec.dim*mm.dtype_size(dtype)*1.1)
    result.memory_used_decode=memory
    result.model_size()
    return result
