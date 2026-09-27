"""Dual-GPU LTX 2.x diffusion-VAE decode (Kaggle 2x T4, no P2P).

The latent is split in two along width, each half keeping `overlap` pixels of context past the
middle. The left half decodes on the VAE's device, the right half on a resident helper copy of the
same VAE on the other GPU, in parallel threads. Each half runs ComfyUI's tiled 3D decode loop
(tiled_scale_multidim, temporal tiles only), then the halves are blended linearly over the overlap.
Both VAEs are loaded on the calling thread first: model management is not thread-safe.
"""
import logging
import threading

import torch

import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
import comfy.model_management
import comfy.utils
import folder_paths

from .h3_dual_vae import _helper

logger = logging.getLogger("MultiGPU")


def _decode_half(vae, z, tile_t, overlap_t, out, err):
    try:
        with torch.inference_mode(), torch.cuda.device(vae.device):
            fn = lambda a: vae.first_stage_model.decode(a.to(vae.device, vae.vae_dtype)).to(dtype=vae.vae_output_dtype())
            px = comfy.utils.tiled_scale_multidim(z, fn, tile=(tile_t, z.shape[3], z.shape[4]), overlap=(overlap_t, 1, 1),
                                                  upscale_amount=vae.upscale_ratio, out_channels=vae.output_channels,
                                                  index_formulas=vae.upscale_index_formula, output_device=vae.output_device)
            out.append(vae.process_output(px))
    except BaseException as e:  # re-raised on the caller's thread
        err.append(e)


class VAEDecodeLTXDualGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",),
                             "helper_vae_name": (folder_paths.get_filename_list("vae"),),
                             "helper_device": (["cuda:0", "cuda:1"],),
                             "overlap": ("INT", {"default": 64, "min": 32, "max": 512, "step": 32}),
                             "temporal_size": ("INT", {"default": 4096, "min": 16, "max": 4096, "step": 8}),
                             "temporal_overlap": ("INT", {"default": 16, "min": 8, "max": 4096, "step": 8})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "multigpu"
    DESCRIPTION = ("LTX 2.x video decode split left/right over two GPUs: helper_vae_name is the same VAE file, kept resident on "
                   "helper_device. overlap in pixels; temporal_size/overlap in frames (lower temporal_size if a half runs out of VRAM).")

    def decode(self, samples, vae, helper_vae_name, helper_device, overlap, temporal_size, temporal_overlap):
        z = samples["samples"]
        if z.is_nested:
            z = z.unbind()[0]
        tile_t = max(2, temporal_size // 8)
        overlap_t = max(1, min(tile_t // 2, temporal_overlap // 8))
        ov = overlap // 32
        w = z.shape[4]
        if not isinstance(vae.first_stage_model, nd.CausalDiffusionVAE) or torch.device(helper_device) == torch.device(vae.device) or w < 4 * ov:
            logger.info("[MultiGPU] VAEDecodeLTXDualGPU: single-device decode (not an LTX diffusion VAE, helper on the VAE's device, or too narrow)")
            images = vae.decode_tiled(z, tile_x=w, tile_y=z.shape[3], overlap=1, tile_t=tile_t, overlap_t=overlap_t)
        else:
            mid = w // 2
            left, right = z[..., :mid + ov], z[..., mid - ov:]
            shape = list(right.shape)
            shape[2] = min(shape[2], tile_t)
            mem = vae.memory_used_decode(shape, vae.vae_dtype)
            helper = _helper(helper_vae_name, helper_device)
            comfy.model_management.load_models_gpu([vae.patcher], memory_required=mem, force_full_load=True)
            comfy.model_management.load_models_gpu([helper.patcher], memory_required=mem, force_full_load=True)
            out_r, err = [], []
            t = threading.Thread(target=_decode_half, args=(helper, right, tile_t, overlap_t, out_r, err))
            t.start()
            out_l = []
            _decode_half(vae, left, tile_t, overlap_t, out_l, err)
            t.join()
            if err:
                raise err[0]
            l, r = out_l[0], out_r[0]
            x0, n = (mid - ov) * 32, 2 * ov * 32
            px = torch.empty(l.shape[:4] + (w * 32,), dtype=l.dtype, device=l.device)
            px[..., :x0] = l[..., :x0]
            px[..., x0 + n:] = r[..., n:]
            ramp = torch.linspace(0.0, 1.0, n, dtype=l.dtype, device=l.device)
            px[..., x0:x0 + n] = torch.lerp(l[..., x0:], r[..., :n], ramp)
            images = px.movedim(1, -1)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)
