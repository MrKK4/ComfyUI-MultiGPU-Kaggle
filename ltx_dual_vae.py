"""Dual-GPU LTX 2.x diffusion-VAE decode (Kaggle 2x T4, no P2P).

The latent is split in two along width, each half keeping `overlap` pixels of context past the
middle. The left half decodes on the VAE's device, the right half on a helper copy of the same VAE
on the other GPU, in parallel threads. Each half runs ComfyUI's tiled 3D decode loop
(tiled_scale_multidim, temporal tiles only), then the halves are blended linearly over the overlap.

VRAM: nothing is evicted beyond a 2-latent-frame tile. Each half's temporal tile is sized to the
VRAM free on its GPU. The helper is not a managed model: loading a plain ModelPatcher fully unloads
the dynamic DiT from that GPU (re-streamed from disk on the next job, ~50 s on Kaggle), so its
weights are moved to the GPU for the decode and back to CPU afterwards.
"""
import logging
import threading

import torch

import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger("MultiGPU")

_HELPERS = {}
_MARGIN = 768 * 1024 * 1024


def _helper(vae_name, device):
    key = (vae_name, str(device))
    if key not in _HELPERS:
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name), return_metadata=True)
        _HELPERS[key] = comfy.sd.VAE(sd=sd, device=torch.device(device), metadata=metadata)
    return _HELPERS[key]


def _fit_tile_t(vae, z, tile_t):
    """Largest temporal tile (latent frames) whose decode estimate fits the VRAM free on vae.device."""
    per_frame = vae.memory_used_decode([1, z.shape[1], 1, z.shape[3], z.shape[4]], vae.vae_dtype)
    free = comfy.model_management.get_free_memory(vae.device) - _MARGIN
    return max(2, min(tile_t, z.shape[2], int(free // per_frame)))


def _decode_half(vae, z, tile_t, overlap_t, out, err):
    try:
        with torch.inference_mode(), torch.cuda.device(vae.device):
            fn = lambda a: vae.first_stage_model.decode(a.to(vae.device, vae.vae_dtype)).to(dtype=vae.vae_output_dtype())
            px = comfy.utils.tiled_scale_multidim(z, fn, tile=(tile_t, z.shape[3], z.shape[4]), overlap=(min(overlap_t, tile_t - 1), 1, 1),
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
    DESCRIPTION = ("LTX 2.x video decode split left/right over two GPUs: helper_vae_name is the same VAE file, used on helper_device. "
                   "overlap in pixels; temporal_size/overlap in frames (upper bound: tiles also shrink to fit free VRAM).")

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
            min_mem = vae.memory_used_decode([1, z.shape[1], 2, z.shape[3], right.shape[4]], vae.vae_dtype)
            comfy.model_management.load_models_gpu([vae.patcher], memory_required=min_mem, force_full_load=True)
            helper = _helper(helper_vae_name, helper_device)
            helper.first_stage_model.to(helper.device)
            try:
                if comfy.model_management.get_free_memory(helper.device) - _MARGIN < min_mem:
                    comfy.model_management.free_memory(min_mem + _MARGIN, helper.device)
                tl, tr = _fit_tile_t(vae, left, tile_t), _fit_tile_t(helper, right, tile_t)
                logger.info("[MultiGPU] VAEDecodeLTXDualGPU: temporal tiles (latent frames of %d): %s %d, %s %d",
                            z.shape[2], vae.device, tl, helper.device, tr)
                out_r, err = [], []
                t = threading.Thread(target=_decode_half, args=(helper, right, tr, overlap_t, out_r, err))
                t.start()
                out_l = []
                _decode_half(vae, left, tl, overlap_t, out_l, err)
                t.join()
            finally:
                helper.first_stage_model.to(comfy.model_management.vae_offload_device())
                with torch.cuda.device(helper.device):
                    torch.cuda.empty_cache()
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
