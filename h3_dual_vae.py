"""Dual-GPU MiniMax H3 video VAE decode (Kaggle 2x T4, no P2P).

The H3 video decoder works in independent temporal clips (overlap tokens included in each clip;
the cross-clip blend happens afterwards). VAEDecodeH3DualGPU decodes clip k on the VAE's own
device and clip k+1 on a helper copy of the same VAE on the other GPU, in parallel threads; the
stock decode_temporal then blends and writes them exactly as before.
"""
import logging
import threading

import torch

import comfy.ldm.minimax.vae as hv
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger("MultiGPU")
_orig_decode_temporal = hv.MiniMaxH3VideoVAE.decode_temporal

def _clips(vae, z):
    # mirrors decode_temporal's padding and clip slicing
    pad_tokens, num_chunks = vae._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    step, span = vae.tokens_chunk_size, vae.tokens_chunk_size + vae.token_overlap
    return [z[:, :, i * step:i * step + span] for i in range(num_chunks)]


def _decode_on(helper, clip_z, home, out, err):
    try:
        dev = helper.device
        with torch.cuda.device(dev):
            dec = helper.first_stage_model._adaptive_decode(clip_z.to(dev))
            out.append(dec.to(home))
            torch.cuda.synchronize(dev)
    except BaseException as e:  # re-raised on the caller's thread
        err.append(e)


def decode_temporal(self, z, output_buffer=None):
    helper = getattr(self, "_mmh3_helper", None)
    if helper is None:
        return _orig_decode_temporal(self, z, output_buffer)
    clips = _clips(self, z)
    ready = {}
    calls = [0]
    own = self._adaptive_decode

    def provide(clip_z):
        k = calls[0]
        calls[0] += 1
        if k not in ready:
            out, err = [], []
            t = None
            if k + 1 < len(clips):
                t = threading.Thread(target=_decode_on, args=(helper, clips[k + 1], clip_z.device, out, err))
                t.start()
            ready[k] = own(clip_z)
            if t is not None:
                t.join()
                if err:
                    raise err[0]
                ready[k + 1] = out[0]
        return ready.pop(k)

    self._adaptive_decode = provide
    try:
        return _orig_decode_temporal(self, z, output_buffer)
    finally:
        del self._adaptive_decode


hv.MiniMaxH3VideoVAE.decode_temporal = decode_temporal


_HELPERS = {}


def _helper(vae_name, device):
    """Plain, fully resident copy of the VAE on `device`. A dynamic-VRAM copy streams its weights
    from the file inside the decode thread, which fails ("HostBuffer.read_file_slice failed")."""
    key = (vae_name, str(device))
    if key not in _HELPERS:
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name), return_metadata=True)
        dynamic = comfy.model_patcher.CoreModelPatcher
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcher
        try:
            _HELPERS[key] = comfy.sd.VAE(sd=sd, device=torch.device(device), metadata=metadata)
        finally:
            comfy.model_patcher.CoreModelPatcher = dynamic
        logger.info("[MultiGPU] H3 helper VAE resident on %s", device)
    return _HELPERS[key]


class VAEDecodeH3DualGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",),
                             "helper_vae_name": (folder_paths.get_filename_list("vae"),),
                             "helper_device": (["cuda:0", "cuda:1"],)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "multigpu"
    DESCRIPTION = "MiniMax H3 video decode split over two GPUs: helper_vae_name is the same VAE file, kept resident on helper_device."

    def decode(self, samples, vae, helper_vae_name, helper_device):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        fsm = vae.first_stage_model
        usable = isinstance(fsm, hv.MiniMaxH3VideoVAE) and torch.device(helper_device) != torch.device(vae.device) and latent.ndim == 5 and latent.shape[2] > 1
        if usable:
            helper = _helper(helper_vae_name, helper_device)
            comfy.model_management.load_models_gpu([helper.patcher], force_full_load=True)
            fsm._mmh3_helper = helper
        else:
            logger.info("[MultiGPU] VAEDecodeH3DualGPU: single-device decode (not H3 video, or helper on the VAE's device)")
        try:
            images = vae.decode(latent)
        finally:
            fsm._mmh3_helper = None
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)
