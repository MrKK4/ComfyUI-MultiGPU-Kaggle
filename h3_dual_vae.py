"""Dual-GPU MiniMax H3 video VAE decode / encode (Kaggle 2x T4, no P2P).

The H3 video decoder works in independent temporal clips (overlap tokens included in each clip;
the cross-clip blend happens afterwards). VAEDecodeH3DualGPU decodes clip k on the VAE's own
device and clip k+1 on a helper copy of the same VAE on the other GPU, in parallel threads; the
stock decode_temporal then blends and writes them exactly as before. The encoder's clips are
independent too: VAEEncodeH3DualGPU encodes the odd clips on the helper. Each helper keeps only the
half of the VAE it runs, so it fits next to the tensor-parallel shards.
"""
import logging
import math
import threading

import torch

import comfy.ldm.minimax.vae as hv
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths

from .h3_vae_tiles import reserve_extra_tiles

logger = logging.getLogger("MultiGPU")
_orig_decode_temporal = hv.MiniMaxH3VideoVAE.decode_temporal
_orig_encode_temporal = hv.MiniMaxH3VideoVAE.encode_temporal

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
        reserve_extra_tiles(dev)          # spatial tile batching wants room before it starts
        # grad mode is thread-local: ComfyUI runs nodes under inference_mode on its own thread only
        with torch.inference_mode(), torch.cuda.device(dev):
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


def _encode_clip(fsm, clip, dev):
    # mirrors encode_temporal: last clip padded with its final frame
    clip = clip.to(dev)
    if clip.shape[2] < fsm.clip_length:
        clip = torch.cat([clip, clip[:, :, -1:].repeat(1, 1, fsm.clip_length - clip.shape[2], 1, 1)], dim=2)
    return fsm._adaptive_encode(fsm._normalize_pixels(clip))


def encode_temporal(self, x, device):
    helper = getattr(self, "_mmh3_helper", None)
    if helper is None:
        return _orig_encode_temporal(self, x, device)
    step = self.clip_length
    clips = [x[:, :, i * step:(i + 1) * step] for i in range(math.ceil(x.shape[2] / step))]
    out, err = {}, []

    def odd_clips():
        try:
            with torch.inference_mode(), torch.cuda.device(helper.device):
                for i in range(1, len(clips), 2):
                    out[i] = _encode_clip(helper.first_stage_model, clips[i], helper.device).to(device)
                torch.cuda.synchronize(helper.device)
        except BaseException as e:  # re-raised on the caller's thread
            err.append(e)

    t = threading.Thread(target=odd_clips)
    t.start()
    for i in range(0, len(clips), 2):
        out[i] = _encode_clip(self, clips[i], device)
    t.join()
    if err:
        raise err[0]
    z = torch.cat([out[i] for i in range(len(clips))], dim=2)
    if self.token_drop > 0:
        z = z[:, :, :-self.token_drop]
    return z


hv.MiniMaxH3VideoVAE.encode_temporal = encode_temporal


_HELPERS = {}


def _helper(vae_name, device, part):
    """Plain, fully resident copy of the VAE's `part` ("encoder" or "decoder") on `device`. A dynamic-VRAM
    copy streams its weights from the file inside the worker thread, which fails ("HostBuffer.read_file_slice failed")."""
    key = (vae_name, str(device), part)
    if key not in _HELPERS:
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name), return_metadata=True)
        dynamic = comfy.model_patcher.CoreModelPatcher
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcher
        try:
            helper = comfy.sd.VAE(sd=sd, device=torch.device(device), metadata=metadata)
        finally:
            comfy.model_patcher.CoreModelPatcher = dynamic
        fsm = helper.first_stage_model
        if part == "decoder":
            fsm.encoder = fsm.quant_conv = None
        else:
            fsm.decoder = fsm.post_quant_conv = None
        helper.patcher.size = 0  # re-measured without the dropped half
        _HELPERS[key] = helper
        logger.info("[MultiGPU] H3 helper VAE %s resident on %s", part, device)
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
            helper = _helper(helper_vae_name, helper_device, "decoder")
            comfy.model_management.load_models_gpu([helper.patcher], force_full_load=True)
            fsm._mmh3_helper = helper
        else:
            logger.info("[MultiGPU] VAEDecodeH3DualGPU: single-device decode (not H3 video, or helper on the VAE's device)")
        reserve_extra_tiles(vae.device)
        try:
            images = vae.decode(latent)
        finally:
            fsm._mmh3_helper = None
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


class VAEEncodeH3DualGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE",), "vae": ("VAE",),
                             "helper_vae_name": (folder_paths.get_filename_list("vae"),),
                             "helper_device": (["cuda:0", "cuda:1"],)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "multigpu"
    DESCRIPTION = "MiniMax H3 video encode split over two GPUs: helper_vae_name is the same VAE file, its encoder kept resident on helper_device."

    def encode(self, pixels, vae, helper_vae_name, helper_device):
        fsm = vae.first_stage_model
        usable = isinstance(fsm, hv.MiniMaxH3VideoVAE) and torch.device(helper_device) != torch.device(vae.device) and pixels.shape[0] > fsm.clip_length
        if usable:
            helper = _helper(helper_vae_name, helper_device, "encoder")
            comfy.model_management.load_models_gpu([helper.patcher], force_full_load=True)
            fsm._mmh3_helper = helper
        else:
            logger.info("[MultiGPU] VAEEncodeH3DualGPU: single-device encode (not H3 video, too short, or helper on the VAE's device)")
        try:
            t = vae.encode(pixels)
        finally:
            fsm._mmh3_helper = None
        return ({"samples": t},)
