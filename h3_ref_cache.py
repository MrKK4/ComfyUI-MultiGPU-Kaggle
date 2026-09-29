"""Reuse the MiniMax H3 Ref2VA prompt + reference-image encode across jobs.

MiniMaxH3ReferenceToVideo also takes the output width/height/length, so ComfyUI re-runs the
Qwen3-VL encode (~60 s on 2x T4) for every new driving video in a face swap even when the prompt
and face are unchanged. The conditioning only depends on the prompt, the reference images, the
ref_image_size mode (and the output size for "match"), so it is cached on exactly those plus the
loaded text encoder / VAE; the empty AV latent is rebuilt for the new size. Jobs with reference
videos or audio are not cached. Disable with MMH3_REF_CACHE=0."""
import hashlib
import logging
import os

logger = logging.getLogger("MultiGPU")


def _fingerprint(obj, attr):
    # ComfyUI re-runs the loaders whenever another workflow (face swap step 1) ran in between, so the same
    # files come back as new objects; key on the weights' names/shapes/dtypes instead of object identity.
    # ponytail: two fine-tunes with identical layout and dtype would collide; hash a weight if that ever matters.
    model = getattr(obj, attr, None) if obj is not None else None
    if model is None:
        return None
    return hashlib.sha1(repr([(k, tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()]).encode()).hexdigest()


def patch_minimax_h3_ref_cache():
    if os.environ.get("MMH3_REF_CACHE", "1") == "0":
        return False
    import sys
    import nodes
    from comfy_api.latest import io
    # ComfyUI loads its built-in node files under an internal module name, so patch the registered class
    node = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ReferenceToVideo")
    if node is None:
        return False
    nm = sys.modules[node.__module__]
    if getattr(node, "_mmh3_ref_cache", False):
        return True
    execute = node.execute.__func__
    # ponytail: one entry (the last prompt + face); keep several if people alternate faces.
    # The encoders enter the key by fingerprint only, so the cache never keeps a model alive.
    cache = {}

    def cached_execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                       ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        if ref_videos or ref_video_audios or ref_audios:
            return execute(cls, clip, prompt, width, height, length, ref_image_size, vae, audio_vae,
                           ref_images, ref_videos, ref_video_audios, ref_audios)
        images = [img for img in (ref_images or {}).values() if img is not None]
        key = (prompt, ref_image_size, (width, height) if ref_image_size == "match" else None,
               tuple((tuple(img.shape), hashlib.sha1(img[:1].contiguous().cpu().numpy().tobytes()).hexdigest()) for img in images),
               _fingerprint(clip, "cond_stage_model"), _fingerprint(vae, "first_stage_model"))
        if cache.get("key") == key:
            logger.info("[MultiGPU] H3 Ref2VA: reusing the cached prompt + reference encode")
            latent, _ = nm._empty_av_latent(width, height, length)
            return io.NodeOutput(cache["cond"], latent)
        out = execute(cls, clip, prompt, width, height, length, ref_image_size, vae, audio_vae,
                      ref_images, ref_videos, ref_video_audios, ref_audios)
        cache.update(key=key, cond=out.args[0])
        return out

    node.execute = classmethod(cached_execute)
    node._mmh3_ref_cache = True
    logger.info("[MultiGPU] H3 Ref2VA prompt + reference encode cache enabled; MMH3_REF_CACHE=0 disables")
    return True
