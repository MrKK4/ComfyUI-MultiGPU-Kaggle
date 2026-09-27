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
import weakref

logger = logging.getLogger("MultiGPU")


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
    # The encoders are held weakly: a hit needs the very same loaded objects, and the cache never keeps a model alive.
    cache = {}

    def cached_execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                       ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        if ref_videos or ref_video_audios or ref_audios:
            return execute(cls, clip, prompt, width, height, length, ref_image_size, vae, audio_vae,
                           ref_images, ref_videos, ref_video_audios, ref_audios)
        images = [img for img in (ref_images or {}).values() if img is not None]
        key = (prompt, ref_image_size, (width, height) if ref_image_size == "match" else None,
               tuple((tuple(img.shape), hashlib.sha1(img[:1].contiguous().cpu().numpy().tobytes()).hexdigest()) for img in images))
        if cache.get("key") == key and cache["clip"]() is clip and cache["vae"]() is vae:
            logger.info("[MultiGPU] H3 Ref2VA: reusing the cached prompt + reference encode")
            latent, _ = nm._empty_av_latent(width, height, length)
            return io.NodeOutput(cache["cond"], latent)
        out = execute(cls, clip, prompt, width, height, length, ref_image_size, vae, audio_vae,
                      ref_images, ref_videos, ref_video_audios, ref_audios)
        cache.update(key=key, clip=weakref.ref(clip), vae=weakref.ref(vae) if vae is not None else (lambda: None), cond=out.args[0])
        return out

    node.execute = classmethod(cached_execute)
    node._mmh3_ref_cache = True
    logger.info("[MultiGPU] H3 Ref2VA prompt + reference encode cache enabled; MMH3_REF_CACHE=0 disables")
    return True
