"""Explicit H3 bundle eviction barriers for preprocessing and VAE work."""

from .h3_tensor_parallel import evict_bundle


class _EvictH3:
    CATEGORY = 'multigpu'
    FUNCTION = 'evict'
    PAYLOAD_TYPE = None

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',),
            'payload': (cls.PAYLOAD_TYPE,),
            'free_gpu0_gib': ('FLOAT', {'default':0.0,'min':0.0,'max':24.0,'step':0.1}),
            'free_gpu1_gib': ('FLOAT', {'default':0.0,'min':0.0,'max':24.0,'step':0.1}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The residency can change without an input changing after a sampler
        # restores its pieces. Comfy must execute this barrier every run.
        return float('nan')

    def evict(self, model, payload, free_gpu0_gib, free_gpu1_gib):
        evict_bundle(model, (free_gpu0_gib, free_gpu1_gib))
        return (payload,)


class TPEvictH3Image(_EvictH3):
    PAYLOAD_TYPE = 'IMAGE'
    RETURN_TYPES = ('IMAGE',)


class TPEvictH3Latent(_EvictH3):
    PAYLOAD_TYPE = 'LATENT'
    RETURN_TYPES = ('LATENT',)
