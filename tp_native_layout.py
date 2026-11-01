"""Read ComfyUI's native int8 checkpoint layout (Comfy-Org, most community files) through the interface the tensor-parallel shard builders use.

The shard builders were written for the older layout: block keys without prefix ("blocks.0.attn.qkv_proj.weight") and the per-layer
quantization config in the safetensors header (`_quantization_metadata` -> {"layers": {"blocks.0.attn.qkv_proj": {...}}}).
ComfyUI's native layout stores the same config as one small uint8 tensor per layer (`<layer>.comfy_quant`, JSON text) and prefixes every key
with "model.diffusion_model.". NativeLayoutFile wraps the safetensors handle: it strips that prefix and builds the metadata from the
`.comfy_quant` tensors. Nothing else changes (weights and scales are read from the file as they are); no torch import here.
"""
import json

PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def needs_wrapper(handle):
    """True for a native-layout file: no `_quantization_metadata` in the header, but `.comfy_quant` tensors."""
    if (handle.metadata() or {}).get("_quantization_metadata"):
        return False
    return any(k.endswith(".comfy_quant") for k in handle.keys())


class NativeLayoutFile:
    def __init__(self, handle):
        self.h = handle
        keys = list(handle.keys())
        self.prefix = next((p for p in PREFIXES if keys and all(k.startswith(p) for k in keys if not k.startswith("__"))), "")
        self.full = {k[len(self.prefix):]: k for k in keys}
        self._meta = None

    def __enter__(self):
        enter = getattr(self.h, "__enter__", None)
        if enter:
            enter()
        return self

    def __exit__(self, *exc):
        close = getattr(self.h, "__exit__", None)
        return close(*exc) if close else False

    def keys(self):
        return list(self.full)

    def get_tensor(self, name):
        return self.h.get_tensor(self.full[name])

    def get_slice(self, name):
        return self.h.get_slice(self.full[name])

    def metadata(self):
        if self._meta is None:
            layers = {}
            for name, full in self.full.items():
                if name.endswith(".comfy_quant"):
                    layers[name[:-len(".comfy_quant")]] = json.loads(bytes(self.h.get_tensor(full).tolist()).decode("utf-8"))
            self._meta = {"_quantization_metadata": json.dumps({"layers": layers}), "format": "pt"}
        return self._meta
