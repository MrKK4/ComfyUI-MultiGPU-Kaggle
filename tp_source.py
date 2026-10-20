"""Weights for the tensor-parallel shards from an already loaded model, whatever loaded it.

ModelWeights looks like a safetensors handle of a Comfy-Org int8 convrot checkpoint (keys / get_tensor / get_slice /
metadata), so every *_tensor_parallel shard builder reads it unchanged:
  - int8 convrot weights (Comfy-Org layout) are passed through exactly (qdata, per-row scale);
  - any other linear weight (bf16 / fp16 / fp8 / other QuantizedTensor layouts / GGUF) is dequantized and re-quantized
    to int8 convrot (group 256) on GPU 0 - the format every shard path computes in. QWEN/KREA/... checkpoints in bf16
    then fit on 2x T4 (half of an int8 model per GPU);
  - norms, tables and other small tensors come back dense.
Weights ComfyUI has patched in place (non-dynamic load with LoRA) are read from the patcher's backup, so the shards
are always the base weights; LoRA stays a side path as with file shards.
"""
import importlib
import json
import logging
import os
from collections import OrderedDict

import torch

logger = logging.getLogger("MultiGPU")
GROUP = 256
REQUANT_ALL = os.environ.get("MULTIGPU_TP_REQUANT") == "1"   # test: treat int8 convrot weights like any other format


def open_weights(source):
    """safetensors handle for a checkpoint path, or `source` itself (a ModelWeights)."""
    if isinstance(source, str):
        from safetensors import safe_open
        return safe_open(source, framework="pt", device="cpu")
    return source


def _dense(w):
    if hasattr(w, "_layout_cls") and hasattr(w, "dequantize"):   # comfy-kitchen QuantizedTensor (fp8, nvfp4, int8, ...)
        return w.dequantize()
    if hasattr(w, "tensor_type") and type(w).__name__ == "GGMLTensor":   # ComfyUI-GGUF
        pkg = type(w).__module__.rsplit(".", 1)[0]
        return importlib.import_module(pkg + ".dequant").dequantize_tensor(w, torch.float16)
    return w


class _Slice:
    def __init__(self, shape):
        self.shape = list(shape)

    def get_shape(self):
        return self.shape


class ModelWeights:
    def __init__(self, diffusion_model, patcher=None, device=None):
        self.dm = diffusion_model
        self.backup = getattr(patcher, "backup", None) or {}
        self.device = device or torch.device("cuda:0")
        self.params = dict(diffusion_model.named_parameters())
        self.params.update(diffusion_model.named_buffers())
        self.linears = {n for n, m in diffusion_model.named_modules()
                        if isinstance(getattr(m, "weight", None), torch.Tensor) and getattr(m, "in_features", None) is not None
                        and len(self._shape(n + ".weight")) == 2}
        self.quant = {}   # linear name -> True when it is (or will be) int8 convrot
        for n in self.linears:
            k = self._shape(n + ".weight")[-1]
            self.quant[n] = self._native_int8(n) or k % GROUP == 0
        self._cache = OrderedDict()
        self.requantized = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cache.clear()
        return False

    # ----------------------------------------------------------------------------------------- safetensors interface

    def keys(self):
        out = list(self.params)
        for n, q in self.quant.items():
            if q:
                out += [n + ".weight_scale", n + ".comfy_quant"]
        return out

    def metadata(self):
        layers = {n: {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP} for n, q in self.quant.items() if q}
        return {"_quantization_metadata": json.dumps({"layers": layers})}

    def get_slice(self, name):
        return _Slice(self._shape(name))

    def get_tensor(self, name):
        if name.endswith(".comfy_quant"):
            conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()
            return torch.tensor(list(conf), dtype=torch.uint8)
        for suffix, i in ((".weight_scale", 1), (".weight", 0)):
            lin = name[:-len(suffix)] if name.endswith(suffix) else None
            if lin in self.linears and self.quant[lin]:
                return self._int8(lin)[i]
        return _dense(self._raw(name)).detach().cpu()

    # ----------------------------------------------------------------------------------------------------- internals

    def _raw(self, name):
        b = self.backup.get(name)
        if b is not None:   # ComfyUI keeps (weight, inplace_update) of weights it patched in place
            return b[0] if isinstance(b, tuple) else getattr(b, "weight", b)
        return self.params[name]

    def _shape(self, name):
        w = self.params[name]
        return tuple(getattr(w, "tensor_shape", None) or w.shape)

    def _native_int8(self, lin):
        if REQUANT_ALL:
            return False
        w = self.params[lin + ".weight"]
        p = getattr(w, "_params", None)
        return (getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout" and getattr(p, "convrot", False)
                and int(getattr(p, "convrot_groupsize", 0)) == GROUP and not getattr(p, "transposed", False))

    def _int8(self, lin):
        hit = self._cache.get(lin)
        if hit is None:
            w = self._raw(lin + ".weight")
            if self._native_int8(lin):
                hit = (w._qdata.detach().cpu(), w._params.scale.detach().float().cpu().reshape(-1, 1))
            else:   # ponytail: re-quantized per run of the shard builder, ~1-3 s per 1 B params on a T4
                import comfy_kitchen.backends.cuda as ck_cuda
                dense = _dense(w).detach().to(self.device, torch.float16)
                q, s = ck_cuda.quantize_int8_convrot_weight(dense, GROUP)
                hit = (q.cpu(), s.float().cpu().reshape(-1, 1))
                self.requantized += 1
                del dense, q, s
            self._cache[lin] = hit
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
        return hit
