"""CPU check of tp_native_layout: a file in ComfyUI's native int8 layout (prefixed keys, `.comfy_quant` tensors, no header metadata) reads like the
older layout the H3 shard builder expects; files that already have `_quantization_metadata` are left alone. Needs numpy only.
usage: python ci/test_native_layout.py"""
import importlib.util
import json
import pathlib

import numpy as np

spec = importlib.util.spec_from_file_location("tp_native_layout", pathlib.Path(__file__).resolve().parents[1] / "tp_native_layout.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CONF = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


class Slice:
    def __init__(self, shape):
        self.shape = shape

    def get_shape(self):
        return self.shape


class FakeFile:
    def __init__(self, tensors, meta=None):
        self.t, self.meta, self.closed = tensors, meta, False

    def keys(self):
        return self.t.keys()

    def metadata(self):
        return self.meta

    def get_tensor(self, k):
        return self.t[k]

    def get_slice(self, k):
        return Slice(list(self.t[k].shape))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def conf_tensor():
    return np.frombuffer(json.dumps(CONF).encode(), dtype=np.uint8)


P = "model.diffusion_model."
native = FakeFile({
    P + "blocks.0.attn.qkv_proj.weight": np.zeros((6, 4), np.int8), P + "blocks.0.attn.qkv_proj.weight_scale": np.ones((6, 1), np.float32),
    P + "blocks.0.attn.qkv_proj.comfy_quant": conf_tensor(), P + "blocks.0.mlp.fc1.comfy_quant": conf_tensor(), P + "blocks.0.norm1.weight": np.ones(4, np.float32)})
assert mod.needs_wrapper(native)
with mod.NativeLayoutFile(native) as f:
    assert "blocks.0.attn.qkv_proj.weight" in set(f.keys()) and not any(k.startswith(P) for k in f.keys())
    layers = json.loads(f.metadata()["_quantization_metadata"])["layers"]
    assert layers == {"blocks.0.attn.qkv_proj": CONF, "blocks.0.mlp.fc1": CONF}, layers
    assert f.get_slice("blocks.0.attn.qkv_proj.weight").get_shape() == [6, 4]
    assert f.get_tensor("blocks.0.norm1.weight").shape == (4,)
assert native.closed

old = FakeFile({"blocks.0.attn.qkv_proj.weight": np.zeros((6, 4), np.int8)}, {"_quantization_metadata": json.dumps({"layers": {}})})
assert not mod.needs_wrapper(old)   # the old layout is returned untouched
plain = FakeFile({"blocks.0.attn.qkv_proj.weight": np.zeros((6, 4), np.int8)})
assert not mod.needs_wrapper(plain)   # no .comfy_quant tensors: not a native int8 file, leave it to the old error message
noprefix = FakeFile({"blocks.0.attn.qkv_proj.comfy_quant": conf_tensor()})
assert mod.NativeLayoutFile(noprefix).prefix == "" and "blocks.0.attn.qkv_proj" in json.loads(mod.NativeLayoutFile(noprefix).metadata()["_quantization_metadata"])["layers"]
print("native layout ok")
