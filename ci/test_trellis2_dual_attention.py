"""CPU numerical/routing checks for the dual-GPU attention implementation.

These exercise the same head split/gather path. CUDA transfers/speed still need
the opt-in notebook's first-call check on an actual Kaggle T4 pair.
"""
import contextlib
import importlib.util
import pathlib
import copy
import sys
import types
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'trellis2_dual_attention.py'
if not SOURCE.is_file():
    SOURCE = ROOT / 'notebook/nodes/trellis2_dual_attention.py'
spec = importlib.util.spec_from_file_location('trellis2_dual_test', SOURCE)
dual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)
torch.manual_seed(17)


def attention(q, k, v, mask=None, **kwargs):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                        attn_mask=mask)
    return out.transpose(1, 2)


moves = []


def transfer(x, device):
    moves.append((tuple(x.shape), x.dtype))
    return x.clone()


cases = 0
for dtype in (torch.float32, torch.float64):
    for heads in (2, 5, 12):
        for nk in (31, 47):  # self and cross attention
            q = torch.randn(2, 31, heads, 16, dtype=dtype)
            k = torch.randn(2, nk, heads, 16, dtype=dtype)
            v = torch.randn_like(k)
            for mask_heads in (1, heads):
                mask = torch.ones((2, mask_heads, 31, nk), dtype=torch.bool)
                mask[..., -5:] = False
                kwargs = {'mask': mask, 'transformer_options': {'sample': torch.ones(1), 'nested': [torch.ones(2)]}}
                reference = attention(q, k, v, **kwargs)
                actual = dual.head_parallel(q, k, v, attention, 'cpu', lambda d: contextlib.nullcontext(), transfer, kwargs)
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
                assert actual.dtype == dtype and torch.equal(kwargs['mask'], mask)
                cases += 1

# A masked/padded row must not borrow attention from a different logical sample.
q = torch.randn(2, 13, 12, 8)
k = torch.randn(2, 21, 12, 8)
v = torch.randn_like(k)
mask = torch.zeros(2, 1, 1, 21)
mask[0, :, :, 11:] = -torch.inf
mask[1, :, :, 17:] = -torch.inf
actual = dual.head_parallel(q, k, v, attention, 'cpu', lambda d: contextlib.nullcontext(), transfer, {'mask': mask})
torch.testing.assert_close(actual, attention(q, k, v, mask=mask), rtol=1e-5, atol=1e-6)
try:
    dual.head_parallel(q[:, :, :1], k[:, :, :1], v[:, :, :1], attention, 'cpu',
                       lambda d: contextlib.nullcontext(), transfer, {})
except ValueError:
    pass
else:
    raise AssertionError('Invalid head count must be rejected')

print('Trellis2 dual attention CPU parity ok:', cases + 1, 'self/cross/masked/padded cases; dtype preserved')

# Exercise the actual ComfyUI integration seam without installing ComfyUI on the PC.
tm = types.ModuleType('comfy.ldm.trellis2.model')


class Container:
    def __init__(self, tensor):
        self.tensor = tensor

    def take(self):
        assert self.tensor is not None, 'Container consumed twice'
        tensor, self.tensor = self.tensor, None
        return tensor


def native(q, k, v, **kwargs):
    if isinstance(q, Container):
        q, k, v = q.take(), k.take(), v.take()
    return attention(q, k, v, **kwargs)


class Trellis2:
    def forward(self, *args, **kwargs):
        assert dual._active.get() == kwargs['expected']
        if kwargs.get('fail'):
            raise RuntimeError('intentional model error')
        return 'native forward'


tm.Trellis2, tm.dense_attention, tm.AttentionTensorContainer = Trellis2, native, Container
fake_modules = {name: types.ModuleType(name) for name in ('comfy', 'comfy.ldm', 'comfy.ldm.trellis2')}
fake_modules['comfy.ldm.trellis2'].model = tm
fake_modules['comfy.ldm.trellis2.model'] = tm
with patch.dict(sys.modules, fake_modules):
    dual._install()
    wrapped = tm.dense_attention
    dual._install()
    assert tm.dense_attention is wrapped
    instance = Trellis2()
    for stages in ('shape_1024', 'shape_and_texture', 'all'):
        for mode in ('structure_generation', 'shape_generation_512', 'shape_generation', 'texture_generation'):
            settings = {'enabled': True, 'stages': stages, 'min_tokens': 1}
            expected = settings if (stages == 'all' or mode == 'shape_generation' or
                                    (stages == 'shape_and_texture' and mode == 'texture_generation')) else None
            instance.forward(transformer_options={'trellis2_dual_attention': settings},
                             trellis2_generation_mode=mode, expected=expected)
            assert dual._active.get() is None
    try:
        instance.forward(transformer_options={'trellis2_dual_attention': settings},
                         trellis2_generation_mode='shape_generation', expected=settings, fail=True)
    except RuntimeError:
        pass
    assert dual._active.get() is None
    token = dual._active.set(settings)
    try:
        actual = wrapped(Container(q), Container(k), Container(v), mask=mask)
        torch.testing.assert_close(actual, native(q, k, v, mask=mask))  # CPU fallback, no double .take()
    finally:
        dual._active.reset(token)

    class Patcher:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=Trellis2())
            self.model_options = {'transformer_options': {'keep': 'original'}}

        def clone(self):
            return copy.deepcopy(self)

    model = Patcher()
    with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'device_count', return_value=2):
        modified, = dual.ModelTrellis2DualGPUAttention().apply(model)
    assert model.model_options == {'transformer_options': {'keep': 'original'}}
    assert modified.model_options['transformer_options']['trellis2_dual_attention']['stages'] == 'shape_1024'
    disabled, = dual.ModelTrellis2DualGPUAttention().apply(modified, enabled=False)
    assert disabled.model_options == model.model_options
    with patch.object(torch.cuda, 'is_available', return_value=False):
        try:
            dual.ModelTrellis2DualGPUAttention().apply(model)
        except RuntimeError as exc:
            assert 'two CUDA GPUs' in str(exc)
        else:
            raise AssertionError('Enabled node must reject a non-CUDA host')
print('Trellis2 dual integration ok: stage scopes, repeated install, exception cleanup, clone/disable, CPU fallback')
