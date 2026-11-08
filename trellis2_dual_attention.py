"""Opt-in exact head-parallel attention for native TRELLIS.2 and Pixal3D.

Preserves the model's compute dtype and weights. Only attention is split; linear
projections and MLPs stay on the primary GPU. Explicit pinned-host copies support
Kaggle T4 pairs without peer access. CUDA speed/quality validation is still needed.
"""
import contextvars
import functools
import logging

import torch

logger = logging.getLogger('MultiGPU')
_active = contextvars.ContextVar('trellis2_dual_attention', default=None)
_checked = set()
_logged = set()


def _move_tree(value, device, transfer):
    if isinstance(value, torch.Tensor):
        return transfer(value, device)
    if isinstance(value, dict):
        return {k: _move_tree(v, device, transfer) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tree(v, device, transfer) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(v, device, transfer) for v in value)
    return value


def _head_kwargs(kwargs, lo, hi, heads):
    out = dict(kwargs)
    mask = out.get('mask')
    if isinstance(mask, torch.Tensor) and mask.ndim >= 3 and mask.shape[-3] == heads:
        out['mask'] = mask[..., lo:hi, :, :]
    return out


def head_parallel(q, k, v, attention, secondary, context, transfer, kwargs):
    """Split [batch, tokens, heads, channels] attention without changing its math.

    context/transfer are injectable so CPU tests exercise this same execution path.
    """
    heads = q.shape[2]
    middle = heads // 2
    if middle == 0 or k.shape[2] != heads or v.shape[2] != heads:
        raise ValueError('Head-parallel attention requires matching Q/K/V head counts >= 2')
    remote = [transfer(t[:, :, middle:].contiguous(), secondary) for t in (q, k, v)]
    remote_kwargs = _move_tree(_head_kwargs(kwargs, middle, heads, heads), secondary, transfer)
    # CUDA launches are asynchronous: enqueue primary work before secondary work.
    local = attention(q[:, :, :middle], k[:, :, :middle], v[:, :, :middle],
                      **_head_kwargs(kwargs, 0, middle, heads))
    with context(secondary):
        other = attention(*remote, **remote_kwargs)
    other = transfer(other, q.device)
    return torch.cat((local, other), dim=2)


def _host_transfer(tensor, device):
    device = torch.device(device)
    if tensor.device == device:
        return tensor
    if device.type != 'cuda':
        return tensor.to(device)
    # Do not rely on cudaMemcpyPeer: Kaggle T4s can have no peer access.
    if tensor.device.type == 'cpu':
        host = tensor if tensor.is_pinned() else tensor.pin_memory()
    else:
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
        with torch.cuda.device(tensor.device):
            host.copy_(tensor, non_blocking=True)
            torch.cuda.current_stream(tensor.device).synchronize()
    with torch.cuda.device(device):
        out = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
        out.copy_(host, non_blocking=True)
        torch.cuda.current_stream(device).synchronize()
    return out


def _log_once(key, message, *args):
    if key not in _logged:
        _logged.add(key)
        logger.info(message, *args)


def _install():
    from comfy.ldm.trellis2 import model as tm
    if getattr(tm.dense_attention, '_trellis2_dual_attention', False):
        return
    original_attention = tm.dense_attention
    original_forward = tm.Trellis2.forward

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        options = kwargs.get('transformer_options') or {}
        settings = options.get('trellis2_dual_attention')
        mode = kwargs.get('trellis2_generation_mode', 'structure_generation')
        enabled = settings and settings.get('enabled') and (
            settings['stages'] == 'all' or mode == 'shape_generation' or
            (settings['stages'] == 'shape_and_texture' and mode == 'texture_generation'))
        token = _active.set(settings if enabled else None)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _active.reset(token)

    @functools.wraps(original_attention)
    def attention(q, k, v, **kwargs):
        settings = _active.get()
        if not settings:
            return original_attention(q, k, v, **kwargs)
        if isinstance(q, tm.AttentionTensorContainer):
            q, k, v = q.take(), k.take(), v.take()
        if q.device.type != 'cuda' or q.ndim != 4 or q.shape[2] < 2 or q.shape[1] < settings['min_tokens']:
            return original_attention(q, k, v, **kwargs)
        options = kwargs.get('transformer_options') or {}
        if options.get('patches') or options.get('patches_replace'):
            _log_once('patches', '[Trellis2 dual attention] custom attention patches: using original attention')
            return original_attention(q, k, v, **kwargs)
        secondary = torch.device(settings['secondary'])
        if secondary == q.device:
            return original_attention(q, k, v, **kwargs)
        half_bytes = sum(t[:, :, q.shape[2] // 2:].numel() * t.element_size() for t in (q, k, v))
        free, _ = torch.cuda.mem_get_info(secondary)
        if free < half_bytes * 2 + 2 * 1024 ** 3:
            _log_once('memory', '[Trellis2 dual attention] secondary VRAM budget insufficient: using original attention')
            return original_attention(q, k, v, **kwargs)
        _log_once(('active', str(q.device), str(secondary), str(q.dtype)),
                  '[Trellis2 dual attention] active on %s + %s, dtype=%s; attention only, no precision change',
                  q.device, secondary, q.dtype)
        out = head_parallel(q, k, v, original_attention, secondary, torch.cuda.device, _host_transfer, kwargs)
        key = (str(q.device), str(secondary), str(q.dtype), q.shape[2], q.shape[-1], q.shape[1] == k.shape[1])
        if settings['verify'] and key not in _checked:
            # Check a bounded query sample against full-head native attention. Keep all keys.
            rows = min(128, q.shape[1])
            check_kwargs = dict(kwargs)
            mask = check_kwargs.get('mask')
            if isinstance(mask, torch.Tensor) and mask.ndim >= 2 and mask.shape[-2] == q.shape[1]:
                check_kwargs['mask'] = mask[..., :rows, :]
            reference = original_attention(q[:, :rows], k, v, **check_kwargs)
            low_precision = q.dtype in (torch.float16, torch.bfloat16)
            torch.testing.assert_close(out[:, :rows], reference,
                                       rtol=0.01 if low_precision else 0.0005,
                                       atol=0.002 if low_precision else 0.00005)
            _checked.add(key)
            logger.info('[Trellis2 dual attention] numerical check passed (%d query rows, all keys)', rows)
        return out

    attention._trellis2_dual_attention = True
    tm.dense_attention = attention
    tm.Trellis2.forward = forward


class ModelTrellis2DualGPUAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'enabled': ('BOOLEAN', {'default': True}),
            'secondary': (['cuda:1', 'cuda:0'],),
            'stages': (['shape_1024', 'shape_and_texture', 'all'],),
            'min_tokens': ('INT', {'default': 8192, 'min': 1, 'max': 1048576}),
            'verify': ('BOOLEAN', {'default': True}),
        }}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'advanced/multiGPU/3D'
    DESCRIPTION = ('Experimental attention-head parallelism for native TRELLIS.2/Pixal3D. '
                   'Preserves compute precision. Linear projections/MLPs stay on the primary GPU. '
                   'Needs two CUDA GPUs; speed and final mesh quality need Kaggle validation.')

    def apply(self, model, enabled=True, secondary='cuda:1', stages='shape_1024', min_tokens=8192, verify=True):
        from comfy.ldm.trellis2 import model as tm
        if not isinstance(model.model.diffusion_model, tm.Trellis2):
            raise ValueError('This node supports native TRELLIS.2 and Pixal3D models only')
        out = model.clone()
        options = dict(out.model_options.get('transformer_options', {}))
        if enabled:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError('Dual attention requires two CUDA GPUs')
            if stages not in ('shape_1024', 'shape_and_texture', 'all'):
                raise ValueError('Unknown dual-attention stage selection')
            _install()
            options['trellis2_dual_attention'] = {
                'enabled': True, 'secondary': secondary, 'stages': stages,
                'min_tokens': int(min_tokens), 'verify': bool(verify),
            }
        else:
            options.pop('trellis2_dual_attention', None)
        out.model_options['transformer_options'] = options
        return (out,)


NODE_CLASS_MAPPINGS = {'ModelTrellis2DualGPUAttention': ModelTrellis2DualGPUAttention}
NODE_DISPLAY_NAME_MAPPINGS = {'ModelTrellis2DualGPUAttention': 'TRELLIS.2 / Pixal3D Dual GPU Attention (Experimental)'}
