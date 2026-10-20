"""Standalone LTX video self-attention TP probe; never imported by node startup."""
import json
import statistics
import sys
import time
from pathlib import Path

# Script lives in /kaggle/working; subprocess cwd points at the ComfyUI checkout.
sys.path.insert(0, str(Path.cwd()))

import torch
from safetensors import safe_open
import comfy.options

MODEL_PATH = sys.argv[1]
# Match the production attention selection, without starting ComfyUI/aimdo.
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], '--use-ck-attention', '--fast', 'fp16_accumulation']
import comfy_kitchen.backends.cuda as ck
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.lightricks.model import (
    apply_rotary_emb_qk, freqs_cis_matrix, generate_freq_grid_pytorch, generate_freqs,
    generate_freq_grid_np,
)
import comfy.cli_args
comfy.cli_args.args.disable_comfy_compiler = True

DEVS = [torch.device('cuda:0'), torch.device('cuda:1')]


def layer_conf(f, keys, meta, name):
    if name + '.comfy_quant' in keys:
        return json.loads(bytes(f.get_tensor(name + '.comfy_quant').tolist()).decode())
    if name in meta:
        return meta[name]
    short = name[name.index('transformer_blocks.'):]
    matches = [v for k, v in meta.items() if k == short or k.endswith('.' + short)]
    if len(matches) != 1:
        raise RuntimeError('No unique quantization config: ' + name)
    return matches[0]


def split_scale(scale, weight, lo, hi):
    if scale.numel() == 1:
        return scale
    if scale.shape[0] == weight.shape[0]:
        return scale[lo:hi]
    raise RuntimeError('Unsupported quantization scale shape: ' + str(scale.shape))


def sync():
    for d in DEVS:
        torch.cuda.synchronize(d)


def event(d):
    e = torch.cuda.Event()
    e.record(torch.cuda.current_stream(d))
    return e


def linear(x, params):
    weight, scale, bias = params
    y = (torch.nn.functional.linear(x, weight) if scale is None else
         ck.int8_linear(x, weight, scale, convrot=True, convrot_groupsize=256))
    return y if bias is None else y + bias


def main():
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Two CUDA devices required')
    for d in DEVS:
        if torch.cuda.get_device_capability(d) != (7, 5):
            raise RuntimeError('This probe targets Kaggle T4 GPUs')
    with safe_open(MODEL_PATH, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        candidates = [k for k in keys if k.endswith('transformer_blocks.0.attn1.to_q.weight')]
        if len(candidates) != 1:
            raise RuntimeError('Cannot identify block-0 video self-attention')
        prefix = candidates[0][:-len('to_q.weight')]
        metadata = f.metadata() or {}
        config = json.loads(metadata.get('config', '{}')).get('transformer', {})
        meta = json.loads(metadata.get('_quantization_metadata', '{}')).get('layers', {})
        names = ['to_q', 'to_k', 'to_v', 'to_out.0']
        if prefix + 'to_gate_logits.weight' in keys:
            names.append('to_gate_logits')
        cpu = {}
        for name in names:
            n = prefix + name
            w = f.get_tensor(n + '.weight')
            scale = None
            if w.dtype == torch.int8:
                conf = layer_conf(f, keys, meta, n)
                if conf.get('format') != 'int8_tensorwise' or not conf.get('convrot'):
                    raise RuntimeError('Expected int8 convrot: ' + n)
                scale = f.get_tensor(n + '.weight_scale')
            elif name != 'to_gate_logits':
                raise RuntimeError('Expected int8 weight: ' + n)
            else:
                w = w.half()
            bias = f.get_tensor(n + '.bias').half() if n + '.bias' in keys else None
            cpu[name] = (w, scale, bias)
        qn, kn = [f.get_tensor(prefix + n + '.weight').half() for n in ('q_norm', 'k_norm')]

    dim = cpu['to_q'][0].shape[0]
    heads, head_dim, half = 32, 128, dim // 2
    if dim != heads * head_dim or half % 256:
        raise RuntimeError('Unexpected video attention dimensions')
    if any(cpu[n][0].shape != (dim, dim) for n in names[:4]):
        raise RuntimeError('Expected square video self-attention projections')
    if qn.numel() != dim or kn.numel() != dim:
        raise RuntimeError('Expected normalization across all heads')

    def move(params, d):
        return tuple(None if t is None else t.contiguous().to(d) for t in params)
    full = {n: move(p, DEVS[0]) for n, p in cpu.items()}
    shards = [{}, {}]
    norms = []
    for r, d in enumerate(DEVS):
        lo, hi = r * half, (r + 1) * half
        for name, (w, scale, bias) in cpu.items():
            if name == 'to_out.0':
                params = (w[:, lo:hi], scale, None)  # add bias once after reduction
            else:
                a, b = (r * heads // 2, (r + 1) * heads // 2) if name == 'to_gate_logits' else (lo, hi)
                params = (w[a:b], None if scale is None else split_scale(scale, w, a, b),
                          None if bias is None else bias[a:b])
            shards[r][name] = move(params, d)
        norms.append((qn[lo:hi].to(d), kn[lo:hi].to(d)))
    full_norms = qn.to(DEVS[0]), kn.to(DEVS[0])
    del cpu, qn, kn
    split_rope = config.get('rope_type', 'interleaved') == 'split'
    theta = config.get('positional_embedding_theta', 10000)
    max_pos = config.get('positional_embedding_max_pos', [20, 2048, 2048])
    freq_generator = generate_freq_grid_np if config.get('frequencies_precision') == 'float64' else generate_freq_grid_pytorch
    print('Attention:', dim, 'hidden,', heads, 'heads; all-head RMS reduction; gating:', 'to_gate_logits' in full,
          '| RoPE:', 'split' if split_rope else 'interleaved', flush=True)
    print('Backend:', optimized_attention.__name__, '| transfer-inclusive timings; synthetic normalized inputs.', flush=True)
    results = []
    torch.manual_seed(42)
    for label, height, width in [('stage1', 8, 14), ('stage2', 16, 28)]:
        tokens = 16 * height * width
        x = torch.randn(1, tokens, dim, device=DEVS[0], dtype=torch.float16)
        host_x = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        x1 = torch.empty_like(x, device=DEVS[1])
        host_stats = [torch.empty((1, tokens, 2), dtype=torch.float32, pin_memory=True) for _ in DEVS]
        recv_stats = [torch.empty((1, tokens, 2), dtype=torch.float32, device=d) for d in DEVS]
        host_y = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        recv_y = torch.empty_like(x)
        # LTX's 3D RoPE, with checkpoint config and identical full/split frequencies.
        tt, yy, xx = torch.meshgrid(torch.arange(16, device=DEVS[0]),
                                  torch.arange(height, device=DEVS[0]),
                                  torch.arange(width, device=DEVS[0]), indexing='ij')
        grid = torch.stack((tt * (8 / 24), yy * 32, xx * 32)).reshape(1, 3, tokens)
        freqs = generate_freqs(freq_generator(theta, 3, dim, DEVS[0]), grid, max_pos, False)
        pad = dim // 2 - freqs.shape[-1] if split_rope else dim % 6
        pe = freqs_cis_matrix(freqs, pad, split_rope, heads, torch.float32)
        rank_pe = [(pe[0][:, :, r * heads // 2:(r + 1) * heads // 2].contiguous().to(d), split_rope)
                   for r, d in enumerate(DEVS)]

        def attn(q, k, v, nheads, rotary, params, inp):
            q, k = apply_rotary_emb_qk(q, k, rotary)
            out = optimized_attention(q, k, v, nheads, transformer_options={})
            if 'to_gate_logits' in params:
                gates = 2 * torch.sigmoid(linear(inp, params['to_gate_logits']))
                out = (out.reshape(1, tokens, nheads, head_dim) * gates.unsqueeze(-1)).reshape(1, tokens, -1)
            return linear(out, params['to_out.0'])

        def single():
            q, k, v = [linear(x, full[n]) for n in ('to_q', 'to_k', 'to_v')]
            q = torch.nn.functional.rms_norm(q, (dim,), full_norms[0], eps=1e-5)
            k = torch.nn.functional.rms_norm(k, (dim,), full_norms[1], eps=1e-5)
            return attn(q, k, v, heads, pe, full, x)

        def parallel():
            with torch.cuda.device(DEVS[0]):
                host_x.copy_(x, non_blocking=True)
                ready = event(DEVS[0])
            with torch.cuda.device(DEVS[1]):
                torch.cuda.current_stream(DEVS[1]).wait_event(ready)
                x1.copy_(host_x, non_blocking=True)
            triples, stats, sent = [], [], []
            for r, d in enumerate(DEVS):
                with torch.cuda.device(d):
                    inp = x if r == 0 else x1
                    q, k, v = [linear(inp, shards[r][n]) for n in ('to_q', 'to_k', 'to_v')]
                    triples.append((q, k, v))
                    stats.append(torch.stack((q.float().square().sum(-1), k.float().square().sum(-1)), dim=-1))
                    host_stats[r].copy_(stats[r], non_blocking=True)
                    sent.append(event(d))
            parts = []
            for r, d in enumerate(DEVS):
                with torch.cuda.device(d):
                    torch.cuda.current_stream(d).wait_event(sent[1-r])
                    recv_stats[r].copy_(host_stats[1-r], non_blocking=True)
                    inv = torch.rsqrt((stats[r] + recv_stats[r]) / dim + 1e-5)
                    q, k, v = triples[r]
                    q = (q.float() * inv[..., 0:1]).to(q.dtype) * norms[r][0]
                    k = (k.float() * inv[..., 1:2]).to(k.dtype) * norms[r][1]
                    parts.append(attn(q, k, v, heads // 2, rank_pe[r], shards[r], x if r == 0 else x1))
            with torch.cuda.device(DEVS[1]):
                host_y.copy_(parts[1], non_blocking=True)
                done = event(DEVS[1])
            with torch.cuda.device(DEVS[0]):
                torch.cuda.current_stream(DEVS[0]).wait_event(done)
                recv_y.copy_(host_y, non_blocking=True)
                out = (parts[0].float() + recv_y.float()).to(x.dtype)
                bias = full['to_out.0'][2]
                if bias is not None:
                    out = out + bias
            sync()  # also guards pinned-buffer reuse and keeps all copy sources alive
            return out

        print('Testing', label, tokens, 'tokens...', flush=True)
        reference = single(); sync()
        reference2 = single(); sync()
        output = parallel(); sync()
        def measure(fn):
            timings = []
            for _ in range(3):
                sync(); start = time.perf_counter(); y = fn(); sync()
                timings.append(time.perf_counter() - start)
            return statistics.median(timings)
        a, b = measure(single), measure(parallel)
        diff = output.float() - reference.float()
        row = dict(stage=label, tokens=tokens, single_ms=a*1000, tp_ms=b*1000, speedup=a/b,
                   relative_l2=(diff.square().sum()/reference.float().square().sum().clamp_min(1e-20)).sqrt().item(),
                   max_abs_error=diff.abs().max().item(), finite=bool(torch.isfinite(output).all()))
        row['single_repeat_relative_l2'] = ((reference2.float()-reference.float()).square().sum()/
                                            reference.float().square().sum().clamp_min(1e-20)).sqrt().item()
        results.append(row)
        print(json.dumps(row), flush=True)
        if not row['finite']:
            raise RuntimeError('Non-finite TP attention output')
        del x, x1, host_x, host_stats, recv_stats, host_y, recv_y, reference, reference2, output, diff
    out = Path('/kaggle/working/ltx_tp_attention_probe.json')
    out.write_text(json.dumps(results, indent=2))
    print('Saved:', out, '| final video quality/full pipeline speed remain untested.', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
