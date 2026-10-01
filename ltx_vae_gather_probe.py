"""Standalone T4 VAE profile and fused NA packing experiment. No node registration."""
import collections
import importlib.util
import inspect
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_windows(Q, K, V, OQ, OK, OV, QS, KS,
                  SB: tl.constexpr, ST: tl.constexpr, SH: tl.constexpr,
                  SW: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
                  B: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr,
                  QT: tl.constexpr, QH: tl.constexpr, QW: tl.constexpr,
                  KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                  TOTAL: tl.constexpr, QUERY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    nt: tl.constexpr = QT * QH * QW if QUERY else KT * KH * KW
    channel = i % HD
    token = (i // HD) % nt
    head = (i // (HD * nt)) % NH
    batch = (i // (HD * nt * NH)) % B
    group = i // (HD * nt * NH * B)
    if QUERY:
        t0 = tl.load(QS + group * 3, valid, 0)
        h0 = tl.load(QS + group * 3 + 1, valid, 0)
        w0 = tl.load(QS + group * 3 + 2, valid, 0)
        tt, hh, ww = token // (QH * QW), (token // QW) % QH, token % QW
    else:
        t0 = tl.load(KS + group * 3, valid, 0)
        h0 = tl.load(KS + group * 3 + 1, valid, 0)
        w0 = tl.load(KS + group * 3 + 2, valid, 0)
        tt, hh, ww = token // (KH * KW), (token // KW) % KH, token % KW
    source = batch * SB + (t0 + tt) * ST + (h0 + hh) * SH + (w0 + ww) * SW + head * SN + channel * SD
    if QUERY:
        tl.store(OQ + i, tl.load(Q + source, valid, 0), valid)
    else:
        tl.store(OK + i, tl.load(K + source, valid, 0), valid)
        tl.store(OV + i, tl.load(V + source, valid, 0), valid)


def pack_group(q, k, v, tiles, nq, nk):
    if q.stride() != k.stride() or q.stride() != v.stride():
        raise RuntimeError('Packing probe requires matching Q/K/V strides')
    qs, ks = tiles[0]
    qt, qh, qw = [s.stop - s.start for s in qs]
    kt, kh, kw = [s.stop - s.start for s in ks]
    if qt*qh*qw != nq or kt*kh*kw != nk:
        raise RuntimeError('Unexpected geometry-group dimensions')
    g, b, nh, hd = len(tiles), q.shape[0], q.shape[-2], q.shape[-1]
    qo = torch.empty((g*b, nh, nq, hd), dtype=q.dtype, device=q.device)
    ko = torch.empty((g*b, nh, nk, hd), dtype=k.dtype, device=k.device)
    vo = torch.empty_like(ko)
    qstarts = torch.tensor([[s.start for s in a] for a, _ in tiles], device=q.device, dtype=torch.int32)
    kstarts = torch.tensor([[s.start for s in a] for _, a in tiles], device=q.device, dtype=torch.int32)
    strides = dict(zip(('SB', 'ST', 'SH', 'SW', 'SN', 'SD'), q.stride()))
    common = dict(**strides, B=b, NH=nh, HD=hd, QT=qt, QH=qh, QW=qw, KT=kt, KH=kh, KW=kw, BLOCK=256)
    for query, total in ((True, qo.numel()), (False, ko.numel())):
        _pack_windows[(triton.cdiv(total, 256),)](q, k, v, qo, ko, vo, qstarts, kstarts,
                                              TOTAL=total, QUERY=query, **common)
    return qo, ko, vo


def make_candidate(ena):
    # Keep official window boundaries, masks, batching and SDPA. Replace only packing.
    src = inspect.getsource(ena.na3d)
    pattern = r'            q_s = torch\.stack.*?            o = functional\.scaled_dot_product_attention'
    src, count = re.subn(pattern,
                        '            q_s, k_s, v_s = _ltx_pack_group(q, k, v, chunk, nq, nk)\n'
                        '            o = functional.scaled_dot_product_attention', src, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError('Installed eager na3d differs from the expected 0.2.35 implementation')
    ns = dict(vars(ena))
    ns['_ltx_pack_group'] = pack_group
    exec(compile(src, '<ltx_fused_na_packing_probe>', 'exec'), ns)
    return ns['na3d']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(candidate, reference):
    diff = candidate.float() - reference.float()
    return dict(relative_l2=(diff.square().sum()/reference.float().square().sum().clamp_min(1e-20)).sqrt().item(),
                max_abs_error=diff.abs().max().item(), finite=bool(torch.isfinite(candidate).all()))


def main():
    # Caller exposes physical GPU1 as logical cuda:0, matching VAE placement without P2P.
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError('This probe targets a T4')
    sys.argv = ['main.py', '--fp16-vae', '--fast-disk', '--reserve-vram', '0.5', '--fast', 'fp16_accumulation']
    import comfy.options
    comfy.options.enable_args_parsing()
    mg = Path('custom_nodes/ComfyUI-MultiGPU')
    ktf = load_module('ltx_probe_turing', mg / 'kitchen_turing_fix.py')
    ktf.patch_comfy_kitchen_turing()
    ktf.patch_na3d_tiles_turing()
    import comfy.cli_args
    comfy.cli_args.args.disable_comfy_compiler = True
    import comfy_kitchen
    import comfy_kitchen.backends.eager.na as ena
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    lowmem = load_module('ltx_probe_lowmem', mg / 'ltx_vae_lowmem.py')
    lowmem.patch_ltx_vae_lowmem()
    fused = make_candidate(ena)
    torch.manual_seed(0)
    rows = []
    print('VAE probe: physical GPU1, fp16. Existing 2**22 eager budget retained.', flush=True)
    print('Experiment: fuse Q/K/V window gathering; preserve neighborhoods, masks and SDPA.', flush=True)
    for dims in ((16, 64, 112), (32, 128, 224)):
        print('Kernel micro-test:', dims, 'k11x11x11', flush=True)
        q, k, v = [torch.randn(1, *dims, 4, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
        def bench(fn):
            out = fn(q, k, v, [11]*3, None, 0.125)
            torch.cuda.synchronize()
            times = []
            for _ in range(2):
                start = time.perf_counter()
                out = fn(q, k, v, [11]*3, None, 0.125)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            return out, statistics.median(times)
        ref, a = bench(ena.na3d)
        out, b = bench(fused)
        row = dict(shape=dims, baseline_ms=a*1000, fused_ms=b*1000, speedup=a/b, **compare(out, ref))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del q, k, v, ref, out
        torch.cuda.empty_cache()
    promising = all(r['finite'] and r['relative_l2'] < 0.002 for r in rows) and rows[-1]['speedup'] >= 1.10
    print('Full candidate decode gate:', 'PASS' if promising else 'SKIP — no sufficient kernel speed/accuracy gain', flush=True)
    # One baseline decode profile, and only one candidate decode when the micro-test passes.
    path = next(Path('models/vae').glob('ltx-2.5-video-vae*.safetensors'))
    sd, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    del sd
    lat = torch.randn(1, 128, 16, 16, 28, generator=torch.Generator().manual_seed(0))
    comfy.model_management.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(lat.shape, vae.vae_dtype))
    torch.cuda.synchronize()
    print('VAE preloaded; decode profile at 896x512, 121 frames. Model load excluded.', flush=True)
    timings = collections.defaultdict(list)
    originals = {}
    def timed(name, fn):
        def wrapper(*args, **kwargs):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            out = fn(*args, **kwargs)
            b.record()
            timings[name].append((a, b))
            return out
        return wrapper
    for cls, name in [('NABlock', 'deterministic blocks'), ('DiffusionNABlock', 'diffusion blocks'),
                      ('NeighborhoodAttention3D', 'attention total'), ('SwiGLU', 'SwiGLU')]:
        original = getattr(nd, cls).forward
        originals[cls] = original
        getattr(nd, cls).forward = timed(name, original)
    orig_na = comfy_kitchen.na3d
    comfy_kitchen.na3d = timed('na3d', orig_na)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    reference = vae.decode(lat)
    torch.cuda.synchronize()
    baseline_sec = time.perf_counter()-start
    print(f'Baseline decode {baseline_sec:.2f}s, peak {torch.cuda.max_memory_allocated()/(1<<30):.2f} GiB', flush=True)
    profile = {name: sum(a.elapsed_time(b) for a,b in events)/1000 for name,events in timings.items()}
    for name, seconds in sorted(profile.items(), key=lambda kv: -kv[1]):
        print(f'  {name}: {seconds:.2f}s ({len(timings[name])} calls)', flush=True)
    print('Nested profile rows overlap; do not sum them.', flush=True)
    for cls, original in originals.items():
        getattr(nd, cls).forward = original
    comfy_kitchen.na3d = orig_na
    result = dict(kernel_micro_tests=rows, candidate_decode_gate=promising, baseline_decode_sec=baseline_sec,
                  baseline_profile_sec=profile)
    if promising:
        comfy_kitchen.na3d = fused  # subprocess only; no production patch
        start = time.perf_counter()
        candidate = vae.decode(lat)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        mse = (candidate.float()-reference.float()).square().mean().item()
        result.update(candidate_decode_sec=elapsed, decode_speedup=baseline_sec/elapsed,
                      video_psnr_db=10*math.log10(1/max(mse,1e-20)),
                      video_max_difference=(candidate.float()-reference.float()).abs().max().item(),
                      candidate_finite=bool(torch.isfinite(candidate).all()))
        print(json.dumps({k:v for k,v in result.items() if k not in ('kernel_micro_tests','baseline_profile_sec')}, indent=2), flush=True)
    path = Path('/kaggle/working/ltx_vae_gather_probe.json')
    path.write_text(json.dumps(result, indent=2))
    print('Saved:', path, '| production behavior unchanged; full job validation still required.', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
