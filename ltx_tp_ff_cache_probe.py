"""Standalone FF shard cache/restore feasibility measurement; no node registration."""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch
from safetensors import safe_open

DEVS = [torch.device('cuda:0'), torch.device('cuda:1')]


def sync():
    for d in DEVS:
        torch.cuda.synchronize(d)


def main():
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Two CUDA devices required')
    for d in DEVS:
        if torch.cuda.get_device_capability(d) != (7, 5):
            raise RuntimeError('This probe targets Kaggle T4 GPUs')
    start = time.perf_counter()
    with safe_open(sys.argv[1], framework='pt', device='cpu') as f:
        keys = set(f.keys())
        firsts = sorted(k for k in keys if k.endswith('.ff.net.0.proj.weight') and 'transformer_blocks.' in k)
        if len(firsts) != 48:
            raise RuntimeError('Expected 48 video feedforward blocks; found ' + str(len(firsts)))
        matrix_bytes = 0
        for key in firsts:
            p = key[:-len('net.0.proj.weight')]
            for suffix in ('net.0.proj.weight', 'net.2.weight'):
                shape = f.get_slice(p + suffix).get_shape()
                matrix_bytes += shape[0] * shape[1]  # validated int8 when materialized below
        available = psutil.virtual_memory().available
        # Leave room for one temporary block and the notebook/OS. Avoid a huge speculative allocation.
        if available < matrix_bytes + (4 << 30):
            raise RuntimeError(f'Not enough host RAM: need {matrix_bytes/(1<<30):.2f} GiB cache + 4 GiB headroom; '
                               f'available {available/(1<<30):.2f} GiB')
        print(f'48-block FF cache: about {matrix_bytes/(1<<30):.2f} GiB host RAM, '
              f'{matrix_bytes/2/(1<<30):.2f} GiB GPU weights per GPU.', flush=True)
        print('Loading only video FF tensors, not attention, text encoder or VAE...', flush=True)
        cache = [[], []]
        for i, key in enumerate(firsts):
            p = key[:-len('net.0.proj.weight')]
            for suffix, axis in [('net.0.proj', 0), ('net.2', 1)]:
                name = p + suffix
                w = f.get_tensor(name + '.weight')
                if w.dtype != torch.int8:
                    raise RuntimeError('Expected int8 weight: ' + name)
                if w.shape[axis] % 512:
                    raise RuntimeError('Shard does not align to convrot groups: ' + name)
                scale = f.get_tensor(name + '.weight_scale')
                bias = f.get_tensor(name + '.bias').half() if name + '.bias' in keys else None
                half = w.shape[axis] // 2
                for rank in (0, 1):
                    lo, hi = rank * half, (rank + 1) * half
                    part = w[lo:hi] if axis == 0 else w[:, lo:hi]
                    cache[rank].append(part.contiguous().pin_memory())
                    if axis == 0 and scale.numel() > 1:
                        if scale.shape[0] != w.shape[0]:
                            raise RuntimeError('Unsupported row scale shape: ' + name)
                        cache[rank].append(scale[lo:hi].contiguous().pin_memory())
                    else:
                        cache[rank].append(scale.contiguous().pin_memory())
                    if bias is not None:
                        if axis == 0:
                            cache[rank].append(bias[lo:hi].contiguous().pin_memory())
                        elif rank == 0:  # output bias added only once
                            cache[rank].append(bias.contiguous().pin_memory())
                del w, scale, bias, part
            if i % 12 == 0:
                print(f'Cached {i+1}/48 FF blocks', flush=True)
    cache_bytes = sum(t.numel() * t.element_size() for rank in cache for t in rank)
    min_available = psutil.virtual_memory().available
    load_sec = time.perf_counter() - start
    print(f'Pinned host cache ready: {cache_bytes/(1<<30):.3f} GiB; '
          f'cold file load/cache {load_sec:.2f}s; RAM available {min_available/(1<<30):.2f} GiB', flush=True)
    # Dedicated streams allow both devices to upload at the same time. Receive allocations
    # stay on compute streams; copies alone are queued on the transfer streams.
    copies = [torch.cuda.Stream(d) for d in DEVS]
    restore_times, release_times = [], []
    for iteration in range(3):
        sync()
        start = time.perf_counter()
        destinations = []
        alloc_events = []
        for rank, d in enumerate(DEVS):
            with torch.cuda.device(d):
                destinations.append([torch.empty(t.shape, dtype=t.dtype, device=d) for t in cache[rank]])
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(d))
                alloc_events.append(ev)
        for rank, d in enumerate(DEVS):
            with torch.cuda.device(d), torch.cuda.stream(copies[rank]):
                copies[rank].wait_event(alloc_events[rank])
                for target, source in zip(destinations[rank], cache[rank]):
                    target.copy_(source, non_blocking=True)
        sync()
        restore_times.append(time.perf_counter() - start)
        for rank in (0, 1):
            # Verify real copies of the first and last tensors; no large verification transfer.
            for idx in (0, len(cache[rank]) - 1):
                if not torch.equal(destinations[rank][idx].cpu(), cache[rank][idx]):
                    raise RuntimeError('Shard restore verification failed')
        peak = [torch.cuda.max_memory_allocated(d)/(1<<30) for d in DEVS]
        start = time.perf_counter()
        del target, source
        del destinations
        gc.collect()
        for d in DEVS:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        sync()
        release_times.append(time.perf_counter() - start)
        min_available = min(min_available, psutil.virtual_memory().available)
        print(f'Restore {iteration+1}: {restore_times[-1]:.3f}s; release {release_times[-1]:.3f}s; '
              f'GPU peaks {peak[0]:.2f}/{peak[1]:.2f} GiB', flush=True)
    # Extrapolation from user's one-layer measurements, not a pipeline speed prediction.
    estimate = 48 * (8*(0.016749127-0.012945034) + 3*(0.064944715-0.049901784))
    restore = statistics.median(restore_times)
    release = statistics.median(release_times)
    result = dict(blocks=48, pinned_host_gib=cache_bytes/(1<<30), cold_file_load_cache_sec=load_sec,
                  median_restore_sec=restore, median_release_sec=release,
                  min_available_host_gib=min_available/(1<<30),
                  projected_ff_compute_saving_sec=estimate,
                  two_restore_release_cycles_sec=2*(restore+release),
                  illustrative_remaining_saving_sec=estimate-2*(restore+release),
                  note='Empty-server feasibility test. Full pipeline placement, quality, TE/VAE pressure and speed remain untested.')
    out = Path('/kaggle/working/ltx_tp_ff_cache_probe.json')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
