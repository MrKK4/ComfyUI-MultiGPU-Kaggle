"""Standalone FF shard cache/restore feasibility measurement; no node registration."""
import gc
import json
import statistics
import struct
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


def ff_inventory(path):
    # Read the small safetensors header before materializing any large tensor.
    with open(path, 'rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        if size > 64 << 20:
            raise RuntimeError('Unexpected safetensors header size')
        header = json.loads(handle.read(size))
    keys = set(header) - {'__metadata__'}
    firsts = sorted(k for k in keys if k.endswith('.ff.net.0.proj.weight') and 'transformer_blocks.' in k)
    if len(firsts) != 48:
        raise RuntimeError('Expected 48 video feedforward blocks; found ' + str(len(firsts)))
    matrix_bytes, extra_bytes, dtypes = 0, 0, {}
    for key in firsts:
        prefix = key[:-len('net.0.proj.weight')]
        for suffix in ('net.0.proj', 'net.2'):
            name = prefix + suffix
            info = header[name + '.weight']
            dtype = info['dtype']
            if dtype not in ('I8', 'F16', 'BF16', 'F32'):
                raise RuntimeError('Unsupported FF storage dtype: ' + name + ': ' + dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            matrix_bytes += info['data_offsets'][1] - info['data_offsets'][0]
            scale_key = name + '.weight_scale'
            if dtype == 'I8' and scale_key not in keys:
                raise RuntimeError('Missing int8 scale: ' + name)
            if dtype == 'I8':
                info = header[scale_key]
                extra_bytes += 2 * (info['data_offsets'][1] - info['data_offsets'][0])
            bias_key = name + '.bias'
            if bias_key in keys:
                info = header[bias_key]
                extra_bytes += info['data_offsets'][1] - info['data_offsets'][0]
    return firsts, matrix_bytes, matrix_bytes + extra_bytes, dtypes


def main():
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Two CUDA devices required')
    for d in DEVS:
        if torch.cuda.get_device_capability(d) != (7, 5):
            raise RuntimeError('This probe targets Kaggle T4 GPUs')
    start = time.perf_counter()
    firsts, matrix_bytes, budget_bytes, storage_dtypes = ff_inventory(sys.argv[1])
    with safe_open(sys.argv[1], framework='pt', device='cpu') as f:
        keys = set(f.keys())
        available = psutil.virtual_memory().available
        # Leave room for one temporary block and the notebook/OS. Avoid a huge speculative allocation.
        if available < budget_bytes + (4 << 30):
            raise RuntimeError(f'Not enough host RAM: need {budget_bytes/(1<<30):.2f} GiB cache + 4 GiB headroom; '
                               f'available {available/(1<<30):.2f} GiB')
        print(f'48-block FF cache: about {matrix_bytes/(1<<30):.2f} GiB host RAM, '
              f'{matrix_bytes/2/(1<<30):.2f} GiB GPU weights per GPU.', flush=True)
        print('Matrix storage types:', storage_dtypes, '| native dtypes preserved during transfers', flush=True)
        print('Loading only video FF tensors, not attention, text encoder or VAE...', flush=True)
        cache = [[], []]
        for i, key in enumerate(firsts):
            p = key[:-len('net.0.proj.weight')]
            for suffix, axis in [('net.0.proj', 0), ('net.2', 1)]:
                name = p + suffix
                w = f.get_tensor(name + '.weight')
                if w.dtype not in (torch.int8, torch.float16, torch.bfloat16, torch.float32):
                    raise RuntimeError('Unsupported FF weight dtype: ' + name)
                if w.shape[axis] % (512 if w.dtype == torch.int8 else 2):
                    raise RuntimeError('Shard does not align to convrot groups: ' + name)
                scale = f.get_tensor(name + '.weight_scale') if w.dtype == torch.int8 else None
                bias = f.get_tensor(name + '.bias') if name + '.bias' in keys else None
                half = w.shape[axis] // 2
                for rank in (0, 1):
                    lo, hi = rank * half, (rank + 1) * half
                    part = w[lo:hi] if axis == 0 else w[:, lo:hi]
                    cache[rank].append(part.contiguous().pin_memory())
                    if scale is not None and axis == 0 and scale.numel() > 1:
                        if scale.shape[0] != w.shape[0]:
                            raise RuntimeError('Unsupported row scale shape: ' + name)
                        cache[rank].append(scale[lo:hi].contiguous().pin_memory())
                    elif scale is not None:
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
                  matrix_storage_types=storage_dtypes,
                  median_restore_sec=restore, median_release_sec=release,
                  min_available_host_gib=min_available/(1<<30),
                  projected_ff_compute_saving_sec=estimate,
                  two_restore_release_cycles_sec=2*(restore+release),
                  illustrative_remaining_saving_sec=estimate-2*(restore+release),
                  note='Block-0 compute extrapolation may not represent mixed-precision layers. Empty-server transfer test; full pipeline placement, quality, TE/VAE pressure and speed remain untested.')
    out = Path('/kaggle/working/ltx_tp_ff_cache_probe.json')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
