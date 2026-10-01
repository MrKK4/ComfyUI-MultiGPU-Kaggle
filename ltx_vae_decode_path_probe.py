"""Compare the exact LTX workflow's one-tile VAE path with plain decode. Diagnostic only."""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError('This probe targets a T4')
    sys.argv = ['main.py', '--fp16-vae', '--fast-disk', '--reserve-vram', '0.5',
                '--fast', 'fp16_accumulation']
    import comfy.options
    comfy.options.enable_args_parsing()
    mg = Path('custom_nodes/ComfyUI-MultiGPU')
    ktf = load_module('ltx_path_turing', mg / 'kitchen_turing_fix.py')
    ktf.patch_comfy_kitchen_turing()
    ktf.patch_na3d_tiles_turing()
    import comfy.cli_args
    comfy.cli_args.args.disable_comfy_compiler = True
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    lowmem = load_module('ltx_path_lowmem', mg / 'ltx_vae_lowmem.py')
    lowmem.patch_ltx_vae_lowmem()

    model_path = next(Path('models/vae').glob('ltx-2.5-video-vae*.safetensors'))
    state, metadata = comfy.utils.load_torch_file(str(model_path), return_metadata=True)
    vae = comfy.sd.VAE(sd=state, metadata=metadata)
    del state
    latent = torch.randn(1, 128, 16, 16, 28, generator=torch.Generator().manual_seed(0))
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=vae.memory_used_decode(latent.shape, vae.vae_dtype))
    torch.cuda.synchronize()
    print('Model preloaded. Comparing the exact workflow tile settings with plain decode.', flush=True)
    print('Latent 1x128x16x16x28 -> 896x512, 121 frames; physical GPU1, fp16.', flush=True)

    def measure(label, fn):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / (1 << 30)
        print(f'{label}: {seconds:.3f}s, allocated peak {peak:.3f} GiB', flush=True)
        return output, seconds, peak

    # Match VAEDecodeTiled(tile_size=1024, overlap=64, temporal_size=4096,
    # temporal_overlap=16) after the node converts pixel settings to latent settings.
    tiled, tiled_sec, tiled_peak = measure(
        'one-tile decode_tiled',
        lambda: vae.decode_tiled(latent, tile_x=32, tile_y=32, overlap=2,
                                 tile_t=512, overlap_t=2))
    plain, plain_sec, plain_peak = measure('plain decode', lambda: vae.decode(latent))
    diff = plain.float() - tiled.float()
    mse = diff.square().mean().item()
    result = {
        'one_tile_decode_sec': tiled_sec,
        'plain_decode_sec': plain_sec,
        'plain_speedup': tiled_sec / plain_sec,
        'one_tile_peak_allocated_gib': tiled_peak,
        'plain_peak_allocated_gib': plain_peak,
        'relative_l2': (diff.square().sum() / tiled.float().square().sum().clamp_min(1e-20)).sqrt().item(),
        'max_abs_error': diff.abs().max().item(),
        'psnr_db': 10 * math.log10(1 / max(mse, 1e-20)),
        'finite': bool(torch.isfinite(plain).all() and torch.isfinite(tiled).all()),
    }
    print(json.dumps(result, indent=2), flush=True)
    output_path = Path('/kaggle/working/ltx_vae_decode_path_probe.json')
    output_path.write_text(json.dumps(result, indent=2))
    print('Saved:', output_path, '| production workflow unchanged.', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
