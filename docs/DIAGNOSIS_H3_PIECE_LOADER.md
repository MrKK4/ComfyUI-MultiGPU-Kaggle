# H3 piece-loader diagnosis — 2026-11-10

Numerical failure is fixed and GPU-verified. The complete verification still fails its 1 GiB VRAM-margin requirement on the first checked split load. Selective eviction remains outside this implementation.

## Reproduction and senior review

- Failed integration: `tsdrp34/h3-pieces-loader-verify`, version 9. One seed (1101), one Euler step, 544×960, 124 frames, two Tesla T4 GPUs, PyTorch 2.13.0+cu130 and ComfyUI 0.44.2.
- Diagnostic: `tsdrp34/h3-pieces-diagnose`, version 1. Frozen failing source is retained under `portkit_runs/h3_piece_loader/diagnosis/source_verify_v9.py`.
- Fixed integration: `safd7sd4/h3-pieces-inference-verify`, version 1. It repeats the failed integration sequence with only worker inference-context inheritance changed; the CPU suite also includes the new regression.
- Claude Opus 5.5 review: benchmark `h3-pieces-gpu-diagnose`, run 6561364, verified output model `anthropic/claude-opus-5-5@default`. Prompt and answer are saved in `portkit_runs/h3_piece_loader/diagnosis/`.

## Numerical cause and proof

PyTorch inference mode is thread-local. ComfyUI's original builder runs on its inference execution thread. The new rank workers did not inherit that context, so they created ordinary tensors and tracked views instead of inference tensors.

The compiled norm receives the shard's fp32 norm tensor directly. Its changed tensor metadata triggers additional Dynamo specializations. In the reproduced cache history, the eight-entry limit is reached and rank 1 takes an eager arithmetic path. Small rounding differences are amplified by int8 activation quantization through the model.

Evidence from the diagnostic:

- All 1,200 resident GPU tensor byte digests and 200 scalars match the original builder across both ranks and all 50 blocks. Fixed-input linear probes also match exactly.
- Block-0 image, timestep embedding and rope fingerprints match. Both ranks' QKV inputs, scale, shift and modulation segments match.
- Rank 0's QKV output matches baseline. Rank 1's split-loaded output matches the eager output, while differing from baseline.
- With compilation disabled, split-loaded and source-built sampled latents match exactly.
- Original-built tensors report inference mode enabled; the failing worker-built tensors do not.

`h3_piece_loader.restore_ranks` now captures the caller's inference setting and enters that setting inside each worker. It preserves both enabled and disabled callers. The new CPU regression failed before this change and passes afterward, including retained-rank restoration. All 20 exporter/loader/memory tests pass locally and on Kaggle.

The fixed GPU integration passes first-load, warm-reuse, rank-1-restore and corruption-fallback output equality, with zero latent error. Rank 0 remains resident during rank-1 restoration. Corruption is rejected and source fallback works. Dual-GPU VAE decode produces finite output. The recompile-limit warning is absent, and block-0 error returns to the original 0.003057 value.

No artifact rebuild or HF upload is needed: the exported bytes were correct. The loader remains off by default. No compiler limits, hash checks, thresholds, production notebooks or tags were changed.

## Separate memory failure

The fixed run still peaks at 14.374 GiB on GPU0 during the first checked split run, against a 13.56 GiB acceptance ceiling. Baseline peaks at 13.231 GiB; fixed warm reuse at 11.780 GiB; fixed rank-1 restoration including dual VAE decode at 12.499 GiB. The aggregate verification verdict therefore remains **FAIL**.

The original-code diagnostic localizes extra reservation to block 0, which includes the single-GPU reference calculation (`MMH3_TP_CHECK=1`):

| GPU0 measurement | Original builder | Split loader |
| --- | ---: | ---: |
| Peak live PyTorch allocation | 12.439 GiB | 12.439 GiB |
| Reserved memory after block 0 | 12.844 GiB | 13.906 GiB |
| Driver-free memory after block 0 | 1.331 GiB | 0.188 GiB |
| Live allocation at sample end | 8.994 GiB | 8.994 GiB |

The split loader itself peaks at approximately 8.99 GiB per GPU. Extra reserved memory during the first checked block explains the observed physical VRAM difference; it is not additional resident model bytes or a persistent live-tensor leak. Allocation order, allocator reuse and asynchronous reference-work lifetimes remain candidates for why that reservation differs. A specific fragmentation or backend cause has not yet been proven.

Claude initially predicted that fixing inference mode would also remove the memory spike. The one-variable GPU test falsified that prediction. A follow-up review carried these counter-results. Do not increase the ceiling or skip the failing check to report a pass. The subsequent cold run with the normal production setting (`MMH3_TP_CHECK=0`) is reported separately below.

The follow-up Opus 5.5 review (run 6561457, saved as `memory_senior_answer.md`) withdrew its compilation explanation and recommended controlled trim/order/production probes. `safd7sd4/h3-pieces-memory-probe`, version 1, completed all three cases in fresh ComfyUI processes. Every case reproduced baseline video/audio latents exactly, with no recompile-limit warning.

| Controlled case | GPU0 NVML peak | GPU1 NVML peak | 1 GiB margin |
| --- | ---: | ---: | --- |
| Trim before block 0, CHECK=1 | 14.343 GiB | 11.890 GiB | Fail |
| Load pieces after model load, CHECK=1 | 13.265 GiB | 11.856 GiB | Pass |
| Existing preload order, normal CHECK=0 | 12.060 GiB | 11.856 GiB | Pass |

The trim released about 0.31 GiB before the check but reserved memory grew again to 13.875 GiB **inside the reference calculation**. Worker and execution stream identifiers were all zero, ruling out the proposed worker/default-stream mismatch. Moving preload after model loading reduced post-reference reservation to 12.875 GiB. This demonstrates load-order sensitivity in the checked reference path and refutes the proposed simple trim fix. It does not identify a specific allocator implementation defect.

Normal cold inference passes the margin for this workload without changing load order. The original CHECK=1 integration failure remains recorded; the production result is not substituted for it. The order change exists only in the diagnostic overlay and has not been adopted into the runtime. Any proposed load-order integration change should return to Claude with these measurements before implementation.

Complete results are in `portkit_runs/h3_piece_loader/diagnosis/memory_output_v1/h3_up/memory_summary.json`. The background monitor saved the reports and stopped with `results_ready`.

## Measurements and scope

Fixed full restore: 167.796 s including 90.301 s for the mandatory first source SHA-256. Rank-1 piece restore: 43.003 s; original rank-1 rebuild in the same run: 42.94 s. Whole-rank loading has not demonstrated a speed advantage. These timings do not establish a cold filesystem cache or predict selective-piece performance.

Reports, raw lifecycle measurements, GPU digests, review answers and the before/after regression record are under `portkit_runs/h3_piece_loader/diagnosis/`. The launcher candidates were refreshed with the numerical fix, while the failing repro remains frozen.

One seed/step does not establish long-video, LoRA, or heavy-preprocessor acceptance. Partial eviction and automatic preprocessor/VAE demand integration have not been added.
