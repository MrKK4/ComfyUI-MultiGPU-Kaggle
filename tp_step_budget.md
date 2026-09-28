# Step budget, what I fixed, and the two commands that decide the next fix

## What changed this turn

1. **`turing_attention.py` — two real bugs found without a GPU, both would have hit on Kaggle:**
   - a `SyntaxError` (duplicate `D` parameter in both new int8 quant kernels) — the module would
     not even import;
   - an arity bug at the launch sites (an extra `dim` positional against kernels that take `N`
     plus a `D` constexpr) — `TypeError` on the first launch of the int8 backend.
   Also: the int8 flash kernel no longer round-trips K/V through fp32 (`int8 → fp32 → int8` in the
   inner loop, pure waste when the whole point is beating the kitchen kernel's throughput), and the
   selftest defaults are now your real shape (`--n 28500 --heads 28 --dim 128`, int8 in the sweep).
2. **`check_kernels.py` — new, no torch needed.** It AST-matches every `_KERNEL[grid](...)`
   launcher call against its kernel definition and fails on arity/keyword mismatches. I verified it
   catches the exact bug above by reintroducing it. Run it after any kernel edit; it is the only
   check available here that would have caught both.
3. **`h3_tensor_parallel.py`** already carries the phase timer from the previous round
   (`MMH3_TP_PHASE=1`, `_SYNC=1`, `_STEPS`, `_SKIP` → `tp_phase.txt`), plus pooled CUDA events in
   `_ChunkedExchange` (2,600 `cudaEventCreate`/step removed). Unchanged this turn.

Nothing has run on a GPU: there is still no torch in this sandbox. Everything below marked
"arithmetic" is arithmetic.

## The step, per class (0.5 MP × 124 f, your profile)

| class | seconds | share | headroom | verdict |
|---|---|---|---|---|
| int8 attention | **14.9** | 35% | 27% of int8 peak; ~7 s realistic | **biggest single lever** |
| int8 matmuls | 10.6 | 25% | 79% of int8 peak | at the wall, ~2 s |
| small elementwise ops | **10.6** | 25% | ~3.9 s HBM floor | **second lever** |
| PCIe exchange (overlapped) | 6.4 | 15% | ~2 s by arithmetic | leave alone for now |
| sum | 42.5 | | | |

Two things worth flagging in these numbers:

- Attention runs at **34.8 TOPS = 27% of the T4's int8 peak**, while the matmuls manage 79% of the
  same peak. The attention kernel is not at the wall; it is 3× off it. It is still 6.6× faster
  than PyTorch fp16 SDPA (111 s/step vs 42.5), so the only thing that can beat it is another,
  better-scheduled **int8** kernel — a faster fp16 path does not exist here.
- Small ops cost 10.6 s against a 3.9 s data floor. The gap is the fp32 residual stream and the
  fp16↔fp32 casts around every block. The block is homogeneous of degree 1 in the residual
  (rms_norm is scale-invariant, gates are linear, adaLN acts on the block input, not the
  residual), so a fixed 1/16 fold puts the residual in fp16 safely (peak 4.6e5 → 2.9e4, 2.2×
  under fp16 max) and halves six fp32 passes plus the casts, for ~5% relative rounding.

## But first: 1.59× the work only cost 1.39× the time

| job | work (px × frames) | warm step |
|---|---|---|
| 0.5 MP × 124 f | 1.00 | 42.5 s |
| 0.7 MP × 141 f | 1.59 | 58.9 s (V25: 141.9 s first step, then (436.3 − 141.9)/5) |

Solving `step = F + P·work`: **F ≈ 15 s fixed + P ≈ 28 s proportional**. A third of the step does
not scale with tokens. Candidates: per-step weight re-staging (~7–9 GB of int8 weights per rank
per step, ≈25–40 s at a 70 W-capped T4's sustained bandwidth — the right size), or per-op CPU
dispatch (one core is pegged at ~100% in your profiles).

**Two contradictions I cannot settle from here, both resolved by the commands below:**

1. The profile says ~48 s of kernel time per GPU per step, but the step is 42.5 s. Those cannot
   both be one step — so either the profile is from a different (larger) step, or the 48 s is the
   sum over both GPUs (⇒ ~24 s per GPU, GPU ~54% busy, ~20 s of per-step idle). Your "both GPUs
   were busy" observation points at the first reading, the SAM profile's 54% busy / 100% core
   points at the second.
2. The same trace has to say whether that fixed ~15 s lives in the matmul class (weight
   streaming), in the gaps between kernels (dispatch), or in the sampler outside the blocks.

## Run these two, then I know which fix to build

```
# 1. per-phase times inside every block: CPU issue time vs true GPU time
MMH3_TP_PHASE=1 MMH3_TP_PHASE_STEPS=2 python main.py ...        # -> tp_phase.txt

# 2. one profiled step, per-device classes
MMH3_TP_DIAG=1 python main.py ...            # -> tp_diag.txt (run start)
touch tp_profile.request                     # -> tp_diag_trace.json (one step), then:
python tp_diag.py analyze                    # per-device resident vs kernel, per class, PCIe GiB
```

Decision table for what comes back:

| what `tp_phase.txt` shows | what it means | fix I build |
|---|---|---|
| CPU ms/block ≈ GPU ms/block, sum ≈ step | GPU-work-bound, no idle | better int8 attention tiles + fp16 residual stream |
| CPU ms/block ≫ GPU ms/block | dispatch-bound (one core) | fewer ops per block: fold scales into the int8 quant, one-shot exchange, cached streams/events, then graph capture |
| in-block wall ≪ step | time is outside the blocks | sampler/nodes, not the DiT |

The same report also settles the two contradictions above: `tp_diag.py analyze` prints per-device
kernel seconds and the PCIe GiB per direction — if the copies are ~1–2 GB/step, the exchange
arithmetic in the old budget was mis-scaled by ~10× and the copies class is noise.

## Job-level wins, independent of all of the above

Sampling is 255 s of an 8–10 minute video; SAM (107 s) + Qwen3-VL (59 s) + VAE (81 s) + CPU
mask/MP4 (40 s) is 287 s — more than the sampler. These do not depend on the DiT diagnosis:

- **SAM 3.1**: ~35 `cudaStreamSynchronize` and ~4k launches per frame, one core 100%, GPU ~40%
  idle. Detection must stay per frame; the fix is batching the per-frame work and syncing once per
  N frames. Worth 25–40 s/video.
- **Qwen3-VL 32B**: split the cache at the ViT/LLM boundary so a prompt tweak re-runs only the LLM;
  the face image alone is already cached. Worth 40–50 s when the prompt repeats.
- **CPU blur/uncrop/MP4**: single-threaded ~40 s; thread it. Worth 20–40 s.

Realistic landing zone for a new video: ~550 s → 400–450 s from these three alone, with the DiT
work on top.

## If the phase report says GPU-work-bound

Then the ladder is: int8 attention kernel (5.8 s of the 14.9 s is recoverable at 65% of int8 peak —
`python turing_attention.py --selftest --n 28500 --heads 28 --dim 128` measures it in one command),
then the fp16 residual stream (~5 s), then the PCIe arithmetic's ~4 s. Both levers together put the
step at ~30 s and sampling at ~180 s.

## What not to do

- Do not switch attention to fp16: 111 s/step vs 42.5.
- Do not fuse the modulation ops by hand: one measured attempt was slower (44.9 vs 42.1–43.4 s/step).
  The fp16 residual stream is the version of that idea which can pay.
- Do not reduce steps or SAM frequency: you already showed both cost quality.
