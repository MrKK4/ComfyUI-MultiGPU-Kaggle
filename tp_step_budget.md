# H3 TP step budget — what is measured, what is not, and what to do next

Rewritten after review. Four claims in the previous version did not hold; they are retracted below
with the reason, because the corrections matter more than the numbers they replaced.

## Retracted

1. **"Per-step weight re-staging (~7–9 GB per rank per step) is the hidden fixed cost."**
   Wrong premise. Tensor parallel does not re-stage weights: `_load_shards` puts each rank's half in
   VRAM once, and `_no_block_prefetch` disables the model's block prefetch precisely because it
   "would page all 21 GB in every step". So the residency proposal I built on top of it ("pin
   fc1/fc2, re-stage the overflow once per job") was fixing something that does not happen.
2. **"48 s of kernel time inside a 42.5 s step is a contradiction."** It is not. The profiled step
   ran ~50.3 s with the profiler attached, and kernels on the compute and copy streams overlap, so
   summed kernel time can exceed wall time. There was nothing to explain.
3. **"1.59× the work cost only 1.39× the time, therefore ~15 s of the step is fixed."** The fit
   used pixels × frames as a single linear work proxy while attention grows with the square of the
   token count, and text/reference/audio tokens do not grow with frame count at all. With two data
   points and three plausible terms (constant, linear, quadratic) the decomposition is
   underdetermined — no fixed term can be read off it, in either direction. The 141-frame numbers
   also disagree with each other (50.3 s profiled, ~52 s, 58.9 s derived from V25), so even the
   two-point slope is unstable.
4. **"fp16 residual stream with a 1/16 fold, ~5% relative rounding."** The fold and the rounding
   figure were hand-waved: the safety of the fold depends on a sampled peak, fp16 rounds the
   residual on every one of 50 writes, and an outlier above 16× that peak clips with no warning.
   It is a quality risk that needs an A/B on a real clip, not a free win. It is also not
   implemented in this branch — it was, and remains, a proposal.

## Two arithmetic errors of my own, found while re-checking

- The GEMM "floor" used 65e12 MAC/s as the T4's int8 peak. The card is 130 TOPS int8 = **65e12
  MAC/s**, so the floor for 5.49e14 MAC/step (both ranks) is **4.2 s**, not 8.4 s. The measured
  10.6 s is therefore ~40% of peak, not "79%, at the wall" — the opposite conclusion from the one
  I drew.
- The attention "floor" ignored memory entirely. Flash attention re-reads K/V once per query tile:
  traffic ≈ `2 · heads · n² · HEAD · bytes / BLOCK_M`. At 28 heads, n = 28.5k, HEAD = 128, int8
  K/V, BLOCK_M = 128 that is ~45 GB **per call**, i.e. ~290 GB/s sustained over the measured 335 ms
  — close to a T4's ~320 GB/s. If that model is right, attention is memory-bound at its own tile
  size, not "3× off the compute wall", and the only way to make it materially faster is to reduce
  K/V traffic (a bigger query tile), not to schedule the same traffic better.

Neither of those floors should be used again without the trace. The formula above is at least
testable: if attention is traffic-bound, ms/call should fall roughly linearly in 1/BLOCK_M.

## Measured facts, kept

| item | value | source |
|---|---|---|
| step, 0.5 MP × 124 f, 6 steps | 42.5 s warm (47.9 s first) | your runs |
| per-class share | attention 35%, int8 matmuls 25%, small elementwise 25%, PCIe exchange 15% | your profile |
| both GPUs | busy for the whole step | your observation |
| int8 attention kernel | 335 ms/call, 34.8 TOPS = 27% of int8 compute peak | your profile |
| fp16 SDPA instead | 111 s/step vs 42.5 | measured, closed |
| small-op fusion attempt | 44.9 vs 42.1–43.4 s/step | measured, closed |
| exchange | overlapped on side streams; 6.4 s/step | your profile |

Everything else in this file is arithmetic on top of those, and inherits their uncertainty.

## What the trace still has to settle

```
MMH3_TP_PHASE=1 MMH3_TP_PHASE_STEPS=2 python main.py ...   # -> tp_phase.txt
MMH3_TP_DIAG=1 python main.py ...                          # -> tp_diag.txt (start)
touch tp_profile.request                                   # -> tp_diag_trace.json (one step)
python tp_diag.py analyze                                  # per device, per class, PCIe GiB
```

1. **Summed kernel seconds per device ÷ step wall.** If the classes sum to ~42 s per device the
   step is work-bound with no idle; if they sum to ~25 s, a third of every step is idle and the
   question becomes why (dispatch, events, allocator).
2. **PCIe GiB per direction.** ~1–2 GB/step means the copies class in the profile is mostly
   bookkeeping and the exchange needs no further work; tens of GB/step means the arithmetic needs
   redoing from the real numbers instead of my estimates.
3. **Attention: which of the two rooflines binds.** Compare ms/call across BLOCK_M in
   `python turing_attention.py --selftest` — flatter than 1/BLOCK_M means compute-bound, roughly
   linear in 1/BLOCK_M means traffic-bound, and the two call for completely different kernels.

## Levers, ranked by how certain they are

**Certain, outside the sampler (from pipeline_levers.md):** SAM 3.1 batching (built, `sam3_fast.py`),
the Qwen3-VL ViT/LLM cache split, threaded CPU post. ~95 s per video combined, no model math touched.

**Needs the trace first:** the attention kernel (traffic vs compute question above), the fp16
residual stream (risk-gated, needs a clip A/B), and the exchange chunking.

**Closed, do not revisit:** fp16 attention (111 s/step), W4A8 (grainy), fewer steps, SAM every 4/8,
hand-fused modulation ops, weight residency (nothing to fix — see retraction 1).

## Fixed in this round

- **Row-chunked attention was mathematically wrong** (opt-in now, `MMH3_TP_ATTN_PIPELINE=1`): it fed
  `qkv[a:b]` to `_attn`, which splits its argument into q, k and v, so each band attended only to
  its own rows — wrong output at about a quarter of the attention work. Now rope is applied once to
  the whole sequence and only the *queries* are banded; `check_tp_attn.py` proves the chunked path
  equals the sequential path and that the old shape fails by 196% of the output scale.
- **Event reuse broke the chunk overlap** (default path): `sent`/`read` events were shared across
  chunks, so every receive waited on the last chunk's record. Per-chunk keys now; `check_tp_events.py`
  reproduces the exact binding (`sent#4` for all four receives) and proves the fix.
- **Cached compute stream reverted**: the exchange now reads `torch.cuda.current_stream` per use
  again, so a stream switch cannot silently drop a wait.
