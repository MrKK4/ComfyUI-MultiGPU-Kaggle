# Every area left to spend time on, and what each is worth

Window measured: **full SAM 3.1 masking → VAE decode complete**, 0.5 MP × 124 f warm.

| stage | seconds | where it is |
|---|---|---|
| SAM 3.1 face masking | 107–144 | full per-frame detection, 0.86 s/frame, ~35 syncs + ~4k launches/frame |
| Qwen3-VL 32B text + face | 59 | face embedding cacheable, prompt changes sometimes |
| DiT sampling, 6 steps | 255 | 42.5 s/step (int8 TP) |
| VAE encode + decode | 81 | dual-GPU |
| **window total** | **502** | ≈ 8.4 min |
| CPU blur / uncrop / MP4 | 40 | after decode, outside the window |
| **whole job** | **542** | ≈ 9.0 min (your 550–610 s for step 1 + step 2 includes load/setup) |

The 6-step sampler is 51% of the window. Everything below is grouped by what it costs to get
and how confident the number is.

## Tier 1 — no new data needed, buildable now (high confidence)

| area | change | now | after | gain |
|---|---|---|---|---|
| **SAM 3.1 batching** | kill the per-frame Python loop: ~35 `cudaStreamSynchronize` and ~4k launches per frame, one CPU core pinned, GPU ~40% idle. Batch frames, do box/refine math on GPU, sync once per N frames | 107 | 70–80 | **27–37 s** |
| **Qwen3-VL cache split** | move the cache boundary to the ViT/LLM junction so a prompt tweak only re-runs the LLM; the face image alone is already cached | 59 | 15–20 | **39–44 s** when the prompt repeats (less if face + prompt change together) |
| **CPU blur/uncrop/MP4 threading** | it is one core for ~40 s; the work is per-frame and independent | 40 | 20–22 | **18–20 s** |

Tier 1 alone: **542 → ~450 s** (≈ −17%), and none of it touches model math, so quality cannot move.

**Status of Tier 1:**

| item | state | how to A/B |
|---|---|---|
| SAM 3.1 batching | **built** — `sam3_fast.py`, on by default (`MMH3_SAM_FAST=0` disables); `check_sam_fast.py` proves both rewritten paths decision-identical to upstream on random masks | `sam_fast.txt` + `sam_profile.request`, against a `MMH3_SAM_FAST=0` run |
| Qwen3-VL ViT/LLM split | **built** — `h3_qwen_cache.py`, on by default (`MMH3_QWEN_CACHE=0` disables); keys on the pixels, so a prompt tweak reuses the vision tower; `check_h3_qwen_cache.py` proves cache integrity under aliasing on both hit and miss paths | `h3_qwen_cache.txt` reports hits and tower seconds skipped |
| post-decode chain | **built** — `post_gpu.py`, **off** by default (`MMH3_POST_GPU=1` or `touch post_gpu.request`); redirects `intermediate_device()` to the GPU with a VRAM floor and cached query; `check_post_gpu.py` proves the fallbacks | `post_gpu.txt` + a run without the flag |

Expected: SAM 27–37 s, Qwen ~39–44 s when the prompt repeats, post chain ~20–30 s of the 40 s CPU
time (the MP4 encode itself stays on the CPU). All three need one GPU run to confirm.

Downside risk: low everywhere. The only real risk is the SAM rewrite changing a mask edge, which
is checkable frame-by-frame against today's output.

## Tier 2 — designed or written, one Kaggle run to validate (medium confidence)

| area | change | per-step | gain (6 steps) | why the number |
|---|---|---|---|---|
| **int8 attention kernel** | our Triton int8 flash kernel in place of the kitchen `qk_int_sv_i8` — but first read the roofline table the selftest now prints | 14.9 → 9.9 (1.5×) or 7.5 (2×) | **30–44 s** *if* it wins | it runs at 27% of int8 compute peak and 42% of bandwidth peak, and a 4× tile cut buys only 10–20%: neither wall binds, so the headroom is scheduling — worth measuring, not assuming |
| **fp16 residual stream** (proposal, risk-gated) | the block is degree-1 homogeneous in the residual, so a 1/16 fold fits fp16 — but it rounds the residual 50× per step and clips silently above 16× the sampled peak, so it needs a clip A/B before it counts as a win | ~10.6 → ~5.5? | **~30 s if it survives the A/B** | the small-op floor itself is unverified arithmetic |

Tier 2 on top of Tier 1: **~450 → ~385 s** (≈ −29% from today).

Risk: the kernel may lose to the kitchen one (the selftest says so in one command — no guessing).
The residual-stream change is invasive in the TP block and must be checked visually.

## Tier 3 — needs one trace first

(The old "fixed ~15 s/step" premise for this tier was an artifact of a two-point fit that ignored
quadratic attention growth — retracted; see tp_step_budget.md, retraction 3. What remains here is the
attention-traffic question and the exchange arithmetic, both of which the trace settles.)

| area | change | per-step | gain | open question |
|---|---|---|---|---|
| **attention tile / traffic** | if attention is K/V-traffic-bound (tp_step_budget.md), the lever is a larger query tile, which cuts K/V re-reads roughly linearly — not better scheduling of the same traffic | 14.9 → ? | up to ~15 s, unproven | the BLOCK_M sweep in `turing_attention.py --selftest` says which roofline binds |
| ~~weight residency~~ | **retracted**: tensor parallel does not re-stage weights — each rank's half is loaded into VRAM once and `_no_block_prefetch` deliberately keeps it there, so there is nothing to pin. The bytes still have to be read from HBM every step, but that is a bandwidth property, not a staging bug | — | none | tp_step_budget.md, retraction 1 |
| **PCIe exchange** | 6.4 s/step overlapped; if `tp_diag.py probe` finds P2P works, exchanges go direct instead of host-staged, and the copies class shrinks | 6.4 → ~4 | **~12 s** | 306 MB/rank/phase × 2 × 50 × 2 crossings cannot fit in 6.4 s — one of the two is mis-scaled; the trace's DtoH/HtoD GiB settles it |

Tier 3 would put the window at **~330 s** if it delivers.

## Tier 4 — speculative, worth a look only after the above

| area | idea | gain | why it is speculative |
|---|---|---|---|
| VAE overlap | start decoding the first latent chunk while the last sampler steps run, or overlap CPU post with decode | 10–20 s | decode is whole-video today; chunked decode needs a new code path |
| stage concurrency | run Qwen's text-only portion during SAM, or SAM's per-frame work on one GPU while the other prepares the VAE | up to 30 s | both stages are already GPU-hungry at peak; VRAM is tight |
| SAM refinement tuning | `refine_iterations` / `detection_threshold` / `max_objects` per frame — cheaper refine, same track | 10–30 s | quality-visible; you have ruled out *detecting* less often, not refining less |

## Closed — do not spend time here

| area | why not |
|---|---|
| fp16 attention / SDPA | 111 s/step vs 42.5 — the int8 kernels are 6.6× faster |
| W4A8 weights | visibly grainy, you rejected it |
| fewer steps / SAM every 4/8 | quality; you rejected both |
| hand-fusing the modulation ops | measured slower (44.9 vs 42.1–43.4 s/step) |
| int8 matmuls | already at 79% of int8 peak; ceiling ~2 s/step |
| VAE kernel work | the kitchen fp16-accum conv is Ampere-only; the sm_75 fallback is what runs today |
| `torch.compile` / graph capture | torch 2.4.1 wheels vs Kaggle's driver, plus dynamic VRAM and custom TP streams |

## Stacked totals

| built | window (SAM → decode) | whole job | step |
|---|---|---|---|
| today | 502 s | 542 s | 42.5 s |
| + Tier 1 | ~430 s | ~450 s | 42.5 s |
| + Tier 2 | ~365–385 s | ~385–405 s | ~30 s |
| + Tier 3 | ~330 s | ~350 s | ~26 s |

In target terms: SAM 107 → ~75, step 42.5 → ~30 (Tier 2), → ~26 if the fixed term is really weight
bytes (Tier 3). At 0.7 MP × 141 f the sampler share is bigger (384 s of 874), so Tier 2 is worth
more there and Tier 1 about the same.

## The order I would build in

1. **Tier 1 now** — SAM batching first (biggest certain win, ~120 lines, no math change), then the
   Qwen split, then the CPU threading. Worth ~95 s/video with no risk to output.
2. **Run the attention selftest** (one command, kernel already written) and wire
   `MMH3_TP_ATTN=int8` if it wins. Worth 30–44 s/video.
3. **Send the trace** (`MMH3_TP_PHASE=1` + `tp_diag.py analyze`) so Tier 3 stops being a
   hypothesis; if it confirms weight bytes, residency is the biggest single remaining item.
4. **fp16 residual stream** last of the sampler work, because it is the most invasive for a ~30 s
   return.
