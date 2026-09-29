# Answers to the review brief, and what changed in this branch

This responds to the five ranked items, the memory plan and the FirstBlockCache probe. Every claim
below was checked against ComfyUI v0.37.0 as vendored here (`comfy/ldm/sam3/tracker.py`,
`comfy/ldm/minimax/vae.py`) or against the branch, and says so. Where the brief is right I say so;
where it is wrong I show the line.

Branch state: `arena/01a0e828-comfyui-multigpu-kaggle`, on top of your `mmh3-kaggle` fix (freeing
cuda:0 before SAM, the weight-keyed Ref2VA cache) and the nine earlier commits.

---

## 1. Conditioning cache — implemented and hardened (your rank 1)

`h3_ref_cache.py` now keys on **every reference image hashed in full**, not `img[:1]` (two faces shot
against the same background shared their first row, so the old key could hand a job the wrong face);
on the encoder's **weight layout plus a 4 kB content sample of four spread-out tensors** (your own
code comment asked for a weight hash "if that ever matters" — this is ~ms, not 14 GB, and it makes
two fine-tunes with identical names/shapes/dtypes distinct); on the **patch keys another node put on
the patcher**; and on the VAE the same way. Four entries LRU, so alternating faces do not thrash.

`h3_ref_cache.txt` now reports hits, misses, the measured seconds spent on each, and the observed
saving per hit — the acceptance test you asked for is a two-job run with the same face and prompt and
a different driving video; a hit that shows ~3 s against a ~60 s miss is the 50–60 s. And
`MMH3_REF_CACHE_VERIFY=1` re-encodes on a hit and compares the conditioning exactly, dropping the
entry and using the fresh result if it differs — the only test that proves a hit returns what a miss
would have. 15 offline checks in `check_h3_ref_cache.py`, including both collision cases and the
stale-entry path.

## 2. SAM — off by default, with a golden A/B (your rank 2, and the flicker)

Your read of the code is right on all three counts, verified here: `_get_connected_components` loops
per object with a `.cpu().numpy()` per object (tracker.py:97); `track_video_with_detection` appends
`pack_masks(masks_out).to(idev)` every frame (tracker.py:1776) with a second per-frame transfer at
line 1119; `_match_and_add_detections` syncs through `.nonzero()...tolist()`, `argmax().item()` and a
per-detection tensor comparison (tracker.py:1578). The prefetch flag and the 16-object multiplex are
there too. Your "next-frame backbone prefetch" claim checks out.

**On the flicker, plainly: I could not reproduce it and I could not explain it from the code.** Re-
reading upstream against `sam3_fast.py` found no deviation in the mask arithmetic — the hole-filling
passes, the component areas and the greedy NMS decisions reproduce upstream exactly — and one in
dtype (`_nms_masks` promoted the overlap matrix to float32 before the threshold compare; exact for
the default 0.5, wrong in general, now removed). Two things could be responsible and only a
measurement separates them: this module changed the *device* of the tensors the node returns, and a
downstream node that branches on `tensor.device` can take a different path; or the flicker is
ComfyUI's own SAM 3.1 multiplex core. So:

* **the module now defaults to off** (`MMH3_SAM_FAST=1` to enable),
* **`MMH3_SAM_GOLDEN=1` runs upstream's implementation beside the fast one on every call**, compares
  results exactly, logs the first differing element, and returns *upstream's* output on any
  mismatch — a golden run cannot introduce a difference,
* `sam_fast.txt` ends with a verdict line that says "value-identical on this run" only if every
  compared call matched,
* the mask accumulator now decides its device **once per run** from free VRAM
  (`MMH3_SAM_VRAM_FLOOR_MB`, default 2048) instead of redirecting blindly — the second face-swap job
  OOMed at 13.1 GiB inside tracking, and holding the whole clip's masks in VRAM is the one thing this
  module does that can make that worse.

`check_sam_fast.py` grew 14 checks, including that the overlap matrix is never promoted, that
decisions still match upstream where fp16 saturates to inf/nan (48 such entries across 12 trials),
and that exact math really would have decided differently on those masks — the regime is
decision-relevant, not cosmetic. I did **not** rewrite `_match_and_add_detections`; with the fast
path off by default and unproven, adding a second behavioural change to the same subsystem is how
the first one got here.

**What I need from you:** the same clip once with `MMH3_SAM_FAST=0` (default) and once with
`MMH3_SAM_GOLDEN=1`. Flicker in both ⇒ it is upstream's tracking, not this file. Flicker only in the
golden run with a mismatch logged ⇒ we have the exact frame and the exact function.

## 3. VAE spatial tiles — half of this was already in core (your rank 3)

ComfyUI v0.37.0 **already batches** spatial decoder tiles (`_decode_tile_row`,
comfy/ldm/minimax/vae.py): `batch = int(max(1, min(4, free // (128 * 2**20 * z_row.shape[0]))))`. So
"batch the same tiles" is done — by the formula, on free VRAM. On a T4 holding the TP shards and a
second VAE copy that formula answers **1**, which is why it never showed up as a win. What was
missing is the reservation, and that is what I added (`h3_vae_tiles.py`):

* `MMH3_VAE_TILE_BATCH` (default 2; `0` = ComfyUI's own auto, `auto` = at least 2),
* `reserve_extra_tiles()`, called from `h3_dual_vae.py` for the main VAE **and** the helper on its own
  device before each decode, so the planner makes room instead of the formula collapsing — the same
  lever Furkan's optimizer pulls through `memory_used_decode`,
* an OOM fallback to single tiles that keeps going and counts the event, so forcing 2 on a full card
  is safe,
* `h3_vae_tiles.txt` prints ComfyUI's auto value next to the one actually used.

Tiles, order, blend and canvas stay ComfyUI's; only the call size changes. Your "T4 unverified" caveat
stands — the same technique was measured bit-identical on a 5090 and 2.4e-4 on a 3090 by the author.
17 checks in `check_h3_vae_tiles.py`, including the case the module exists for (ComfyUI's formula
says 1, this says 2, all tiles still identical and in order).

The **encode** path has no batching in core at all (`tiled_encode` loops tiles one at a time) — that
is 42 s in your budget, and batching it is the same trick. I have not built it because it would be an
unverified change to a path nothing has measured; say the word and it is about the same size as this
one.

## 4. Postprocessing — needs one thing from you (your rank 4)

`post_gpu.py` already exists (default off, `MMH3_POST_GPU=1` or a `post_gpu.request` file) and is
exactly the global redirect that failed at VHS. Your proposal — device-local work *inside explicit
nodes* with a CPU handoff at the VHS boundary — is the right shape, but it needs the node class
names of your chain to be anything other than a guess. From the notebook lines you cite I can infer
the roles (dual-GPU decoded crop, `SubjectUncrop`, two VHS saves, `GrowMaskWithBlur` upstream of
`SubjectCrop`) but not the registered class names, and a redirect keyed on the wrong names is a no-op
that looks like a win. Send me the three or four names (`GrowMaskWithBlur`, the Subject* nodes, the
save nodes) or paste the node list from the notebook and I will scope the redirect to them and
demote each node's outputs to CPU at its boundary.

Also noted, and I agree: `GrowMaskWithBlur` is upstream of `SubjectCrop` in **Step 1**, so nothing
there should be credited against Step 2's ~40 s.

## 5. Turing Utils W8A8 in `_attn` — benchmark first (your rank 5)

Nothing to change in the plan: the microbenchmark has to come first. `turing_attention.py --selftest
--n 28500 --heads 28 --dim 128` already writes the real-shape table for our kernel
(`turing_attention.txt`: TOPS, % of the 130 TOPS peak, GB/s, % of the 320 GB/s wall, both floors).
The A/B is: install the pack, run its bundled sm75+ W8A8 kernel at the same shape, write both tables
side by side, and only then wire it behind `MMH3_TP_ATTN`. Two findings from the research that shape
what to expect: our kernel is at 27% of the int8 peak while a hand-written **fp16** kernel has been
measured at 66% of the fp16 peak on the same chip, and SageAttention's own int8 kernel reaches 52% of
int8 peak on a 4090 — so a 1.5–2× on the attention share is a real target, worth ~25–37 s by
Amdahl on the 213 s of sampling, not 0–35 s of sampler time.

---

## Corrections to the brief

| claim | status |
|---|---|
| "Furkan's vendored Sol CUDA/Triton path: documented hardware paths start above sm75 … not the T4 route" | **Right for his bundled backend** (SM80+). But core's *native* comfy-kitchen Sol path is documented as shipping on **SM75+**, and the Turing-specific `comfyui-turing-utils` pack claims native sm75 Sol + W8A8 with H3 prefix protection. Those are the T4 routes; both are unmeasured on a T4. |
| "the native v0.37.0 port already includes multiplex tracking and next-frame backbone prefetch" | **Verified** (tracker.py prefetch at 1668/1673/1682/1734/1784; 16-object multiplex state). |
| "per-object CPU connected-components processing … a per-frame packed-mask transfer … GPU scalar decisions in association" | **Verified**, with line numbers above. |
| "the attached notebook still pins `kaggle-ok-13`" | **Cannot verify here.** There is no notebook in the repo, and neither `kaggle-ok-13` nor `kaggle-ok-14` exists on the remote — only `main`, `mmh3-kaggle` and this arena branch. If the notebook pins a *branch*, point it at this one; the fixes are here, not there. |
| "one FP32 residual is ~613 MB" | **Correct** (28500 × 5376 × 4 B = 612.9 MB). |
| "~20 GB at 280 MB/s ≈ 71 s of disk transfer alone" | **Correct** (72.7 s), and it is the reason not to delete shards to make room for a cache. |
| "35.5 × 49/50 ≈ 34.8 s ceiling for one skipped tail" | **Consistent with our step classes** (42.5 s step = 14.9 attention + 10.6 GEMM + 10.6 small + 6.4 exchange); the ceiling is real but the hit rate at 6 steps is not established. |
| "your ~35 TOPS is ~27% of the int8 peak, but that does not establish 3.7× available" | **Agreed, and I have since measured the shape of the gap**: at BLOCK_M=128 the attention moves ~45.6 GB per call = 42% of the bandwidth wall (not ~90% as I wrongly claimed earlier), and 27% of the compute peak. Doubling the tile halved the traffic for 10–12%, so it is **not** traffic-bound — scheduling is the gap, and the ceiling is ~2.4× on attention only. |
| "the last four generation subcomponents are not a fresh, independently measured decomposition" | **Correct, and it applies to my own numbers too** — they are arithmetic remainders of the same three measurements, not new timing. |

## FirstBlockCache at six steps — the probe, not the feature

I have not implemented it. Your screening design is the right one and I would build it as: gather a
fixed stratified subset of rows (face-region video, surrounding video, reference/text, audio) before
and after block 0 in `H3TensorParallel.block`, accumulate `mean|r_t − r_{t−1}| / max(mean|r_{t−1}|, ε)`
on GPU, and transfer the six scalars after sampling. Two constraints worth stating: the cache has to
compose with our `patches_replace` double-block tensor parallel (it is another block-level
replacement, and installing one must not drop the other), and the memory is not the sampled rows —
one full 28500 × 5376 fp32 residual is 613 MB, so the sampled estimate has to be the first step.
duckyshell's numbers for the same family (5090, 20 steps, 90.64 → 60.82 s) are a *long-schedule*
result; nothing establishes a hit at 6 turbo steps with 10Eros Ref2VA.

## What to run on the next GPU session, in this order

1. **Two jobs, same face and prompt, different driving video** — `h3_ref_cache.txt`: one miss, one
   hit, and the saving. Add `MMH3_REF_CACHE_VERIFY=1` to one run to prove the hit equals a fresh
   encode. This is the only 50–60 s in the list that is already proven to exist as work.
2. **One clip with `MMH3_SAM_FAST=0`** (the default) and one with `MMH3_SAM_GOLDEN=1` — this settles
   the flicker, and `sam_fast.txt` says whether the fast path is value-identical if it does not.
3. **One decode, twice** — `MMH3_VAE_TILE_BATCH=0` then `=2`, compare `h3_vae_tiles.txt` and the
   decoded frames. No wiggle room: the tiles are ComfyUI's either way.
4. **The attention microbenchmark** (item 5) before any sampler work.

Everything in 1–3 is a one-line env change and each has its own report file, so a single session can
run all of them.
