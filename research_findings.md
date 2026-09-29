# Research findings: 8 questions, with sources

Every row says who measured it. `[estimate]` marks my own arithmetic on top of someone else's
number; `[my estimate]` marks my judgement about your pipeline. Where nothing exists, it says so.

Your pipeline for reference: 124 f / 0.5 MP crop / ~28.5k tokens / 6 steps / 42.5 s per step,
SAM 107 s, Qwen encode 59 s, VAE 81 s, CPU post ~40 s.

---

## Q1. SAM 3.1 video tracking that keeps the T4 busy

**What Meta's release actually claims** (RELEASE_SAM3p1.md, 2026-03-27 — measured by Meta on H100):

| claim | measured by |
|---|---|
| Object Multiplex: objects grouped into 16-object buckets, `O(N) → O(⌈N/16⌉)` passes | Meta, [RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) + [arXiv 2511.16719 §H](https://arxiv.org/html/2511.16719v2) |
| ~7× at 128 objects, single H100; 16 → 32 FPS at medium density | same |
| "Reduced CPU-GPU synchronization in detection-tracker association"; batched postprocessing and vision encoder; better torch.compile fusion | same |
| `postprocess_batch_size` accumulates frames before post-processing | DeepWiki on `sam3_multiplex_tracking.py` |

| approach | source | measured result | T4? | quality | effort | could save [my estimate] |
|---|---|---|---|---|---|---|
| Object Multiplex | Meta release notes | 7× at 128 objects, 2× at medium density, **H100** | code runs, but gain scales with object count | none claimed | already in your core nodes | **~0 for you: you track 1 face → 1 bucket pass either way** |
| Batched vision encoder / postprocessing | Meta release notes | "batched … to increase GPU utilization", no numbers published | unknown | none | needs ComfyUI-side port | 10–25 s [my estimate] |
| ComfyUI's port (PR #13408, kijai) | [PR #13408](https://github.com/Comfy-Org/ComfyUI/pull/13408) | "re-implementation for ComfyUI, extra dependency free and optimized for single GPU"; ~2× faster than the third-party wrapper SAM3 pack | yes, runs today | n/a | done | — |
| torch.compile / CUDA graphs for SAM3 | searched: no source found | **nothing exists** | — | — | — | — |
| Faster SAM3 video node | searched: ComfyUI-Easy-Sam3, wouterverweirder/comfyui-sam3, TBG-SAM3 all wrap the same model, none claims faster tracking | **no faster node exists** | — | — | — | — |
| Batch backbone across frames | — | **not possible as-is**: the tracker is stateful (memory attention over past frames); Meta's "batched vision encoder" batches *objects/buckets*, not frames | — | would change masks | — | — |
| Remove the per-frame host round trips | code read of `comfy/ldm/sam3/tracker.py`: 2 `.cpu()` calls per frame in `fill_holes_in_mask_scores`, 1 sync per detection in `_nms_masks`, one `intermediate_device()` D2H per frame | your profile: 35 syncs/frame, GPU ~60% busy | yes | decision-identical (proved in `check_sam_fast.py`) | done in this branch | 25–40 s [my estimate] |

**Plainly:** Meta's 7× number is about *many objects*; you have one. The parts of 3.1 that would
help you (sync removal, batched postprocessing, compile) are **not** in ComfyUI's reimplementation —
its own hole-filling still does two host round trips per frame. Your best lever here is the sync
removal, which is what `sam3_fast.py` in this branch already does. Also note: ComfyUI disables
SageAttention for SAM3 ([PR #13529](https://github.com/Comfy-Org/ComfyUI/pull/13529), NaNs), so
SAM3's attention is fp16 by design.

---

## Q2. Attention faster than a SageAttention-style int8 kernel at ~35 TOPS on T4

| approach | source | measured result (hardware) | T4/sm_75? | quality | effort | could save [my estimate] |
|---|---|---|---|---|---|---|
| **Your current kernel** (kitchen int8) | your runs | 335 ms/call @ 28.5k tokens, 34.8 TOPS = **27% of T4 int8 peak** | yes | baseline | — | — |
| SageAttention v1 (INT8 QK, FP16 PV) — the *only* SageAttention for Turing | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — `sageattn_qk_int8_pv_fp16_cuda_sm75`, v1 branch, `pip sageattention==1.0.6` | paper: 340 TOPS on RTX 4090 @ head_dim 64/128 = 52% of that card's int8 peak; **no T4 numbers published** | yes (v1 only) | negligible loss in the paper's tests | low (a pip install) — but it's the same math your kitchen kernel already implements | ~0 vs kitchen; useful only as an A/B |
| SageAttention2 / 2++ / 3 | same README: "Optimized kernels for **Ampere, Ada and Hopper**"; SA2 = INT4 QK + FP8 PV | 3–5× over FA2 on 4090 | **no** (FP8 + no Turing build) | — | — | — |
| FlashAttention-Turing (fp16, hand-written) | [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) | T4, head_dim 128: up to **2.19× vs PyTorch** non-causal (PyTorch = xformers mem-efficient on Turing); "long sequences … up to **66% compute throughput**" | yes | fp16 exact | high (kernel port + integration) | 66% of fp16 peak = 21.5e12 MAC/s vs your 17.4e12 → **1.24× = ~8 s/video** [estimate] |
| The same efficiency in int8 | arithmetic on the two rows above | if an int8 kernel hit 66% of int8 peak = 43e12 MAC/s, attention 14.9 s → **~6 s/step (2.4×)** | — | int8 rounding (already accepted) | high | ~50 s/video [estimate] |
| SpargeAttn / SVG / SVG2 / Sparse-vDiT | [SpargeAttn](https://arxiv.org/html/2502.18137v4), [SVG](https://arxiv.org/html/2502.01776v1), [SVG2](https://www.researchgate.net/publication/392106085), [Sparse-vDiT](https://arxiv.org/html/2506.03065v1) | 1.83× (Mochi, L40), 2.28–2.33× (CogVideoX/HunyuanVideo, H100), 1.76–1.85× (A800) end-to-end | **no sm_75 support found**; built on FA2/FP8/CuTe | PSNR 24–29 vs baseline | — | — |
| **comfy-kitchen Sol-Attn + token routing** | [PR #156](https://github.com/Comfy-Org/comfy-kitchen/pull/156), [capability matrix](https://pypi.org/project/comfy-kitchen/) (`sol_attn`: eager ✓ cuda ✓ triton ✗) | token_aug 0–256 top-scoring extra tokens; "CUDA only for now" | **unstated for sm_75** — must be tested | designed to preserve quality; dense prefix protection | low (a flag/node) | unproven on T4 |
| **comfyui-turing-utils** (Turing-specific pack) | [brahianrosswill/comfyui-turing-utils](https://github.com/brahianrosswill/comfyui-turing-utils) | "bundled **sm75+ W8A8 attention**… retains stable Sage's INT8 Q/K score domain, quantizes V channel-wise to INT8, packs probabilities to uint8, and evaluates **both QK and PV with Turing Tensor Cores**"; native sm75+ Sol sparse; "sm80+ uses async copies… **without** Triton" | yes, claims native sm75 | claims stable-Sage numerics | low–medium | unknown; the int8-PV core is the same scheme as yours, so the win would be scheduling [my estimate 10–30 s/video] |

**Plainly:** your kernel is the SageAttention-1 algorithm already, running at 27% of the int8 wall
while a hand-written fp16 kernel has been measured at 66% of the fp16 wall on the same chip. That is
the finding: **the gap is schedule efficiency, not precision** — and it caps at ~2.4× (≈50 s/video).
SageAttention2+, SpargeAttn, SVG2 and FA2/3 are all Ampere+ and cannot run on your cards.

---

## Q3. Step caching at 6 turbo steps

| approach | source | measured result | T4? | quality | effort | could save [my estimate] |
|---|---|---|---|---|---|---|
| **FirstBlockCache for H3** | [FurkanGozukara/ComfyUI-TeaCache](https://github.com/FurkanGozukara/ComfyUI-TeaCache) — "MiniMax H3 Speed Optimizer (NVlabs Sana sol-engine port)" | "block 0 runs every step; when its output residual barely moved, the remaining **49 blocks are skipped**… the reference's **dominant 2.58× stage**"; ported from [NVlabs Sana sol-engine](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/optimized), "whose measured full line is **3.97×** on the denoise+decode hot path". "FirstBlockCache always computes the final step, **including 4/8-step schedules**." | yes (pure PyTorch patch) | threshold + schedule window + consecutive-skip cap exposed | medium — must be wired around your `patches_replace` TP blocks | **1.2–1.5× on 255 s sampling = 45–85 s** [my estimate, contingent on hit rate] |
| ParaAttention FBCache (generic) | [ParaAttention](https://github.com/chengzeyi/ParaAttention), [diffusers docs](https://huggingface.co/docs/diffusers/optimization/para_attn) | FLUX on L20: 26.36 s → 17.01 s (**1.55×**, rdt=0.08), up to 2× at rdt=0.12; "nearly zero quality loss" | yes (model-level patch) | threshold-dependent | medium | same lever as above |
| Comfy-WaveSpeed | [Comfy-WaveSpeed](https://deepwiki.com/chengzeyi/Comfy-WaveSpeed) | 1.5–3× claimed; thresholds tabulated per model (FLUX 0.12, HunyuanVideo 0.1) | yes | as above | low | same |
| TeaCache at **low step counts** | [sd-forge-blockcache](https://github.com/DenOfEquity/sd-forge-blockcache), ComfyUI-TeaCache README | "low step models (**Hyper**) will need **higher threshold to do anything**"; "The `4x` name describes the reference stack, **not a promised speedup on every GPU or step count**" | — | high thresholds at 6 steps raise artifact risk | — | your 6-step case is the *hard* case for caching |
| MagCache / T8 block cache | searched | MagCache found for other models; **no H3 support found**; T8 = no results | — | — | — | — |

**Plainly:** the lever exists, is H3-specific, and is already ported from NVIDIA's reference — but
6 turbo steps is the worst case for it, and every source warns the speedup is step-count dependent.
Worth an A/B on one clip with a conservative threshold before believing any number.

---

## Q4. Qwen3-VL 32B encode

**First, why it costs 59 s — my arithmetic on your numbers:** the int4 encoder is 14.2 GB; one pass
over it at the T4's 320 GB/s is **44 s**. Your 59 s is 1.33× that. So the encode is
**weight-bandwidth-bound, not compute-bound** [estimate]. Consequences: caching is the only fix for
repeat runs (done), and a smaller encoder scales the cost down roughly with its size.

| approach | source | measured result | T4? | quality | effort | could save [my estimate] |
|---|---|---|---|---|---|---|
| Cache the vision tower (ViT/LLM split) | this branch (`h3_qwen_cache.py`), keys on pixels | boundary is `MiniMaxQwen3VL.preprocess_embed`; prompt-only changes reuse the tower | yes | none (bit-identical) | done | the tower is a small part of a 50-layer LLM pass — **5–12 s**, not the 40 s I estimated earlier [my estimate] |
| **ClipProj: 4B/8B + learned projection** | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3), [comfyui-wiki](https://comfyui-wiki.com/en/news/2026-08-09-clipproj-minimax-h3), [note.com measurement](https://note.com/sepiablue/n/n6a78bb01511a?hl=en) | VRAM 15.7 → 4.5 GB (int8 4B) / 8.3 GB (bf16 4B); ridge-regression projection, **cross-prompt cosine ~0.71**; user's own 124 f / 8 step run: **135.23 s (32B) vs 132.05 s (4B)** — no wall-clock gain in that setup because the encoder is unloaded during sampling | not arch-specific | author: swap costs "about what re-rolling the seed costs" for speech; the projection was fitted on **prompts** | low (a custom node + matrices) | encode 59 s → **~15–20 s** if the smaller model is resident [estimate from the 44 s bandwidth floor], **but see the caveat** |
| NVFP4 32B encoder (keeps vision tower in bf16) | [6block/MiniMax-H3-Qwen3-VL-NVFP4](https://huggingface.co/6block/MiniMax-H3-Qwen3-VL-NVFP4) | 16.64 GB, SSIM 0.891 / LPIPS 0.099 vs bf16; "runs on RTX 4090" — **NVFP4 needs Blackwell/sm_120** | **no** (FP4 is not a Turing format) | — | — | — |
| Official smaller H3 text encoder | searched | **does not exist** — H3 ships Qwen3-VL-32B truncated to 50 layers; every alternative is community | — | — | — | — |

**The caveat that matters for a face swap:** ClipProj's projection is calibrated on prompt
embeddings, and it replaces the **vision tower too** — the part that encodes your reference face.
For an identity-locked face swap that is exactly the component you least want to change. Test it on
your own clips before trusting the "same quality" claim.

---

## Q5. Better use of 2 PCIe GPUs than TP with ~15% exchange

| approach | source | measured result | T4/sm_75? | quality | effort | could save [my estimate] |
|---|---|---|---|---|---|---|
| PipeFusion (patch pipeline) | [arXiv 2405.14430](https://arxiv.org/html/2405.14430v1) | 4×A100 PCIe: 2.01× vs best other method @1024px; "latency on PCIe is **on par with NVLink**"; memory 32–36% of SP | architecture-neutral method but implemented in xDiT/diffusers | FID unchanged | **very high** (port H3 into xDiT) | your exchange is 15% and overlapped; their win is at much bigger comms ratios — **likely ~0, and weeks of work** [my estimate] |
| xDiT (SP + PipeFusion + CFG) | [arXiv 2411.01738](https://arxiv.org/html/2411.01738v1) | 4.55× on 6×L40 **PCIe** (CogVideoX-5B); 6.0× on 12×L40 Ethernet | same caveat | FID comparable | very high | same |
| ParaAttention context parallel (Ulysses/Ring) | [ParaAttention](https://github.com/chengzeyi/ParaAttention) | 2×L20 with FBCache+FP8+CP: 5.35× vs 1-GPU baseline — **CP not isolated from the other two** | CP kernels are FA/Ulysses paths (Ampere+) | none claimed | high | ~0 [my estimate] |
| Your TP | your runs | 1.6–1.8× vs Split; 58.8 vs 104.5 s/step @0.7 MP | working today | validated by your `MMH3_TP_CHECK` | done | — |

**Plainly:** I found **no published 2-GPU result for a ComfyUI-native model on PCIe-only T4s**, and
nothing that would beat a working TP implementation without moving H3 into another engine. The
papers' PCIe results are for models and engines that already live in diffusers/xDiT. Rank this last.

---

## Q6. Cutting H3 tokens without quality loss

| approach | source | measured result | T4? | quality | effort | could save [my estimate] |
|---|---|---|---|---|---|---|
| **Sol-Attn / SLA / VSA sparse attention in core ComfyUI** | [PR #16072](https://github.com/Comfy-Org/ComfyUI/pull/16072) (kijai) + [comfy-kitchen PR #156](https://github.com/Comfy-Org/comfy-kitchen/pull/156); H3 exposes layout + block metadata | author's numbers: 768p/15 s, 4 steps, RTX 5090 — dense first step ≈96 s of a 183 s run, ~44 s on `comfy_kitchen_int8` | **SM75/Turing is a "reduced-feature supported target"** per [Zironic H3 Optimizations](https://comfy.icu/extension/Zironic__H3-Optimizations): "dense Comfy Kitchen INT8 and the shipped native sparse Kitchen path are eligible; BF16 Triton, FROST, FP8 and Sparse Sage remain unavailable on SM75" | protected prefixes/reference blocks; SLA mode expects an SLA-trained LoRA | low–medium (a node/flag) | attention is 35% of your step; if routing skips ~50–85% of blocks, **up to 25–40 s/video** [my estimate, unproven on T4] |
| **Turbo-SLA (H3-specific)** | [comfy.icu/TuringUtilsSlaSparseAttentionPatch](https://comfy.icu/node/TuringUtilsSlaSparseAttentionPatch) (wjie98/comfyui-svdint4 "Turing Utils") | fixed-budget 128Q×64KV Top-K, `sparsity_ratio=0.85` "matches the published H3 Turbo-SLA hyperparameter"; W8A8 Tensor Core path; "native sm75" claimed | yes per its docs | "should be used with an SLA-trained LoRA" — i.e. it is a *trained* sparse mode, not a bolt-on | low (install pack) | same as above; requires the SLA LoRA |
| Reference/face protection while sparsifying | same pack + Sol node | `sparse_reference_image=false` keeps reference-image Query dense and its KV exact | — | designed to protect identity-bearing inputs | — | — |
| Region-restricted attention for the crop | your workflow already crops to 0.5 MP | — | — | — | done | — |
| H3 latent upscaler (low-res → refine) | searched | **nothing exists for H3** | — | — | — | — |
| RIFE for the face region | [ComfyUI now ships native FILM/RIFE v4.26](https://www.reddit.com/r/comfyui/comments/1tpn7i9/fyi_theres_now_native_frame_interpolation_in/) | frame interpolation, not a sampling reduction | yes | interpolated frames on a face are the risk | low | sample 62 frames instead of 124 → ~½ the sampling **but** this halves temporal information the model sees; quality risk on identity [my estimate] |
| SVG / SpargeAttn style sparsity | see Q2 | 1.8–2.3× end-to-end on other video DiTs | **no sm_75** | PSNR 24–29 | — | — |

---

## Q7. The ~25% small-op overhead

| approach | source | measured result | T4? | quality | effort | could save [my estimate] |
|---|---|---|---|---|---|---|
| Note what kitchen already fuses | [ComfyUI-TeaCache README](https://github.com/FurkanGozukara/ComfyUI-TeaCache) | "ComfyUI core already fuses **QKV projection, RMSNorm+partial-RoPE and SwiGLU** via comfy-kitchen" — so your 25% is the *other* elementwise work: modulation scale/shift, gates, casts, fp32 residual | yes | — | — | — |
| Fused residual-add + RMSNorm, T4-specific | [jarnesino/fused-rmsnorm-residual-add-triton](https://github.com/jarnesino/fused-rmsnorm-residual-add-triton) | written *for the T4*; targets 320 GB/s; tuned warp counts, in-place output measured within 0.1%; documents the register/spill cliff at 255 regs (T4 limit) | **yes, explicitly** | exact | medium (Triton, you already have the toolchain) | part of the 10.6 s class |
| Fused RMSNorm+RoPE | [AICL-Lab/triton-fused-ops](https://github.com/AICL-Lab/triton-fused-ops) | up to ~3.0× vs PyTorch, ~40% less memory traffic | Triton (arch-neutral) | exact | medium | — |
| Cost of *not* fusing | [vLLM PR #55311](https://github.com/vllm-project/vllm/pull/55311) | the unfused path is "5–6 kernels and **two fp32 round trips per norm**" | — | — | — | this is your 25% |
| Bounded-fp16 execution for H3 | [Zironic H3 Optimizations](https://comfy.icu/extension/Zironic__H3-Optimizations) | "bounded FP16 QKV/MLP/FinalLayer execution" eligible on SM75 — the same idea as your 1/64 scaling, extended | yes | fp16 rounding | low (install) | unknown |
| Hand-fused modulation (your attempt) | your run | 44.9 vs 42.5 s/step — slower | — | — | reverted | 0 |

---

## Q8. Hardware reality check

| item | source | number |
|---|---|---|
| T4 spec | [NVIDIA product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-product-brief.pdf) | 130 TOPS int8, 65 TFLOPS fp16, **no** bf16/FP8, 320 GB/s, 70 W max |
| T4 power under full load | [ServeTheHome](https://www.servethehome.com/nvidia-tesla-t4-ai-inferencing-gpu-review/5/) | **74 W** measured (36 W idle) |
| Sustained int8 TOPS under the 70 W cap | searched | **no published measurement exists.** The only arch-relevant datapoint found: a hand-written fp16 kernel hits **66% of fp16 compute throughput** on T4 at head_dim 128 ([ssiu](https://github.com/ssiu/flash-attention-turing)) — i.e. the chip *can* be driven to ~2/3 of its tensor-core ceiling |
| Best measured efficiency on a *modern* card, same algorithm as yours | SageAttention paper | 340 TOPS on RTX 4090 = **52% of int8 peak** |
| Your kernel | your runs | 27% of int8 peak |
| Cheap cloud alternatives | [altstreet 2026 table](https://altstreet.investments/tools/gpu/gpu-price-comparison), [promptquorum](https://www.promptquorum.com/local-llms/cloud-gpu-rental-comparison-2026), [RunPod](https://www.runpod.io/articles/guides/ai-server-cost) | RTX 4090 24 GB **$0.14–0.34/hr** (Vast spot / RunPod community); L40S 48 GB $0.39–1.09/hr; A100 80 GB $0.69–1.80/hr; H100 $1.49–2.69/hr |

**Plainly:** the T4's ceiling for a well-scheduled int8 attention kernel is plausibly 2× what you
get now, and no source contradicts that. But the cheapest *large* win is not a kernel: a rented
**RTX 4090** has ~5× the T4's int8 rate, 24 GB of VRAM, and **sm_89 FP8**, which unlocks the whole
SageAttention2/FP8 kernel family that is unavailable to you today. At $0.14–0.34/hr that is the only
route to a 2–4× on this workload without a rewrite [my estimate]. Your Kaggle 2×T4 is free, so this
is a cost decision, not a technical one.

---

## Ranked shortlist

1. **SAM 3.1 per-frame sync removal.** Biggest *certain* win: your own profile (35 syncs/frame, GPU
   60% busy) plus the code read showing two host round trips per frame in hole-filling and one D2H
   per frame for masks. Built in this branch, decision-identical by test. 25–40 s/video, no quality
   risk. Meta's headline numbers do not apply to a single tracked face — this is the part of 3.1
   that does.
2. **FirstBlockCache for H3.** The only *step-level* lever that needs no new kernel, is already
   ported from NVIDIA's reference for this exact model ("the reference's dominant 2.58× stage"), and
   is a pure PyTorch patch that fits your `patches_replace` structure. 45–85 s/video at 1.2–1.5×,
   with the explicit caveat that 6 turbo steps is the hard case and every source warns the win is
   step-count dependent. A/B one clip before believing it.
3. **Sparse attention on sm_75** — two independent routes to test cheaply: core `comfy-kitchen`
   Sol-Attn/SLA via [PR #16072](https://github.com/Comfy-Org/ComfyUI/pull/16072), and the
   Turing-specific [comfyui-turing-utils](https://github.com/brahianrosswill/comfyui-turing-utils)
   which *claims* native sm75 Sol + W8A8 with H3-specific prefix protection. Attention is 35% of
   your step; if a 50% skip holds up on T4, that is 25–40 s/video. Highest ceiling, unproven on your
   arch — test before planning around it.
4. **Your int8 kernel, made 2× better.** Already written in this branch; the research says the
   target is real (66% efficiency is achievable on this chip in fp16; SageAttention hits 52% of int8
   peak on a 4090). ~50 s/video at 2.4×. Certifiable with the `--selftest` you already have.
5. **Rent a 4090 (or L40S) for a day.** Not a kernel, and not free, but it is the only 2–4× that
   needs no porting, and FP8 unlocks kernels the T4 can never run. $0.14–0.34/hr.

Honourable mention: your Qwen encode is **weight-bandwidth-bound** (14.2 GB ÷ 320 GB/s ≈ 44 s of the
59 s) — so caching (done) and a smaller encoder are the only levers, and a smaller encoder means
changing the vision tower that encodes your reference face. Test ClipProj on your own clips first.

## Does not exist (searched, nothing found)

- No published **T4 benchmark for SageAttention** (the sm_75 path exists in v1; no numbers).
- No **sustained int8 TOPS measurement for the T4 under its 70 W cap**.
- No **faster SAM 3/3.1 video node** and no way to batch the backbone across frames (the tracker is
  stateful by design).
- No **H3 latent upscaler**.
- No **sm_75 support** in SpargeAttn, SVG, SVG2, Sparse-vDiT, FA2 or FA3.
- No smaller **official** H3 text encoder; no **MagCache or T8** support for H3.
- No published **2-GPU PCIe result for a ComfyUI-native model** to compare your TP against.
