# Review: pre-split pieces and partial eviction

Reviewed 2026-11-09 against `tp-modular` at `cc1907e`. Scope: review the plan and existing runtime, consult Claude, and run existing CPU checks. No piece implementation or new GPU measurements were performed.

**Verdict: promising, but revise the plan before implementing it.** The smallest first step is a measurement probe, including today's retained-rank rebuild. The current 75-second full rebuild alone is not a sufficient baseline.

## Model structure checked after the initial review

The user's central proposal **does fit this model's structure**: keep GPU0's half loaded, divide GPU1's half into smaller groups, and remove only enough groups for another task. Restore the removed groups before H3 sampling resumes. The implementation concerns below do not invalidate that proposal.

Read the actual public Olivia int8 checkpoint's 99,664-byte safetensors header using HTTP range requests, without downloading the 20 GB weights. It contains exactly 50 main blocks (`blocks.0` through `blocks.49`), with matching tensor names, shapes, and dtypes across all 50. The four main projection weights are int8, with shapes `[21504, 5376]`, `[5376, 7168]`, `[28672, 5376]`, and `[5376, 14336]`.

The current TP splitter divides the large weights of **every block** between GPUs; it does not assign 25 complete blocks to each GPU. Per GPU, main weights account for 192,675,840 bytes per block, about **0.897 GiB per five-block group**, or **8.972 GiB across all 50 blocks**. Scales and norms add a small amount, consistent with the roughly 8.99 GiB measured shard residency. Thus ten roughly 0.9 GiB groups on GPU1 are a structurally reasonable starting point.

Also traced the local ComfyUI H3 source (`comfy/ldm/minimax/model.py` in `D:/MY AI/comfyui-source-audit`): it constructs repeated `DiTBlock` modules and executes them sequentially through the block replacement interface used by this fork. Both GPU halves are used inside each replaced block. GPU0 cannot continue the current H3 generation path alone while GPU1's pieces are absent; it can retain its weights while the other task runs.

Example only: if the next GPU1 task needs 6 GiB of total free memory including its working space, and GPU1 currently has 4 GiB free, the shortage is 2 GiB plus the chosen margin. Release enough roughly 0.9 GiB groups to cover that shortage, run the task, release its allocations, then reload only the missing groups. Actual free memory, allocation peaks, and transfer speed still require the planned GPU measurements.

Probe and machine-readable result: `portkit_runs/h3_pieces_review/probe_header.py` and `checkpoint_structure.json`. This verifies checkpoint structure and byte counts, not payload integrity, GPU execution, or reload performance.

## Follow-up: VAE lifecycle and recommended piece count

Read the complete current `h3_dual_vae.py`, TP cleanup, and `E:/Local WebUI/Comfy Remote/remote/auto_dual.py`, plus local ComfyUI VAE estimates and encode/decode loading boundaries. Consulted **Claude Opus 5.5** again with this actual source: [VAE review task](https://www.kaggle.com/benchmarks/tasks/tsdrp34/h3-vae-pieces-review/1), completed run **6543838**, returned identity `anthropic/claude-opus-5-5@default`. Raw answer and provenance are in `portkit_runs/h3_vae_review/`. No new GPU measurements or runtime edits.

The user's sequence is sound: make room before preprocessing/encoding, offload competing models and restore missing H3 pieces before sampling, then make room again for decoding if required. After decoding, keep missing pieces absent until the next sample actually needs them. A workflow without preprocessing may still need eviction for its VAE; such a workflow can therefore incur partial restore on its next generation.

Current behavior:

- Standard H3 video decode in Comfy Remote routes to dual decode when the VAE is on GPU1 and the recognized causal H3 VAE/name conditions match. GPU1 is primary and GPU0 holds a decoder helper. This is conditional, not a universal route for every VAE/decode variant.
- Standard encode does **not** automatically select `VAEEncodeH3DualGPU`; that node exists separately. H3 conditioning nodes also call `vae.encode()` directly, bypassing the standard encode node. A complete guard must cover those entry points too.
- Sampling cleanup already frees exchange buffers and GPU LoRA copies while retaining H3 shards. Before sampling, managed models on both GPUs are requested to unload. Cached VAE helpers are managed models; clearing `_mmh3_helper` at the end of a VAE node does not itself offload their weights.
- The dual helper is loaded with `force_full_load=True` but without an explicit clip-specific activation budget. The primary VAE's ordinary budget does not by itself reserve helper workspace on the other GPU. Dual decode also returns a decoded helper clip to the primary GPU, which must be included in peak calibration.
- Neither H3 VAE node currently partially evicts H3 weights. Standard encode's `park_for` probe is ineffective for H3 because H3 does not define that function. Claude overgeneralized this to other TP families: `ltx_tensor_parallel.py` **does** define `park_for`.

**Initial recommendation: two blocks per piece, 25 pieces per GPU, 50 files total.** Both GPUs should have the same eviction capability, even if GPU0 normally retains everything. Keep grouping configurable and benchmark one-block pieces before fixing the default.

| Blocks per piece | Pieces per GPU | Main weights per piece | Approximate maximum extra eviction per affected GPU |
| --- | --- | --- | --- |
| 1 | 50 | 0.179 GiB | less than 0.18 GiB |
| **2** | **25** | **0.359 GiB** | **less than 0.36 GiB** |
| 5 | 10 | 0.897 GiB | less than 0.90 GiB |

These sizes exclude small scales/norms; exact resident bytes belong in the manifest. Five-block pieces are coarser than needed for a small VAE shortage, especially when both GPUs need space. One-block pieces are worth preferring if measured file overhead is negligible. Neither 1 nor 2 is established as fastest.

For example, if the *remaining shortage after including a margin* is 1 GiB on GPU1, three two-block pieces free about 1.08 GiB. GPU0 needs its own calculation. The calculation should first reclaim unused managed models, then evict H3 only for the remaining shortage, and re-measure actual free memory. Already resident task weights must not be charged twice.

Remaining design/validation requirements:

- First cold means no H3 shards yet, not guaranteed-empty GPUs. Cached helpers, text encoders, and models outside ComfyUI's registry still need accounting.
- A preprocessor's working-memory demand is not universally declared. Use calibrated profiles for intended models/shapes/devices, with conservative handling for unknown cases. Cached nodes that do not execute need no eviction.
- Dual VAE needs budgets for both GPUs; keeping GPU0 intact is a preference. Measure any policy that chooses single-GPU VAE instead of evicting GPU0; current node fallback conditions do not implement a memory/cost-based policy.
- Include weights still to load, measured working space, transfers/output buffers, and a margin. Do not blindly add the same output-buffer estimate twice: the primary/helper formulas in Claude's answer are uncalibrated and may overlap the ordinary VAE estimate.
- Bound total pinned staging across both GPUs and all reader workers. A limit per rank is not a global host-RAM limit.
- Every VAE worker must finish before eviction/restoration. Source inspection shows the normal joins occur after primary work; exceptional primary failures also need reliable joining before memory can be reused.
- Before sampling, budget missing pieces **and sampling working space/non-block allocations**, not merely H2D capacity. Refuse partial state. GPU numerical checks, repeated-cycle leak checks, per-stage memory measurements, and actual reload timing remain mandatory.

## Consultation and evidence

Followed `.claude/skills/dual-gpu-port/SKILL.md` and read `docs/PORTING.md`, `docs/REPAIR.md`, and `docs/golden/h3.json`.

The self-contained prompt included the original plan and the complete current `h3_tensor_parallel.py`, `tp_auto.py`, `tp_source.py`, and `tp_native_layout.py`.

- [Kaggle consultation](https://www.kaggle.com/benchmarks/tasks/tsdrp34/h3-pieces-design-review/1), run **6543468**.
- Returned model identity: **`anthropic/claude-opus-5-5@default`**; state: completed.
- Local prompt, raw answer, and provenance: `portkit_runs/h3_pieces_review/` (gitignored).
- Kaggle also performed its automatic initial validation using Gemini. The conclusions below were checked against the subsequent Claude response and the repository source.

Claude's recommendations are design hypotheses, not measured results. The qualifications below intentionally depart from parts of its response.

## Changes needed in the plan

| Priority | Finding | Required correction |
| --- | --- | --- |
| Blocker | `tp_auto.attach()` supplies a `ModelWeights` object, not a file path. `ModelWeights` exposes no checkpoint filename or file SHA. It can also requantize weights. | Start with explicit file-backed native-int8 H3 loaders. Leave automatic sources on the existing path until stable base-weight identity is designed and verified. Never use `id(dm)` as persistent identity. |
| Blocker | Golden peaks include resident weights and other allocations. The proposed free-memory formula double-counts weights if those peaks are treated as extra headroom. | Measure additional memory from a defined preparation/forward boundary, separately accounting for missing shards, not-yet-loaded non-block modules, LoRA, activations, and transient reload allocations. Do not combine differently scoped golden runs. |
| Blocker | `tp_auto._build()` runs before `_prepare_sampling()` frees other models. Explicit loaders return early from `tp_auto.attach()`, and `block()` has another lazy load path. | Give the enabled piece path one restore boundary after model cleanup and before execution. Check both explicit and automatic attachment behavior. Prove the exact ComfyUI loading order on Kaggle. |
| Blocker | `block()` checks only for an absent whole rank. A rank containing missing block entries passes that guard; prefetch then accesses the next block directly. `_memdump()` also assumes every block is present. | Make partial state explicit and keep it inaccessible to sampling. Restore and publish complete state before execution. Audit every reader, cache alias, release path, and diagnostic. |
| Blocker | A sampler hook runs after a preceding preprocessor. It cannot provide the promised automatic eviction before that preprocessor allocates. | Limit v1 eviction to explicit calls or a specifically integrated preprocessor. Automatic demand hooks require a separate design review. Measure the actual target GPU; evicting rank 1 does not help a model allocating on GPU0. |
| High | The shard dictionaries contain block projections, scales, norms, and small metadata. Other layers remain owned by ComfyUI; `_mods()` invokes live `adaln_proj`. | Omit `common.safetensors` from v1. Persist exactly the runtime shard dictionaries, preserving scalar metadata and the replicated output scales. Keep existing non-block patches and LoRA ownership. |
| High | `_SHARD_CACHE[path]` and `tp.shards` can reference the same lists. `release_encoder_rank()` changes both references. Temporary tensors, futures, or diagnostics can also retain allocations. | Specify one authoritative owner and a transaction boundary. Synchronize GPU work, release all holders, and publish only after every transfer succeeds. A lock must cover the sampling lifetime and exceptional cleanup, not just the reload call. |
| High | Full rebuild or one-GPU fallback does not necessarily fit. A CUDA failure is different from a file validation error. | Permit a safe pre-execution refusal. Retry source reconstruction only when the CUDA context is healthy and a fresh budget check passes. Never continue sampling with partial state; do not promise same-process recovery after CUDA OOM/context errors. |
| High | `/kaggle/temp` pieces disappear with the session. Export, hashing, writing, and round-trip checking add first-session cost. | Distinguish first-session build/export, existing-piece load, and eviction reload. Report export amortization and the number of reloads needed to recover that cost. |
| Medium | `CHUNKS` controls activation exchange, not the persisted weight split. File bytes also differ from resident tensor bytes. | Treat chunk count as benchmark configuration unless a demonstrated layout dependency exists. Record file bytes, resident tensor bytes, tensor shapes/dtypes, block coverage, scalar metadata, grouping, and layout version separately. |
| Medium | Reverse block order is deterministic but need not choose the fewest pieces if piece sizes vary. | Define either trailing-order eviction or minimum-count eviction. Define `bytes_needed` as the desired total free device memory, with the safety margin specified separately. Re-measure after eviction. |

The original acceptance limit is about **13.56 GiB** if usable capacity is 14.56 GiB and the required margin is 1 GiB. The existing legacy 40.7k-token GPU0 peak is **14.3 GiB**, already above that limit. Scope the new acceptance workload explicitly; do not imply that existing high-token measurements satisfy the new margin.

## Recommended sequence

1. **Measure without changing the runtime.** Time source reads, CPU slicing/packing, cleanup, and transfers separately, for full rebuild and `release_encoder_rank()` restoration. Record source storage location and observed cache state. Measure 1/4/8-thread piece-sized reads, SHA-256 throughput, write/round-trip cost, and bounded staging RAM. Label first reads honestly; cache eviction advice alone does not prove a cold read.
2. **Measure the intended preprocessors.** Identify exact depth/pose models, execution device, clip dimensions, and peak activations. Until these are known, dummy allocations validate bookkeeping only, not production preprocessor safety.
3. **Choose the smallest implementation using those results.** Compare pre-split reload with a retained-rank rebuild and, if justified, a small read/staging optimization. Do not commit to a new buffer format or arbitrary speedup threshold before the measurements.
4. **Implement an opt-in file-backed exporter/reader.** Preserve the existing safetensors proposal initially, serialize only actual shards, bound host staging, verify exact CPU round trips, validate source identity, and publish the manifest last through an atomic export transaction. Keep the default-off runtime path unchanged.
5. **Verify whole-rank restore before partial pieces.** Place restoration after cleanup, include explicit loaders, and prevent lazy mid-forward restoration when pieces are enabled. Prove the source-error and insufficient-memory outcomes before adding partial state.
6. **Add explicit piece eviction and restore.** Preserve base-weight and LoRA semantics, synchronize and lock state transitions, add state/preflight logs, and test repeated cycles and failure injection. Automatic preprocessor demand integration is a later reviewed change.
7. **Run short Kaggle verification and update golden data only from results.** Compare same-seed outputs before/after export and eviction, validate all reload/failure paths, and report both GPUs' allocated/reserved/free memory and reload timings. Keep tagging, notebook changes, HF uploads, and enablement behind the original approval requirements.

## Advice from Claude that needs qualification

- **Flat GPU buffers and 4096-byte alignment:** an additional architecture proposal, not a prerequisite. Fragmentation benefits under `cudaMallocAsync` and kernel compatibility with offset views are unmeasured. Keep this out of v1 unless probes establish a reason.
- **CRC/xxhash instead of SHA-256 on reload:** does not meet the original manifest/hash requirement. Measure SHA-256 cost and include it in timings; do not weaken corruption validation silently.
- **Path/size/mtime/header hash identity:** useful lookup hints, not proof that checkpoint tensor bytes are unchanged. Define full content validation and the lifetime of any cached validation result. Export does not automatically provide a full-file hash just because tensor data was read through safetensors.
- **String-prefix path containment:** insufficient. Use resolved path-component containment, validate piece names and symlinks, and delete only directories owned by this cache. Do not delete another valid model's cache merely because it differs from the current source.
- **One-GPU H3 is categorically impossible:** too strong. ComfyUI can offload/stream weights, but fit, host memory, performance, and safe detachment are unverified here. The plan must not guarantee this fallback.
- **Block0 error as a substitute for output equality:** insufficient. Keep the planned same-seed output comparison. Error versus the single-GPU reference and error versus the pre-eviction TP result answer different questions.
- **A particular owner class or fixed 2x speedup gate:** possible implementation/policy choices, not established requirements. Preserve the ownership invariant and use measured end-to-end benefit to justify complexity.

## Verification completed and limits

- `python -m unittest ci.test_h3_memory`: **7 passed**. These are existing CPU regressions for memory/LoRA cleanup, not tests of pieces.
- `git diff --quiet kaggle-ok-67 -- h3_tensor_parallel.py tp_auto.py tp_source.py`: no runtime differences for those files at review time.
- Local PyTorch reports no CUDA devices. No claims of measured piece reload speed, GPU numerical equality, preprocessor peaks, or T4 memory safety follow from this review.
- Runtime code, the original plan, golden numbers, tags, notebooks, and HF artifacts were not changed by this review.
