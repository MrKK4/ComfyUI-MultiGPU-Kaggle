# Porting a new model to dual-GPU tensor parallel (Kaggle 2x T4)

This is the contract and the playbook. It is written so that a model with no memory of this project can follow it. Everything here is measured on Kaggle 2x Tesla T4 unless marked (guess).

## 1. How the automatic dual GPU works

1. `tp_auto.install()` replaces `comfy.samplers.CFGGuider.sample`. Every sampler goes through it.
2. Before sampling, `tp_auto.attach(model_patcher)` looks at `model_patcher.model.diffusion_model` and asks each registered module `supports(dm)`.
3. The first module that says yes gets `mod.attach(model_patcher, source, key)`. `source` is a `tp_source.ModelWeights` (a safetensors-like view of the loaded model: any loader, any weight format; int8 convrot is passed through exactly, anything else is dequantized and re-quantized to int8 convrot group 256 on GPU0). `key` is `("auto", id(dm))`.
4. `attach` installs wrappers and block replacements on the patcher and returns the TP instance. `tp_auto._build` makes the shards before sampling so a failure falls back to one GPU.
5. If nothing supports the model, or attaching raises, the run uses one GPU and logs `no tensor-parallel support, one GPU`.
6. `tp_auto.MODULES` is the registry: `(display name, module, wrapper key)`. A new family is one new module plus one line there.

## 2. Module contract (copy the closest existing module and change it)

A module `<family>_tensor_parallel.py` must provide:

- `supports(dm) -> bool`: cheap class/attribute check on the diffusion model. Must not touch the GPU.
- `attach(model, source, key)`: build the TP object (`_INSTANCES[key] = ...`), register the wrappers under the wrapper key from `MODULES` (`diffusion_model` and `outer_sample` kinds), replace the blocks (either `model.set_model_patch_replace(fn, "dit", "double_block", i)` or a per-block `forward` override), return the instance.
- `_INSTANCES: dict`, and, if shards are cached by source, `_SHARD_CACHE: dict`. Anything that frees shards must pop BOTH (a one-line bug we hit: `release()` kept the cache entry and freed nothing).
- instance attributes: `path` (source id), `blocks`, `shards` (or `gpu`/`host`), `release(drop_shards=False)`.
- `_load_shards(path, n_blocks)` (or the family equivalent) that reads the weights of block `i` from `source` and returns per-rank shards. It must work from `ModelWeights`, not only from a file.
- Optional: plain-LoRA side paths (`set_lora`), see `h3_tensor_parallel.py`. Unsupported patch kinds must raise, never be ignored.

Existing references: `h3_tensor_parallel.py` (MiniMax H3, int8 convrot, the best tested), `qwen21_tensor_parallel.py`, `krea2_tensor_parallel.py`, `ltx_tensor_parallel.py`, `trellis2_dual_attention.py` (attention-only). Variants live next to the base by suffix; plug-ins live in their own folder (`veda/` is a sparse-attention plug-in chosen through the attention backend option, not a copy of the TP file).

### Experimental H3 pre-split file loading

`h3_piece_loader.py` reads layout-v1 safetensors pieces created by `portkit/h3_pieces.py`. It is off by default and applies only to file-backed H3 TP sources, including `UNETLoaderH3TensorParallel`; automatic `ModelWeights` sources keep the existing builder. The original checkpoint remains necessary for the non-block layers.

To opt in, set all three variables before starting ComfyUI:

- `MMH3_TP_PIECES=1`
- `MMH3_TP_PIECES_DIR=/path/to/the/folder/containing/manifest.json`
- `MMH3_TP_PIECES_MANIFEST_SHA256=<trusted SHA-256 of manifest.json>`

Download the files separately into a runtime directory such as `/kaggle/temp`; the inference loader does not access the network. Use real files (for example Hugging Face's `local_dir` download), not snapshot symlinks. Keep the source and piece revisions pinned.

Before allocation, the loader checks the pinned manifest, source header, full source SHA-256, shapes, dtypes, scalar metadata, and complete per-rank block coverage. The full source digest is reused only while its resolved file identity/stat fingerprint stays unchanged. Each piece is read and hashed once into a bounded buffer before transfer. At most one piece per missing rank is pinned concurrently; PyTorch may cache those host allocations after the buffers are released.

File-backed restoration runs after managed-model cleanup and before sampling. Resident ranks are retained, and newly loaded ranks are published only after every worker completes. A file-validation failure logs a rejection and rebuilds from the original checkpoint. Memory refusal and CUDA errors propagate rather than triggering a potentially unsafe retry. Unexpected missing ranks during a forward are refused when pieces are enabled.

Rank workers preserve the caller's PyTorch inference mode. This context is thread-local: creating ordinary tensors in a worker while the original builder creates inference tensors changes Dynamo guards for the compiled norm kernels. Cache exhaustion can then silently change the arithmetic path. The loader regression checks both inference-mode settings and retained-rank restoration.

The legacy file-backed path supports loading and whole-rank restoration. The source-free bundle path below adds selective two-block eviction. Weight layout compatibility is controlled by the explicit layout version, not by comparing the current wrapper's source-code hash with the exporter's historical reference hash.

### Complete H3 bundle artifact and source-free loader

The public Hugging Face repo `ms13d/h3-olivia-tp2-pieces` now contains `bundle-v1/5b0fa95c09944191-5b414db68bb9` at commit `6749ffd51a785d6a9eeb6b92f4914d809f0b6080`. Its 1,007,639,416-byte companion holds all 548 original tensors outside the four split projections of each block. The bundle manifest (SHA-256 `76ee4c140d8d1ff9fd4fdb6668df9692c44601357ac41e917a12bba41e64e236`) pins the unchanged 50 `layout-v1` pieces at commit `0a67d46330a98720ddaacec16aa4d8733008d7cf`, the exact source header, all 948 tensor offsets and hashes, 200 TP module descriptors, and the detected ComfyUI H3 configuration.

Modal CPU verified that the bundle and pinned pieces reconstruct the original checkpoint SHA-256 `5b0fa95c09944191afd863239226683ae352e09104680f43c73ae206d48fe176` in a fresh process while the original source filename was hidden. An independent local CPU run obtained the same hash without the source checkpoint. The download and committed Hugging Face file hashes were checked. The full source file is unnecessary for this reconstruction.

`UNETLoaderH3TPBundle` consumes the companion and pinned pieces without the full checkpoint. Kaggle cold/warm one-step latent hashes exactly matched the full-checkpoint baseline; a four-step run repeated exactly and dual-GPU VAE decoded finite frames. These GPU results supersede the artifact manifest's historical `standalone_loader_validated=false` and `gpu_consumer_validated=false` fields, which are immutable in the pinned upload.

`TPEvictH3Image` and `TPEvictH3Latent` are explicit execution barriers. Connect the image/latent output to a preprocessor or VAE node. Set `free_gpu0_gib` and `free_gpu1_gib` to the measured free-VRAM targets for that stage; zero leaves a GPU alone. The node evicts the fewest trailing two-block pieces needed on each GPU, keeps the rest resident, and reruns on every workflow execution. The sampler restores only missing pieces before any H3 forward, with a lock against eviction during sampling. A failed restore raises and refuses to sample; a source-free bundle cannot fall back to a one-GPU checkpoint path.

Kaggle source-free verification evicted two GPU1 pieces, raised free VRAM to 6.10 GiB, allocated and freed 4 GiB for a simulated preprocessor, restored only those two pieces (2.15 s read/hash), and reproduced both baseline latent hashes. The cold load read all 25 pieces per rank (~75.7 s read/hash). A second run evicted one piece from each GPU after sampling, retained exact one-step latents, and decoded 124 finite frames through the standard dual-GPU H3 VAE. A third run used the user's exact `H3ControlDepth`/native Depth Anything 3 node at resolution 504 on one 448×768 frame: after intentionally evicting two GPU0 pieces, it produced finite depth, restored only those pieces (2.14 s read/hash), and matched both baseline latent hashes. The user's actual PDMD control workflow uses a single-GPU Sol decoder, a separate integration path.

The 124-frame ControlNet test with the user's video completed smoke, depth, ControlNet, sampling and standard int8 VAE decode with finite results. An instrumented repeat showed that the depth output already differed before ControlNet sampling; the same-seed final latents differed too. The user accepts that repeat variation for this workflow. Both runs used the Olivia bundle in place of the workflow's PRISM checkpoint. See `portkit_runs/h3_standalone/control_repeat_senior_answer.md` for Claude's hypotheses and `portkit_runs/h3_standalone/control_diag_output_v12/h3_up/control_verify.json` for the measured hashes and latent errors.

### H3 notebook automatic bundle mode

`MMH3_TP_BUNDLE_AUTO=1` is an H3-notebook-only opt-in. The notebook downloads the pinned Olivia bundle and pieces once, sets the bundle paths and manifest SHA, and exposes known H3 checkpoint aliases. The extension redirects H3 selections in ordinary `UNETLoader` and `UNETLoaderH3TensorParallel` to the Olivia split bundle; it logs the requested name and replacement. `weight_dtype` must be `default` for the ordinary loader. The full Olivia checkpoint is not downloaded by default. Non-H3 model names retain their normal loader.

The routed `UNETLoader.load_unet` wrapper exposes `__mmh3_bundle_router__` with `is_routed(name)` and `bundle_key(name)`. Comfy Remote's later loader wrapper uses those to retain the matching bundle's warm shards when a saved H3 workflow re-executes its loader, while releasing unrelated split models. Older extensions without the marker keep the normal release path.

After ComfyUI handles a managed `free_memory` request, the hook checks actual free VRAM and evicts only enough H3 pieces on the requested GPU to cover the request plus 256 MiB. It skips eviction during a sampler lease. The sampler restores missing pieces with SHA checks before H3 executes. Arbitrary custom nodes that allocate CUDA memory without requesting it through ComfyUI are not automatically covered; there is no unsafe retry after CUDA OOM. The explicit eviction nodes remain available for those workflows.

Kaggle verified ordinary `UNETLoader` routing from the PRISM filename with the full checkpoint absent. A forced 6 GiB managed request during a warm DA3 run evicted GPU0 pieces, restored only those pieces, and reproduced both baseline latent hashes exactly. A separate 124-frame ControlNet run with the user's video and no eviction nodes completed smoke, depth, ControlNet, sampling, and standard int8 H3 VAE decode; all functional checks passed. This tests the Olivia bundle and standard int8 VAE, not the PRISM weights or Sol decoder. The notebook pin still needs a new tested extension tag before this mode can be enabled in a published notebook.

## 3. The standard recipe for a transformer block

Megatron-style, two ranks, one process (rank r runs on `cuda:r`):

- Column-parallel (output split): qkv projection split by heads (each rank gets `heads/2` heads, rows of the fused qkv weight regrouped as `[q|k|v]` per rank), MLP first projection split by FFN width (gated MLPs: split both halves so each rank keeps matching gate/up rows).
- Row-parallel (input split): attention out projection and MLP second projection take the rank's half of the input columns and produce a partial sum of the full output. The two partial sums must be added: exchange them.
- Replicated on both ranks: norms, adaLN/modulation, rope tables, q/k norm (per head, shared), the residual stream (fp32).
- Exchange: no peer-to-peer on these T4s, so partial sums go GPU -> pinned host -> other GPU, in row chunks on side copy streams to overlap with compute. Payload = tokens x hidden x 2 bytes (fp16). Measured exposed cost about 3% of a step.
- Attention: each rank attends with its own heads, so no K/V exchange.
- int8 convrot linear: `comfy_kitchen` `int8_linear(x, w, scale, convrot=True, convrot_groupsize=256)`. Split dimensions must stay multiples of 256 on the input side.
- Everything per token (norms, modulation, rope, FFN) needs no communication. Anything that reduces across tokens inside a block needs its own small exchange: look for it in the probe report.

## 4. Feasibility rules (what the probe checks)

A model is a good TP candidate when ALL of these hold; the probe prints each as PASS/FAIL/UNKNOWN:

1. Blocks are uniform (same class, same shapes) and the count is known.
2. `heads % 2 == 0`; FFN width `% (2*256) == 0`; qkv/out/fc dims multiples of 256 where int8 convrot is used.
3. Attention goes through one function (`optimized_attention`-like) that receives `transformer_options`, so a sparse/other backend can plug in.
4. The block can be replaced: either the model's forward consults `transformer_options["patches_replace"]["dit"][("double_block", i)]`, or the block has a plain `forward` that can be overridden per instance.
5. No operation in the block mixes tokens besides attention (no global pooling, no token-dependent routing).
6. Quantization: int8 convrot (`.comfy_quant` JSON tensors, per-row or tensorwise scales) is used as is; other formats are dequantized to fp16 and requantized (check the cost: about 1-3 s per billion parameters on a T4).
7. Memory: shard bytes per GPU (about half the int8 file) leave room: resident ~9 GiB per GPU for a 21 GB file, activation headroom about 1.9 GiB at 19k tokens, about 0.2 GiB at 40k tokens (peak was 14.3-14.9 GiB). A model whose int8 weights exceed about 22 GB does not fit.
8. Multi-stream models (video + audio + text, reference tokens): list the streams and which pass through TP; segments that use different modulation need `mod_segments`-style handling (see H3).

## 5. Environment facts that cost us time (do not rediscover)

- Kaggle's runtime mixes cuDNN libraries; fp16 conv fails with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. Fix: start ComfyUI with `LD_PRELOAD` of the pip cuDNN libs (the Comfy Remote Start cell does this when needed). Disabling cuDNN instead makes conv3d use a 7+ GiB im2col buffer and OOMs the H3 VAE encode.
- After an OOM the CUDA context can die (illegal memory access). Treat OOM as fatal for the process; prevent it.
- `cudaMallocAsync` is the allocator in the ComfyUI process; ComfyUI's OOM retry repeats the same call.
- T4 limits: int8 GEMM reaches about 30-40 TOPS, fp16 about 21-29 TFLOPS, memory-efficient attention about 12 TFLOPS (flash is not available on sm75), clocks run around 700-800 MHz under the 70 W cap. A step is compute bound; the exchange is cheap.
- Host RAM is about 30 GB; most of what looks used is page cache. Do not pin more than about 10 GB.
- ComfyUI's context windows (`MiniMaxH3ContextWindows`-style nodes) work with the TP path and cap tokens per forward.
- Never let a fallback silently ignore a patch (ControlNet, model patches): raise or run on one GPU and log it.

## 6. The port loop (what `portkit/tp_port.py` automates)

1. `probe`: a Kaggle kernel loads the model through ComfyUI and writes `PROBE_REPORT.md` + `probe.json` + the source text of the block and forward functions.
2. `prompt`: builds `task_port_<name>.py` (a `kaggle_benchmarks` task with one self-contained prompt: this document, the closest reference module, the probe data, the required output). The Kaggle Claude has no internet and no memory between calls, so the prompt must contain everything.
3. Put the returned code into `<family>_tensor_parallel.py`, add the registry line in `tp_auto.py`.
4. `verify`: a Kaggle kernel compares TP against the single-GPU block (rel error, finite, memory, step time, exchange-off time, release test) and prints PASS/FAIL with the numbers.
5. If it fails or is slow: `repair` builds a new self-contained prompt from the previous code and the verify numbers (see `docs/REPAIR.md`).

## 7. Acceptance criteria

- Per-block output relative error vs the single-GPU block <= the error between two int8 runs of the same block (calibrate it first; typical about 1e-2) and no NaN/Inf over 3 seeds.
- Full model forward on a fixed latent is finite and its output correlates >= 0.99 with the single-GPU forward.
- Peak VRAM at the target token count leaves >= 1 GiB per GPU, or the report says exactly where it fails.
- Step time with TP <= 0.65 x the single-GPU step estimate, and the exchange-off run shows the exchange costs < 15% of a step.
- `release(drop_shards=True)` returns allocated memory on both GPUs to < 3 GiB (test: `memory_allocated` before and after).
- `supports(dm)` is False for other models and attach failures fall back to one GPU with a log line.
- A golden file (`docs/golden/<family>.json`) with the accepted numbers is committed.
