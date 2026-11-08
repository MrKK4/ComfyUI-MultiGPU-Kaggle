# Native TRELLIS.2 / Pixal3D dual-GPU attention (experimental)

Base branch: `tp-universal` (c928510). The new node is opt-in and is not part of automatic model TP selection.

Insert `ModelTrellis2DualGPUAttention` after the diffusion loader, before the model goes into CFG/sampling patches.
The input must be a native ComfyUI TRELLIS.2 or Pixal3D model. Set secondary GPU to `cuda:1`, stages to
`shape_1024`, min_tokens to 8192, verify to true. Enable two CUDA GPUs. The existing loader and checkpoint stay
unchanged. The native model must run with standard PyTorch attention, not `--use-ck-attention`.

This splits self/cross-attention by heads. Primary-GPU linear projections, MLPs, sparse coordinates and Pixal3D
camera projection remain unchanged. Input/output head chunks are exchanged through pinned CPU memory so Kaggle
T4s do not need P2P. Both GPUs compute attention concurrently through asynchronous CUDA launches. FP32/BF16/FP16
compute choice stays exactly as selected by ComfyUI; this does not enable FP16 for models that exclude it.

Default scope is the second, high-resolution shape pass only. `shape_and_texture` also enables the texture pass;
`all` includes the earlier stages. Short attention sequences, attention patches, or an insufficient secondary
VRAM budget use the original attention implementation and log relevant fallbacks. Unexpected CUDA errors are
not swallowed. The standard nodes/workflows remain unchanged unless this model node is inserted and enabled.

On first use for each self/cross-attention layout, verification compares 128 query rows (all keys) against the
original native attention. A mismatch stops the run. Passing that check covers the attention output, not final
mesh quality. CUDA transfer correctness, memory pressure and speed need a real two-GPU run; no speedup has
been measured yet. Attention-only parallelism cannot provide full-model tensor-parallel acceleration.

Validation: `python -B ci/test_trellis2_dual_attention.py` runs CPU numerical parity (25 layouts) plus stage,
exception-cleanup, model-clone, disable and CPU-fallback checks. It does not exercise CUDA transfers.

Comfy Remote embeds a copy of `trellis2_dual_attention.py` into its TRELLIS.2/Pixal3D notebooks. Sync that copy
after editing this module; the notebook builder is offline and does not fetch an unpublished branch change.
