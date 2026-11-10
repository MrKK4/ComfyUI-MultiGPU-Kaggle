# Plan: pre-split rank files + partial eviction (H3 first)

Branch: new branch from `tp-modular` (tag `kaggle-ok-67`). Behind a switch, OFF by default. Read first: `docs/PORTING.md`, `docs/golden/h3.json`, `h3_tensor_parallel.py` (`_SHARD_CACHE`, `release_gpu`, `_INSTANCES`), `tp_auto.py` (CFGGuider.sample hook), `tp_source.py`. Rules: terse chat, no tokens in files, no tags or notebook pin changes without the user's yes, nothing on `C:`, short Kaggle runs, after two failed runs the next one is a diagnostic. Do not touch `notebooks/experimental/`.

## Problem (measured, Kaggle 2x T4, H3 Olivia int8)
- Shards resident: GPU0 11.27 GiB, GPU1 8.99 GiB (of about 14.56 usable). Free with shards resident: about 3.1 / 3.9 GiB.
- Any model needing more VRAM (preprocessor, ControlNet prep, later models) forces `release_gpu(drop_shards=True)`, then a full rebuild: about 75 s of shard load, reload job about 239 s. Cold start pays the same 76 s.
- Idea (user): keep the shards as per-rank files in the final on-GPU layout, in small pieces. Evict only as many pieces from GPU1 as the other model needs; GPU0 keeps its shard (idle meanwhile). Reload only the evicted pieces, with no slicing or packing.

## Design
1. **Piece files.** Per rank, block ranges of about 5 blocks (about 0.9 GiB): `rank{r}_blocks_{a}-{b}.safetensors`, plus `common.safetensors` (unsharded small layers: token refiner, condition_proj, final_layer, patch projections, adaln table) and `manifest.json`. Pieces hold exactly what the runtime keeps on GPU: int8 weights, scales, row order, 256-group alignment, LoRA-independent (LoRA stays a side path).
2. **Manifest** fields: layout_version, exporter_version, source file sha256 and size, tp_degree 2, convrot group size, chunks (`MMH3_TP_CHUNKS` default 4), block ranges per file, per-file sha256 and bytes. The loader refuses mismatches and falls back to the current build-from-original path with one log line.
3. **Partial release:** `release_gpu(drop_shards=True, rank=1, bytes_needed=N)` evicts the fewest pieces (from the last blocks backwards or by a stored order) so that free VRAM >= N + margin; keeps the rest and records which pieces are missing. Full release (today's behaviour) stays when `rank`/`bytes_needed` are not given.
4. **Auto reload:** the CFGGuider.sample hook in `tp_auto.py` checks piece presence before sampling and reloads missing pieces (read -> pinned staging buffer -> H2D, both ranks concurrently), then logs `[TP] reloaded N pieces (X GiB) in S s`. Never start a sampler with a missing piece.
5. **Where evicted pieces live:** piece files on local disk (`/kaggle/temp`, not `/kaggle/working`, to keep output small). Page cache helps if it holds them; do not depend on it. Do not pin 18 GB of host RAM (MemAvailable low point 11.4 GB).
6. **Exporter:** a function in the H3 module (`export_pieces(out_dir)`) that builds shards from the original file once and writes the files; the first load may call it automatically when the switch is on and no valid files exist. Optional later: upload the files to the user's HF repo (needs licence check of each upstream model, private repo + read token via Kaggle secret, short-lived write token; never in files).
7. **Switch:** env `MMH3_TP_PIECES=1` (default 0). Default off keeps every notebook behaving as today.

## Steps
1. **Measure first (one short Kaggle kernel, `tests/h3_upscale` style):** disk read speed with 1 / 4 / 8 threads, cold and warm cache, on files of 0.9 GiB; real peak VRAM of the preprocessors the user will use (depth, pose; ask which); how much of the 75 s shard load is packing vs reading. Decision rule: if reading dominates (>60%), piece size and page cache matter more than packing; if packing dominates, pre-split alone gives most of the win.
2. **Exporter + reader** with the manifest; unit test on CPU (tiny synthetic block set): export -> load -> tensors equal the build-from-original path bit for bit.
3. **Partial release + auto reload** in `h3_tensor_parallel.py` / `tp_auto.py`. CPU-testable parts: piece bookkeeping, fewest-pieces choice, refusal on manifest mismatch.
4. **Verify on Kaggle (kit `kv` style, one run):**
   - cold start with pieces vs without (shard seconds);
   - evict rank-1 pieces to free N GiB, run a dummy 6-9 GiB allocation on GPU1 as the "preprocessor", free it, run sampling: output equals the pre-evict output (same seed; `[TP-CHECK]` block0 rel_err <= 3e-2 and finite), reload seconds logged, peak memory within the 1 GiB margin;
   - full release + reload still works (regression);
   - a sampler start with a deliberately missing piece triggers the auto reload, not a crash.
5. **Golden numbers:** add the measured pieces numbers to `docs/golden/h3.json` (cold start seconds, evict/reload seconds per GiB, pieces count).
6. **Docs:** short section in `docs/PORTING.md`: the module contract gains optional `release_gpu(rank, bytes_needed)` and a piece loader; new ports may skip it. Note the layout-version coupling.
7. **Ask the user** before: tagging (next `kaggle-ok-68`), bumping `MULTIGPU_TAG` in `E:\Local WebUI\Comfy Remote\notebook\build\build.py`, uploading anything to HF, and enabling the switch in a notebook.

## Acceptance
- Output with pieces on is numerically equal (block check + same-seed comparison) to the current path.
- Evict-then-reload of the needed pieces is faster than today's full rebuild by a measured margin; report seconds per GiB.
- No peak above 14.56 - 1.0 GiB in the verify runs; both GPUs return to the pre-evict allocation after reload.
- Switch off = byte-identical behaviour to `kaggle-ok-67`.

## Limits to state in the report
All pieces must be resident while sampling; the gain flattens with finer pieces (cost = bytes / read speed); preprocessor peak includes activations, so size by measurement; files are tied to layout version and 2-way TP; applies to ported models only (each needs its own exporter).

## Safety and quality of life (required, part of the same work)
A. **Pre-flight before every sampler start** (in the `tp_auto.py` hook, one log block `[TP-PREFLIGHT]`):
   1. Piece table per GPU: every piece of every rank present and resident (names, bytes, GPU); manifest bytes match; shapes and dtypes of a sample tensor per piece match the manifest. Hash only at load time (not per step).
   2. Free VRAM per GPU (`torch.cuda.mem_get_info`) >= bytes still to reload + estimated sampling peak. Estimate from tokens using `docs/golden/h3.json` (peak GiB at 18.9k and 40.7k tokens, interpolate, add the 1 GiB margin).
   3. If short: unload other models first (ComfyUI `comfy.model_management.unload_all_models()` / `free_memory`, then `gc.collect()`, `torch.cuda.empty_cache()`), re-measure; if still short, stop with a clear error naming GPU, needed GiB, free GiB and the suggestion (context windows, fewer frames, smaller size). Never start a sampler that will OOM: an OOM can kill the CUDA context.
B. **Order of operations (no OOM on reload):** free the other model's memory on the target GPU and confirm via `mem_get_info` BEFORE any piece is read to the GPU; then reload; then run A.1 again. Reload pieces from both ranks concurrently, but only after A.3 passes for both GPUs.
C. **Atomic eviction and reload:** a lock so eviction never happens while a sampler runs. If a reload fails midway (I/O error, hash mismatch, OOM guard), drop the half-loaded pieces of that rank, fall back to the full build-from-original path once, and if that fails fall back to one GPU with a log line. State is always one of: all resident, partially evicted (known list), fully released.
D. **Visibility:** `[TP-STATE]` line after every evict/reload: per GPU resident GiB, pieces resident/evicted, seconds taken, MB/s. A `[TP-PREFLIGHT]` OK line per sampler start.
E. **Explicit workflow nodes (optional, default behaviour is automatic):** `TPEvict(rank, free_gib)` placed before a heavy preprocessor and `TPEnsureLoaded` placed before the sampler, so a workflow can control timing; both report the same state table. Without them the hook still evicts on demand and reloads before sampling.
F. **Disk safety:** check free space before writing piece files (exporter needs about the model size on `/kaggle/temp`), refuse with a message if short; remove stale piece directories whose manifest hash does not match the current source; never write pieces into `/kaggle/working` (output size).
G. **Round-trip check in the exporter:** after writing, read the pieces back and compare to the in-memory shards (equal bit for bit); write `manifest.json` last, so a crashed export is never mistaken for a valid one.

Add to the Kaggle verify (step 4): preflight blocks a deliberately oversized second-model allocation with the clear error instead of an OOM; reload after a corrupted piece (flip one byte in a copy) is detected and falls back to the build-from-original path; sampling is refused while a rank is partially evicted and the auto reload cannot fit.

Add to acceptance: no sampler ever starts with a missing or corrupted piece; every failure path ends in a log line and a working state (full path or one GPU), never a crash or a silent wrong result.
