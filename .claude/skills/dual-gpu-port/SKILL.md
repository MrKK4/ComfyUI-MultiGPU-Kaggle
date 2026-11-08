---
name: dual-gpu-port
description: Port a new diffusion model to two-GPU tensor parallel on Kaggle 2x T4 in the ComfyUI-MultiGPU fork. Use when a new model or workflow needs dual GPU, when a tensor-parallel port is wrong or slow, or when asked how the dual-GPU setup works. Runs the probe -> prompt -> implement -> verify -> repair loop with the Kaggle Claude (text only, no internet, no memory between calls).
---

# Dual-GPU port loop

Read `docs/PORTING.md` first (contract, recipe, feasibility rules, environment facts). Read `docs/REPAIR.md` when something is wrong or slow. Work on branch `tp-modular`. Never put tokens or keys in files; Kaggle tokens come from env vars. Ask before pushing tags or changing the notebooks.

## Steps

1. **Probe (feasibility).** `python portkit/tp_port.py probe --name <n> --model <local file or HF resolve URL>`. It reads only the safetensors header (no GPU). Read `portkit_runs/<n>/probe/PROBE_REPORT.md`:
   - Any `FAIL` on R1, R6 (mixed formats), R7 means the model needs a different plan: R1 FAIL = blocks differ (for example bf16 edge blocks): requantize them or use a uniform file; R7 FAIL = too big for 2 x 15 GB.
   - `UNKNOWN` rules (attention path, block replaceability, token mixing) need the deep probe on Kaggle (`probe --deep` inside a kernel with ComfyUI installed, see "Kaggle runs"), which also dumps the model source into `block_source.txt`.
2. **Prompt.** `python portkit/tp_port.py prompt --name <n> --reference <h3|qwen21|krea2|ltx>`: pick the reference whose architecture is closest (MiniMax H3 / Qwen-Image 2.1 / Krea 2 / LTX). Output: `portkit_runs/<n>/task_port_<n>.py`. Run that file in a Kaggle notebook with Benchmarks access; the answer is the `solution` field. Save it as `portkit_runs/<n>/attempt_1/answer.md`.
3. **Implement.** Copy the CODE block to `<n>_tensor_parallel.py`, add the registration lines (import + `MODULES` entry in `tp_auto.py`) from the answer.
4. **Verify (Kaggle).** Run the verify kernel (see below). It reports PASS/FAIL for each criterion of PORTING.md section 7 and prints the numbers.
5. **Repair.** If anything fails: `python portkit/tp_port.py repair --name <n> --reference <r> --code <module> --verify-report <VERIFY_REPORT.md> --problem <slow|wrong|oom|crash> --symptom "<fact with numbers>" --attempt 2`. Run the new task file, replace the code with the CODE block, verify again. After 3 attempts on the same criterion, reframe (REPAIR.md) or accept one-GPU fallback.

## Rules of thumb (from measurements, do not rediscover)

- A step on 2 x T4 is compute bound; the host-staged exchange costs about 3% of a step. Do not spend prompts on exchange speed unless the exchange-off run shows more than 5-10%.
- Peak memory, not speed, is what breaks 40k-token jobs. Resident shards are about half the int8 file per GPU.
- `release()` must pop every cache holding shard tensors; verify with `torch.cuda.memory_allocated` before/after.
- Start ComfyUI with the cuDNN preload when fp16 convs fail (the Comfy Remote Start cell does it); never disable cuDNN for VAE work.
- An OOM can kill the CUDA context. Prevent, do not catch.
- Unsupported patches (ControlNet, model patches) must raise or fall back to one GPU with a log line, never be ignored.

## Kaggle runs

Kernels are built by `E:/Local WebUI/Comfy Remote/tests/h3_upscale/build_test.py` (modes `m0`..`m3`, `wfa`, `wfb` are examples: they set the Models list, the node packs, the harness and the run script) and pushed with the `kaggle` CLI using a token in the `KAGGLE_API_TOKEN` environment variable. Two accounts exist (jhguyf6cyc, safd7sd4); the second GPU quota is spent separately. Read results with `kaggle kernels output <owner>/<slug> -p <dir> --file-pattern 'report\.txt'` (never download whole outputs). The harness (`common_pre.py`, `common_main.py` in the same folder) already logs per-second GPU memory and utilisation, per-node and per-step timings, `dmon` clocks and `/proc/meminfo`.

## Golden numbers

`docs/golden/<family>.json` holds accepted numbers per family. A new port must match or beat them or explain why not.
