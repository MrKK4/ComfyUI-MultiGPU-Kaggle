# Porting kit status — 2026-11-09

The B/C/D correctness and recovery trial passed at the direct-denoiser scope below. This does not certify all acceptance criteria in PORTING.md or a production Idol workflow.

## Evidence

- H3 corrected verify report (safd7sd4/h3-kit-verify, script version 367055450): PASS; exchange on 23.0 s, off 22.1 s, approximately 4% overhead. Golden results were refreshed and ea9f5ba was pushed.
- Idol is a new family for this fork. Its light probe and real GPU deep probe completed; 24 uniform blocks, source capture, and linear/attention benchmarks were obtained in safd7sd4/idol-kit-probe.
- The task included the actual deep report, probe.json and block_source.txt. Opus 5.5 generated the module from that prompt. Recorded input use: 30,879 tokens, below 120k.
- [Initial GPU verification](trials/idol_good_verify.md): PASS on three synthetic-input seeds, full-forward cosine about 0.99984, block0 relative error 0.00183 and zero replica drift; memory, release and reload passed.
- [Deliberately broken module](trials/idol_broken_verify.md): FAIL. Appending XCHG_OFF = True changed full-forward cosine to approximately -0.45. This is a real GPU fault injection, not a measurement replay.
- The repair builder carried the original model-specific task, the exact broken code, real failure report/log and deep-probe evidence into a fresh Opus call (73,293 recorded input tokens, below 120k).
- [Repaired module GPU verification](trials/idol_repaired_verify.md): PASS after one generated repair. The returned code was tested without local implementation edits.

## Intentional fallback and remaining limits

Idol warm TP forwards were approximately 1.02 s versus 0.98 s on one GPU in the initial trial. Two warm samples per path establish a smoke comparison, not a robust speed benchmark. Idol remains deliberately unregistered in tp_auto.MODULES, so production keeps the one-GPU fallback. The root idol_tensor_parallel.py is the tested generated example, not enabled production support.

The trial uses synthetic latent/camera/context inputs and does not exercise image encoders, the Gaussian decoder, sampler hooks, optional reference latents, LoRA, ControlNet, or real production workflows. See golden/idol.json for measured data and scope. Existing documented model-size and architecture limits remain in force.

The repair response was recovered from task tp-repair-idol-2 version 2. The model call returned successfully, but the Kaggle task wrapper rejected hardcoded model selection afterward. The transcript identifies the assistant as anthropic/claude-opus-5-5@default. Keep generated tasks on kbench.llm and select models through the CLI; do not copy that one-off hosting override into the kit.

## Kit fixes verified during this trial

- Reject missing/ambiguous sampler progress rather than substituting VAE progress; accept real sub-second sampler timings.
- Require successful exchange-switch jobs and matching nonempty module logs. Switch only modules with active instances.
- Keep Comfy Remote's kv_head.py / kv_run.py copies in sync with the kit verifier.
- Include structured deep-probe data in first prompts, and retain original task instructions and deep source in repairs.
- Use a more conservative token estimate based on the measured port/repair input counts.
- Local tests cover numerical/nonfinite/memory/release/reload failures, missing sampler ticks, failed/empty switches, inactive modules, fast sampler timing and fresh repair context. Local fixture tests are not represented as GPU tests.

## Shipping still open

No merge, tag, notebook generation or MULTIGPU_TAG bump was performed. These remain a user decision. Pre-existing untracked handover/task files and local Backups remain; unrelated user files were not removed or committed.
