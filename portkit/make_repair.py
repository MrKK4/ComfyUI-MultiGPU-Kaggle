"""Builds task_repair_<name>_<n>.py: a follow-up prompt for a Kaggle Claude that remembers nothing. It restates the task,
includes the module it wrote and our measurements, and asks for a targeted change, so it reads as a continuation.

usage: python make_repair.py --name mymodel --reference h3 --code <module file as it is now> --verify-report <VERIFY_REPORT.md>
          --problem slow|wrong|oom|crash|other --symptom "what we saw" [--probe-dir <dir>] [--attempt 2] [--extra-log <file>] [--out <dir>] [--budget 120000]
"""
import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_prompt import CHARS_PER_TOKEN, REFERENCES, ROLE, ROOT, read, trim  # noqa: E402

GUIDE = {
    "slow": "The module runs and is numerically correct but a step is slower than the compute-bound estimate. Decide, from the numbers, which of these dominates: (a) the exchange of partial sums (compare the exchange-off step time), "
            "(b) extra work duplicated on both ranks (norms, modulation, rotation, quantization of activations), (c) small or badly shaped GEMMs after the split (alignment, contiguity, int8 kernel really used), "
            "(d) synchronization (events waited too early, copies on the compute stream), (e) memory pressure causing allocator stalls. Change only what the numbers point at.",
    "wrong": "The output differs from the single-GPU block. Check in this order: head/row ordering of the fused qkv split, gate/up halves of a gated MLP, rope applied with global (not rank-local) positions, "
             "the 1/S scale or convrot group alignment on the row-parallel inputs, partial sums added twice or not at all, modulation segments sliced for chunks, dtype of the residual. Name the first block output that diverges.",
    "oom": "The module runs out of memory. Compute the memory budget per GPU from the report: resident shards, LoRA side paths, per-forward buffers (qkv output, K/V, partial sums, full-sequence receive buffers), optional prefetch. "
           "Remove the largest avoidable buffer first (free it earlier, chunk over tokens, bound the receive buffers). Do not rely on empty_cache after a failed allocation: the CUDA context can die.",
    "crash": "The module raises or hangs. Read the traceback in the log first; say what state the module was in. Typical causes: a device mismatch between ranks, an event or stream used across devices, "
             "a missing key in the source (prefix, .comfy_quant metadata), a patch type the module does not support.",
    "other": "Treat the symptom text as the problem statement and use the measurements to confirm it.",
}

CONTINUATION = """This is a follow-up to a task you handled in an earlier conversation (attempt {prev} of this port). You do not remember it, so everything you need is below: the task, the module you wrote, and what we measured when we ran it on the real hardware.
Do not start from scratch unless the numbers show a design flaw: keep the module contract and file layout, change the minimum that fixes the problem, and say what changed and why."""

OUTPUT = """# REQUIRED OUTPUT (exactly these sections)
1. DIAGNOSIS: what the measurements say, as a ranked list of hypotheses with the number that supports or contradicts each.
2. CHANGES: a list of the edits (function name, what changes, expected effect and size).
3. CODE: the complete corrected file in ONE ```python block (no omissions).
4. NEXT MEASUREMENT: the one log line or timing that would show whether the change worked, and the value you expect.
5. IF THIS DOES NOT WORK: the next hypothesis you would test."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", default="h3", choices=sorted(REFERENCES))
    ap.add_argument("--code", required=True)
    ap.add_argument("--verify-report", required=True)
    ap.add_argument("--problem", default="other", choices=sorted(GUIDE))
    ap.add_argument("--symptom", required=True)
    ap.add_argument("--probe-dir")
    ap.add_argument("--attempt", type=int, default=2)
    ap.add_argument("--extra-log")
    ap.add_argument("--out", default=".")
    ap.add_argument("--budget", type=int, default=120000)
    a = ap.parse_args()
    limit = int(a.budget * CHARS_PER_TOKEN)
    parts = {
        "ROLE": ROLE,
        "CONTEXT": CONTINUATION.format(prev=a.attempt - 1),
        "THE TASK AS IT WAS ASKED": read(ROOT / "docs" / "PORTING.md"),
        "THE MODULE YOU WROTE (attempt %d), as we ran it" % (a.attempt - 1): read(a.code),
        "WHAT WE MEASURED (verify report from Kaggle, 2x T4)": read(a.verify_report),
    }
    if a.probe_dir and (pathlib.Path(a.probe_dir) / "PROBE_REPORT.md").exists():
        parts["MODEL FACTS (probe report)"] = read(pathlib.Path(a.probe_dir) / "PROBE_REPORT.md")
    if a.extra_log:
        parts["EXTRA LOG LINES"] = trim(read(a.extra_log), 20000)
    parts["THE PROBLEM"] = "Class: %s.\nSymptom: %s\n\n%s" % (a.problem, a.symptom, GUIDE[a.problem])
    parts["REFERENCE MODULE (for structure only, it is the tested one): " + REFERENCES[a.reference]] = read(ROOT / REFERENCES[a.reference])
    parts["OUTPUT"] = OUTPUT
    warnings = []
    over = sum(len(v) for v in parts.values()) - limit
    if over > 0:
        key = "REFERENCE MODULE (for structure only, it is the tested one): " + REFERENCES[a.reference]
        parts[key] = trim(parts[key], max(3000, len(parts[key]) - over)); warnings.append("trimmed the reference module")
    text = "\n\n".join("=" * 78 + "\n# " + k + "\n" + "=" * 78 + "\n" + v for k, v in parts.items())
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fn = "%s_%d" % (re.sub(r"\W+", "_", a.name), a.attempt)
    (out / ("repair_%s_prompt.txt" % fn)).write_text(text, encoding="utf-8")
    task = ("import kaggle_benchmarks as kbench\n\nPROMPT = %r\n\n@kbench.task(name=%r)\ndef repair_%s(llm) -> dict:\n    response = llm.prompt(PROMPT)\n    return {\"solution\": response}\n\nrepair_%s.run(kbench.llm)\n"
            % (text, "tp-repair-%s-%d" % (a.name.replace("_", "-"), a.attempt), fn, fn))
    compile(task, "task", "exec")
    (out / ("task_repair_%s.py" % fn)).write_text(task, encoding="utf-8")
    print("wrote task_repair_%s.py: %d chars, about %d tokens%s" % (fn, len(text), len(text) / CHARS_PER_TOKEN, (" | " + ", ".join(warnings)) if warnings else ""))


if __name__ == "__main__":
    main()
