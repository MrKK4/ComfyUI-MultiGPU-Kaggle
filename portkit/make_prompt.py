"""Builds task_port_<name>.py: a kaggle_benchmarks task whose single prompt is self-contained (the Kaggle Claude has no
internet and no memory between calls): porting contract + closest reference module + probe data + output format.

usage: python make_prompt.py --name mymodel --reference h3 --probe-dir <dir with PROBE_REPORT.md [block_source.txt]> [--out <dir>] [--budget 120000] [--notes "text"]
"""
import argparse
import ast
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCES = {"h3": "h3_tensor_parallel.py", "qwen21": "qwen21_tensor_parallel.py", "krea2": "krea2_tensor_parallel.py", "ltx": "ltx_tensor_parallel.py"}
CHARS_PER_TOKEN = 2.0   # conservative vs measured Idol port/repair: about 2.3 chars per input token

ROLE = """You are a senior PyTorch / CUDA engineer who writes tensor-parallel inference code for ComfyUI diffusion transformers.
You have no internet access and no memory of earlier conversations: everything you need is in this message. Answer from this message and your own knowledge.
Do not ask questions; where something is unknown, state the assumption in the ASSUMPTIONS section and write the code so it fails loudly (raise) if the assumption is false."""

OUTPUT = """# REQUIRED OUTPUT (exactly these sections, in this order)
1. PLAN: at most 15 lines: how each linear is split (column / row parallel), what is replicated, where partial sums are exchanged, how attention heads are divided, how rope / modulation / extra streams are handled.
2. CODE: the complete file `{name}_tensor_parallel.py` in ONE ```python block. Follow the module contract and copy the structure of the reference module. No placeholders, no "..." and no omitted functions.
3. REGISTRATION: the exact lines to add to `tp_auto.py` (the MODULES tuple and the import).
4. ASSUMPTIONS TO VERIFY: a bullet list. For each, say which log line or measurement would show it is wrong.
5. EXPECTED LOG LINES: the module must log `[TP-CHECK] block0 rel_err=<float> finite=<bool>` once, comparing the first tensor-parallel block with the original single-GPU block on the same input, and `[TP] sharded <n> blocks`.
Keep to the contract: `supports(dm)`, `attach(model, source, key)`, `_INSTANCES`, `release(drop_shards=False)` that frees GPU memory completely (pop every cache that holds shard tensors), plain LoRA side paths only (raise on anything else), no silent fallbacks."""


def read(p):
    return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")


def modelweights_excerpt():
    src = read(ROOT / "tp_source.py")
    tree = ast.parse(src)
    lines = src.splitlines()
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "ModelWeights":
            return "\n".join(lines[n.lineno - 1:n.end_lineno])[:9000]
    return src[:6000]


def tpauto_excerpt():
    src = read(ROOT / "tp_auto.py")
    m = re.search(r"MODULES = \(.*?\)\n", src, re.S)
    a = re.search(r"def attach\(patcher\):.*?\n\n\n", src, re.S)
    return (m.group(0) if m else "") + "\n" + (a.group(0) if a else "")


def trim(text, limit):
    return text if len(text) <= limit else text[:limit] + "\n[... trimmed to fit the prompt budget ...]\n"


def build(name, reference, probe_dir, budget, notes):
    ref = read(ROOT / REFERENCES[reference])
    docs = read(ROOT / "docs" / "PORTING.md")
    report = read(pathlib.Path(probe_dir) / "PROBE_REPORT.md")
    bs = pathlib.Path(probe_dir) / "block_source.txt"
    blocksrc = read(bs) if bs.exists() else "(no source text: the probe was run without --deep; reason from the report and your knowledge of the architecture)"
    probe_json = pathlib.Path(probe_dir) / "probe.json"
    parts = {
        "ROLE": ROLE,
        "TASK": "Port the diffusion transformer described under PROBE DATA to two-GPU tensor parallel (Kaggle, 2x Tesla T4, no peer-to-peer) by writing `%s_tensor_parallel.py`, "
                "modelled on the reference module `%s`. %s" % (name, REFERENCES[reference], ("Extra notes from the user: " + notes) if notes else ""),
        "PORTING GUIDE": docs,
        "HOW MODULES ARE REGISTERED (tp_auto.py excerpt)": tpauto_excerpt(),
        "WEIGHT SOURCE INTERFACE (tp_source.ModelWeights)": modelweights_excerpt(),
        "REFERENCE MODULE: " + REFERENCES[reference]: ref,
        "PROBE DATA": report,
        "STRUCTURED PROBE DATA (probe.json)": read(probe_json) if probe_json.exists() else "(not available)",
        "MODEL SOURCE CODE (from ComfyUI)": blocksrc,
        "OUTPUT": OUTPUT.format(name=name),
    }
    limit = int(budget * CHARS_PER_TOKEN)
    total = sum(len(v) for v in parts.values())
    warnings = []
    if total > limit:   # trim the model source first, then the reference, then the guide
        for key, share in (("MODEL SOURCE CODE (from ComfyUI)", 0.25), ("REFERENCE MODULE: " + REFERENCES[reference], 0.45), ("PORTING GUIDE", 0.3)):
            over = sum(len(v) for v in parts.values()) - limit
            if over <= 0:
                break
            parts[key] = trim(parts[key], max(4000, int(len(parts[key]) - over)))
            warnings.append("trimmed %s" % key)
    text = "\n\n".join("=" * 78 + "\n# " + k + "\n" + "=" * 78 + "\n" + v for k, v in parts.items())
    return text, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", default="h3", choices=sorted(REFERENCES))
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--budget", type=int, default=120000, help="prompt budget in tokens (the Kaggle Claude accepts 200k in, 64k out)")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    text, warnings = build(a.name, a.reference, a.probe_dir, a.budget, a.notes)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = re.sub(r"\W+", "_", a.name)
    (out / ("port_%s_prompt.txt" % fn)).write_text(text, encoding="utf-8")
    task = ("import kaggle_benchmarks as kbench\n\nPROMPT = %r\n\n@kbench.task(name=%r)\ndef port_%s(llm) -> dict:\n    response = llm.prompt(PROMPT)\n    return {\"solution\": response}\n\nport_%s.run(kbench.llm)\n"
            % (text, "tp-port-" + a.name.replace("_", "-"), fn, fn))
    compile(task, "task", "exec")
    (out / ("task_port_%s.py" % fn)).write_text(task, encoding="utf-8")
    print("wrote task_port_%s.py: %d chars, about %d tokens%s" % (fn, len(text), len(text) / CHARS_PER_TOKEN, (" | " + ", ".join(warnings)) if warnings else ""))


if __name__ == "__main__":
    main()
