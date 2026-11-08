"""One entry point for the dual-GPU port loop (docs/PORTING.md section 6, docs/REPAIR.md).

  python portkit/tp_port.py probe  --name N --model <path|URL>              header-only feasibility check (no GPU, no packages)
  python portkit/tp_port.py prompt --name N --reference h3|qwen21|krea2|ltx  builds portkit_runs/N/task_port_N.py for the Kaggle Claude
  python portkit/tp_port.py repair --name N --reference R --code F --verify-report V --problem slow|wrong|oom|crash|other --symptom "..." [--attempt K]
  python portkit/tp_port.py runs                                            lists portkit_runs/
Everything is written under portkit_runs/<name>/ (kept in git ignore): probe/, task_port_*.py, attempt_<k>/.
"""
import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parent / "portkit_runs"


def run(script, *args):
    r = subprocess.run([sys.executable, "-I", str(HERE / script), *map(str, args)])
    if r.returncode:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--name", required=True); p.add_argument("--model", required=True)
    p.add_argument("--deep", action="store_true"); p.add_argument("--comfy"); p.add_argument("--tokens", default="18900")
    p = sub.add_parser("prompt"); p.add_argument("--name", required=True); p.add_argument("--reference", default="h3")
    p.add_argument("--probe-dir"); p.add_argument("--notes", default=""); p.add_argument("--budget", default="120000")
    p = sub.add_parser("repair"); p.add_argument("--name", required=True); p.add_argument("--reference", default="h3")
    p.add_argument("--code", required=True); p.add_argument("--verify-report", required=True)
    p.add_argument("--problem", default="other"); p.add_argument("--symptom", required=True); p.add_argument("--attempt", default="2")
    p.add_argument("--probe-dir"); p.add_argument("--extra-log"); p.add_argument("--budget", default="120000")
    sub.add_parser("runs")
    a = ap.parse_args()
    if a.cmd == "runs":
        for d in sorted(RUNS.glob("*")) if RUNS.exists() else []:
            print(d.name, sorted(x.name for x in d.iterdir()))
        return
    base = RUNS / a.name
    if a.cmd == "probe":
        extra = ["--deep", "--comfy", a.comfy] if a.deep else []
        run("probe_model.py", "--model", a.model, "--out", base / "probe", "--tokens", a.tokens, *extra)
    elif a.cmd == "prompt":
        run("make_prompt.py", "--name", a.name, "--reference", a.reference, "--probe-dir", a.probe_dir or base / "probe", "--out", base,
            "--notes", a.notes, "--budget", a.budget)
    else:
        extra = (["--probe-dir", a.probe_dir] if a.probe_dir else (["--probe-dir", str(base / "probe")] if (base / "probe").exists() else [])) + (["--extra-log", a.extra_log] if a.extra_log else [])
        run("make_repair.py", "--name", a.name, "--reference", a.reference, "--code", a.code, "--verify-report", a.verify_report, "--problem", a.problem,
            "--symptom", a.symptom, "--attempt", a.attempt, "--out", base / ("attempt_" + a.attempt), "--budget", a.budget, *extra)


if __name__ == "__main__":
    main()
