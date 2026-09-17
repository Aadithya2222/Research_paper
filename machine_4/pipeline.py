"""One-command pipeline: run the whole study end to end.

    python pipeline.py --check          # environment and data readiness only
    python pipeline.py --pilot          # 4 tasks, all conditions, then report
    python pipeline.py --full           # every task, all conditions, then report
    python pipeline.py --analyse-only   # re-run analysis on existing raw data

Stages
    0  environment      python + packages + Ollama + models
    1  validation       config fairness + task integrity
    2  smoke            one task through every condition
    3  experiment       A -> C -> D -> B  (phase order is enforced)
    4  evaluation       grade answers, score roles, classify failures
    5  statistics       paired tests, overhead decomposition, mechanism
    6  figures/tables   regenerate everything from raw data
    7  report           the PDF comparison of A / B / C / D

Each stage stops the pipeline on failure rather than continuing with
data that will not support the analysis. Use --skip to bypass a stage you
have already completed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
PY = sys.executable

BAR = "=" * 70


def _run(cmd: List[str], label: str) -> bool:
    print(f"\n{BAR}\n  {label}\n{BAR}")
    print("$ " + " ".join(cmd[1:] if cmd[0] == PY else cmd) + "\n")
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - started
    ok = result.returncode == 0
    print(f"\n  -> {'OK' if ok else 'FAILED (exit ' + str(result.returncode) + ')'}"
          f"   [{elapsed:.1f}s]")
    return ok


def stage_env() -> bool:
    return _run([PY, "reproduce.py"], "STAGE 0  Environment check")


def stage_validate(task_set: str) -> bool:
    """Validation is advisory here: warnings should not stop a pilot."""
    tasks = ROOT / "tasks" / "tasks_300.json"
    corpus = ROOT / "tasks" / "corpus.json"
    if tasks.exists() and corpus.exists():
        _run([PY, "validate_benchmark.py", "--tasks", str(tasks)],
             "STAGE 1  Benchmark validation")
    else:
        print(f"\n{BAR}\n  STAGE 1  Benchmark validation\n{BAR}")
        print("  Skipped: tasks_300.json or corpus.json not present.")
        print("  Using the starter task set instead.")
    return True


def stage_smoke() -> bool:
    return _run([PY, "smoke_test.py"], "STAGE 2  Smoke test")


def stage_experiment(mode: str, task_set: str, limit: Optional[int],
                     tier: Optional[int],
                     max_hours: Optional[float] = None) -> bool:
    if mode == "pilot":
        cmd = [PY, "run_task.py", "--first", str(limit or 4),
               "--task-set", task_set, "--save"]
        if tier:
            cmd += ["--tier", str(tier)]
        return _run(cmd, f"STAGE 3  Pilot: {limit or 4} tasks x 4 conditions")

    cmd = [PY, "run_experiment.py", "--all", "--task-set", task_set]
    if limit:
        cmd += ["--limit", str(limit)]
    if tier:
        cmd += ["--complexity-tier", str(tier)]
    if max_hours:
        cmd += ["--max-hours", str(max_hours)]
    return _run(cmd, "STAGE 3  Main experiment (A -> C -> D -> B)")


def stage_evaluate(task_set: str) -> bool:
    return _run([PY, "evaluate.py", "--task-set", task_set],
                "STAGE 4  Evaluation")


def stage_stats() -> bool:
    return _run([PY, "analyze.py"], "STAGE 5  Statistics and mechanism analysis")


def stage_figures() -> bool:
    return _run([PY, "make_figures.py"], "STAGE 6  Figures and tables")


def stage_report() -> bool:
    return _run([PY, "make_report.py"], "STAGE 7  PDF report")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the whole study")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="environment and validation only")
    mode.add_argument("--pilot", action="store_true",
                      help="small run (default 4 tasks) then full report")
    mode.add_argument("--full", action="store_true",
                      help="every task, all conditions, then full report")
    mode.add_argument("--analyse-only", action="store_true",
                      help="re-run evaluation, stats, figures and report")

    ap.add_argument("--task-set", default="tasks/tasks.json")
    ap.add_argument("--limit", type=int, help="number of tasks")
    ap.add_argument("--tier", type=int, choices=[1, 2, 3])
    ap.add_argument("--max-hours", type=float,
                    help="Stop the experiment stage after N hours; rerun to "
                         "continue. Use for short batches on a hot laptop.")
    ap.add_argument("--skip", default="",
                    help="comma-separated stages to skip, e.g. env,smoke")
    args = ap.parse_args()

    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    print(BAR)
    print("  COORDINATION TAX BENCHMARK -- FULL PIPELINE")
    print(BAR)
    mode_name = ("check" if args.check else "pilot" if args.pilot
                 else "full" if args.full else "analyse-only")
    print(f"  mode: {mode_name}")
    print(f"  tasks: {args.task_set}"
          + (f"  (first {args.limit})" if args.limit else "")
          + (f"  tier {args.tier}" if args.tier else ""))

    t0 = time.time()

    if "env" not in skip and not stage_env():
        print("\nEnvironment check failed. Fix the missing items above.")
        return 1

    if args.check:
        stage_validate(args.task_set)
        print(f"\n{BAR}\n  CHECK COMPLETE  [{time.time()-t0:.1f}s]\n{BAR}")
        print("  Next:  python pipeline.py --pilot")
        return 0

    if not args.analyse_only:
        if "validate" not in skip:
            stage_validate(args.task_set)
        if "smoke" not in skip and not stage_smoke():
            print("\nSmoke test failed. Do not run the full experiment yet.")
            return 1
        if "experiment" not in skip:
            run_mode = "pilot" if args.pilot else "full"
            if not stage_experiment(run_mode, args.task_set, args.limit,
                                    args.tier, args.max_hours):
                print("\nExperiment stage failed. Inspect results/run.log.")
                return 1

    if "evaluate" not in skip and not stage_evaluate(args.task_set):
        print("\nEvaluation failed. Check raw data integrity.")
        return 1
    if "stats" not in skip and not stage_stats():
        print("\nStatistics failed.")
        return 1
    if "figures" not in skip:
        stage_figures()
    if "report" not in skip and not stage_report():
        print("\nReport generation failed.")
        return 1

    elapsed = time.time() - t0
    print(f"\n{BAR}")
    print(f"  PIPELINE COMPLETE  [{elapsed/60:.1f} min]")
    print(BAR)
    print("\n  results/reports/    the PDF comparison of A / B / C / D")
    print("  results/tables/     LaTeX tables for the manuscript")
    print("  results/figures/    PNG, PDF and SVG figures")
    print("  results/raw/        every individual run")
    print("\n  Remember: a null or negative compute-matched difference is a")
    print("  valid result. Report what you measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
