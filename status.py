"""Show how far the experiment has got, and what is left.

    python status.py
    python status.py --task-set my_tasks.json

Safe to run at any time, including while the experiment is running in
another window. It only reads files.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.runner import completed_keys, load_existing   # noqa: E402
from benchmark.tasks import load_tasks                        # noqa: E402
from core.config import load_config                           # noqa: E402

CONDITIONS = ["A", "C", "D", "B"]      # execution order
LABEL = {"A": "A single ReAct", "B": "B matched Best-of-N",
         "C": "C pipeline LangGraph", "D": "D pipeline CrewAI"}


def _bar(done: int, total: int, width: int = 24) -> str:
    if total == 0:
        return "." * width
    filled = int(width * done / total)
    return "#" * filled + "." * (width - filled)


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment progress")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task-set", default="my_tasks.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ts = Path(args.task_set)
    if not ts.exists():
        for alt in ("my_tasks.json", "tasks/tasks.json"):
            if Path(alt).exists():
                ts = Path(alt)
                break
    tasks = load_tasks(ts, cfg)
    runs = int(cfg.get("experiment.runs_per_task", 1))
    total_per_cond = len(tasks) * runs

    print("=" * 66)
    print("  EXPERIMENT PROGRESS")
    print("=" * 66)
    machine = cfg.get("machine.id")
    if machine:
        print(f"  machine {machine}   ", end="")
    print(f"tasks {len(tasks)}   runs each {runs}   "
          f"config {cfg.config_hash}")
    print()

    grand_done = 0
    durations: List[float] = []
    all_complete = True

    for cond in CONDITIONS:
        done_keys = completed_keys(cfg, cond)
        done = len(done_keys)
        grand_done += done
        remaining = total_per_cond - done
        if remaining > 0:
            all_complete = False

        recs = load_existing(cfg, cond)
        lat = [r["latency_seconds"] for r in recs
               if r.get("latency_seconds") is not None]
        durations += lat
        ok = sum(1 for r in recs if r.get("success_execution"))
        fails = Counter(r.get("failure_type") for r in recs
                        if r.get("failure_type"))

        mean_lat = sum(lat) / len(lat) if lat else 0.0
        eta_h = remaining * mean_lat / 3600 if mean_lat else 0.0

        pct = 100 * done / total_per_cond if total_per_cond else 0
        print(f"  {LABEL[cond]:<24} [{_bar(done, total_per_cond)}] "
              f"{done:>4}/{total_per_cond:<4} {pct:5.1f}%")
        if done:
            print(f"    answered {ok}/{done}   mean {mean_lat:6.1f}s/task"
                  + (f"   est. remaining {eta_h:.1f}h" if remaining else
                     "   COMPLETE"))
            if fails:
                top = ", ".join(f"{k}={v}" for k, v in fails.most_common(3))
                print(f"    failures: {top}")
        print()

    grand_total = total_per_cond * len(CONDITIONS)
    pct = 100 * grand_done / grand_total if grand_total else 0
    mean_all = sum(durations) / len(durations) if durations else 0.0
    eta_all = (grand_total - grand_done) * mean_all / 3600 if mean_all else 0.0

    print("-" * 66)
    print(f"  OVERALL  [{_bar(grand_done, grand_total, 30)}] "
          f"{grand_done}/{grand_total}  {pct:.1f}%")
    if mean_all:
        print(f"  mean {mean_all:.1f}s per task-condition")
        if grand_done < grand_total:
            print(f"  estimated time remaining: {eta_all:.1f} hours")
    print()

    # --- what to do next -------------------------------------------------
    d_done = len(completed_keys(cfg, "D"))
    b_done = len(completed_keys(cfg, "B"))

    if all_complete:
        print("  All conditions complete.")
        print("  Next:  python run_experiment.py --all "
              "--task-set calibration_tasks.json")
        print("  Then:  python send_back.py")
    else:
        nxt = next((c for c in CONDITIONS
                    if len(completed_keys(cfg, c)) < total_per_cond), None)
        if nxt == "B" and d_done < total_per_cond:
            print("  Condition B cannot start: Condition D is not finished.")
            print(f"  D has {total_per_cond - d_done} task(s) left.")
        print(f"  Next up: condition {nxt}")
        print()
        print("  Continue where it stopped (nothing is repeated):")
        print(f"      python run_experiment.py --all --task-set {ts}")
        print()
        print("  Or run a 2-hour batch and stop cleanly:")
        print(f"      python run_experiment.py --all --task-set {ts} "
              f"--max-hours 2")

    if b_done and d_done < total_per_cond:
        print()
        print("  WARNING: Condition B has records while Condition D is")
        print("  incomplete. B's budget was matched to a partial D. Those B")
        print("  records should be deleted and rerun once D has finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
