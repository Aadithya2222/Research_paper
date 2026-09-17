"""Per-task evaluation: run ONE task through ALL FOUR conditions and
compare them side by side immediately.

    python run_task.py --task Q001
    python run_task.py --task Q001 --parallel
    python run_task.py --tasks Q001,Q002,Q003,Q004
    python run_task.py --first 4

Why this exists
---------------
`run_experiment.py` runs one condition across every task, which is the
right shape for the main experiment but gives you no feedback for hours.
This script runs one task across every condition, so you see A vs B vs C
vs D on the same question within minutes. Use it to debug prompts, sanity
check the compute matching, and inspect what each architecture actually
produced.

Two execution modes
-------------------
sequential (default)
    Conditions run one after another. Only one model request is in flight
    at a time, so latency, TTFT and memory are measured cleanly. THIS IS
    THE MODE FOR ANY NUMBER THAT GOES IN THE PAPER.

parallel (--parallel)
    Conditions run concurrently against the same Ollama server. Faster in
    wall-clock terms, but the conditions contend for CPU, GPU and memory,
    so latency and resource figures are contaminated by that contention.
    Records produced this way are tagged `timing_valid: false` and the
    analysis pipeline excludes them from timing and resource statistics.
    Quality and token counts remain valid, because they do not depend on
    how many requests were in flight.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.runner import run_one, _raw_path                 # noqa: E402
from benchmark.tasks import Task, load_tasks                    # noqa: E402
from core.config import detect_hardware, load_config, setup_logging  # noqa: E402
from core.llm import client_from_config                         # noqa: E402
from tools.local_tools import build_registry                    # noqa: E402
from validation.checks import validate_config                   # noqa: E402

LOG = logging.getLogger("run_task")

ALL_CONDITIONS = ["A", "C", "D", "B"]   # B last: it needs D's budget
LABELS = {
    "A": "A  single ReAct",
    "B": "B  matched Best-of-N",
    "C": "C  pipeline LangGraph",
    "D": "D  pipeline CrewAI",
}


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _budget_from_records(records: Dict[str, Dict[str, Any]],
                         cfg, task_id: str) -> Dict[str, Any]:
    """Build Condition B's budget from D's COST ONLY, in memory.

    Mirrors the two-phase protocol used by the main runner: B is told
    what D spent, never how well D did.
    """
    d = records.get("D")
    a = records.get("A")
    if d is None or d.get("total_tokens") is None:
        return {"target_total_tokens": None, "target_llm_calls": None,
                "median_single_run_tokens": None}
    return {
        "target_total_tokens": float(d["total_tokens"]),
        "target_llm_calls": float(d["llm_calls"]),
        "median_single_run_tokens": (float(a["total_tokens"])
                                     if a and a.get("total_tokens") else None),
    }


def check_matching(rec_b: Dict[str, Any], rec_d: Dict[str, Any],
                   cfg) -> Dict[str, Any]:
    """Verify Eq. (3): is B actually within tolerance of D?

    The main pipeline chooses N from D's budget but does not afterwards
    confirm the realised spend landed inside the tolerance. This reports
    it explicitly, per task, so out-of-tolerance pairs can be excluded
    and the exclusion rate stated in the paper.
    """
    tol = float(cfg.get("compute_matching.token_tolerance_percent", 10.0)) / 100.0
    call_tol = int(cfg.get("compute_matching.call_tolerance", 1))
    tb, td = rec_b.get("total_tokens"), rec_d.get("total_tokens")
    kb, kd = rec_b.get("llm_calls"), rec_d.get("llm_calls")

    if not tb or not td:
        return {"matched": None, "reason": "token counts unavailable"}

    tok_dev = abs(tb - td) / td
    call_dev = abs((kb or 0) - (kd or 0))
    matched = tok_dev <= tol and call_dev <= call_tol
    return {
        "matched": bool(matched),
        "token_deviation": round(tok_dev, 4),
        "token_tolerance": tol,
        "call_deviation": call_dev,
        "call_tolerance": call_tol,
        "tokens_B": tb, "tokens_D": td,
        "calls_B": kb, "calls_D": kd,
        "reason": ("within tolerance" if matched else
                   f"token dev {tok_dev:.1%} vs tol {tol:.0%}, "
                   f"call dev {call_dev} vs tol {call_tol}"),
    }


def run_task_all_conditions(cfg, task: Task, conditions: List[str],
                            parallel: bool, judge_client,
                            hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Run one task under every requested condition."""
    client = client_from_config(cfg)
    registry = build_registry(cfg)
    records: Dict[str, Dict[str, Any]] = {}

    # Condition B always runs last and alone, because it needs D's budget.
    main = [c for c in conditions if c != "B"]
    needs_b = "B" in conditions

    def _run(cond: str, budget: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if cond == "B":
            from agents.conditions import run_condition_b
            from core.llm import Ledger
            from core.resources import ResourceMonitor
            from core.config import make_experiment_id
            from datetime import datetime, timezone

            started = time.perf_counter()
            with ResourceMonitor(float(cfg.get("experiment.resource_sample_hz", 5))) as mon:
                out = run_condition_b(cfg, client, registry, task.to_dict(),
                                      budget or {}, judge_client)
            wall = time.perf_counter() - started
            led = out.ledger.summary(
                include_judge=bool(cfg.get("compute_matching.include_judge_cost", True)))
            rec = {
                "experiment_id": make_experiment_id("B", task.complexity_tier or 0,
                                                    1, task.task_id),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "condition": "B", "task_id": task.task_id,
                "complexity_tier": task.complexity_tier,
                "complexity_score": task.complexity_score,
                "evaluation_type": task.evaluation_type, "run_id": 1,
                "config_hash": cfg.config_hash,
                "model": cfg.require("model.name"),
                "quantization": cfg.get("model.quantization"),
                "temperature": cfg.get("model.temperature"),
                "seed": cfg.get("model.seed"), "num_ctx": cfg.get("model.num_ctx"),
                "input_tokens": led["input_tokens"],
                "output_tokens": led["output_tokens"],
                "total_tokens": led["total_tokens"],
                "llm_calls": led["llm_calls"],
                "agent_llm_calls": led["agent_llm_calls"],
                "judge_llm_calls": led["judge_llm_calls"],
                "tool_calls": led["tool_calls"], "retries": led["retries"],
                "iterations": out.iterations,
                "latency_seconds": round(wall, 4),
                "model_latency_seconds": led["model_latency_seconds"],
                "tool_latency_seconds": led["tool_latency_seconds"],
                "ttft_seconds": led["first_ttft_seconds"],
                **mon.summary().to_dict(),
                "final_answer": out.final_answer,
                "stopped_reason": out.stopped_reason,
                "failure_type": out.failure_type,
                "success_execution": out.final_answer is not None,
                "stage_outputs": out.stage_outputs, "handoffs": out.handoffs,
                "notes": out.notes, "call_ledger": led["call_ledger"],
                "hardware": hardware,
            }
            return rec
        return run_one(cfg, cond, task, 1, client, registry, judge_client, hardware)

    if parallel and len(main) > 1:
        LOG.warning("PARALLEL MODE: conditions contend for CPU/GPU/RAM. "
                    "Latency and resource figures are NOT valid for the paper.")
        with ThreadPoolExecutor(max_workers=len(main)) as pool:
            futures = {pool.submit(_run, c): c for c in main}
            for fut, cond in futures.items():
                try:
                    records[cond] = fut.result()
                except Exception as exc:
                    LOG.error("Condition %s failed: %s", cond, exc)
    else:
        for cond in main:
            LOG.info("running condition %s ...", cond)
            try:
                records[cond] = _run(cond)
            except Exception as exc:
                LOG.error("Condition %s failed: %s", cond, exc)

    if needs_b:
        budget = _budget_from_records(records, cfg, task.task_id)
        if budget["target_total_tokens"] is None:
            LOG.warning("No usable Condition D budget; skipping Condition B. "
                        "Run D successfully first.")
        else:
            LOG.info("running condition B (matched to D: %.0f tokens) ...",
                     budget["target_total_tokens"])
            try:
                records["B"] = _run("B", budget)
            except Exception as exc:
                LOG.error("Condition B failed: %s", exc)

    # Tag timing validity so downstream analysis can exclude contended runs.
    for rec in records.values():
        rec["timing_valid"] = not parallel
        rec["execution_mode"] = "parallel" if parallel else "sequential"

    return records


def print_comparison(task: Task, records: Dict[str, Dict[str, Any]],
                     cfg, parallel: bool) -> None:
    """Side-by-side table for one task."""
    print("\n" + "=" * 78)
    print(f"TASK {task.task_id}   tier {task.complexity_tier}   "
          f"C={task.complexity_score}")
    print("=" * 78)
    text = task.task_text
    print(text if len(text) <= 300 else text[:300] + " ...")
    print(f"\nmode: {'PARALLEL (timing invalid)' if parallel else 'sequential'}")

    hdr = (f"\n{'Condition':<24}{'ok':>4}{'tokens':>9}{'calls':>7}"
           f"{'tools':>7}{'iters':>7}{'latency':>10}{'RAM MB':>9}")
    print(hdr)
    print("-" * 78)
    for cond in ALL_CONDITIONS:
        r = records.get(cond)
        if r is None:
            print(f"{LABELS[cond]:<24}{'--':>4}{'not run':>9}")
            continue
        ok = "yes" if r["success_execution"] else "NO"
        print(f"{LABELS[cond]:<24}{ok:>4}{_fmt(r['total_tokens'],0):>9}"
              f"{_fmt(r['llm_calls'],0):>7}{_fmt(r['tool_calls'],0):>7}"
              f"{_fmt(r['iterations'],0):>7}"
              f"{_fmt(r['latency_seconds'],1)+'s':>10}"
              f"{_fmt(r['peak_ram_mb'],0):>9}")

    # Compute matching check (Eq. 3)
    if "B" in records and "D" in records:
        m = check_matching(records["B"], records["D"], cfg)
        print("\nCOMPUTE MATCHING (Eq. 3)")
        if m.get("matched") is None:
            print(f"  cannot verify: {m['reason']}")
        else:
            mark = "OK " if m["matched"] else "OUT OF TOLERANCE"
            print(f"  {mark}  B={m['tokens_B']} tok / D={m['tokens_D']} tok  "
                  f"deviation {m['token_deviation']:.1%} "
                  f"(tolerance {m['token_tolerance']:.0%})")
            print(f"        calls B={m['calls_B']} D={m['calls_D']} "
                  f"(deviation {m['call_deviation']}, tol {m['call_tolerance']})")
            if not m["matched"]:
                print("        This pair should be EXCLUDED from the matched "
                      "analysis and counted in the exclusion rate.")

    # Overhead decomposition for this single task
    if "A" in records and "C" in records:
        a, c = records["A"], records["C"]
        if a["total_tokens"] and c["total_tokens"]:
            print("\nOVERHEAD (this task only, n=1 -- indicative, not evidence)")
            print(f"  architecture  C-A : "
                  f"{c['total_tokens']-a['total_tokens']:+d} tokens, "
                  f"{c['llm_calls']-a['llm_calls']:+d} calls, "
                  f"{c['latency_seconds']-a['latency_seconds']:+.1f}s")
        if "D" in records and records["D"]["total_tokens"] and c["total_tokens"]:
            d = records["D"]
            print(f"  framework     D-C : "
                  f"{d['total_tokens']-c['total_tokens']:+d} tokens, "
                  f"{d['llm_calls']-c['llm_calls']:+d} calls, "
                  f"{d['latency_seconds']-c['latency_seconds']:+.1f}s")

    # Failures and degradation
    problems = [(c, r) for c, r in records.items()
                if r.get("failure_type") or r.get("notes", {}).get("researcher_degraded")]
    if problems:
        print("\nISSUES")
        for cond, r in sorted(problems):
            if r.get("failure_type"):
                print(f"  {cond}: FAILED ({r['failure_type']}, "
                      f"stopped={r['stopped_reason']})")
            if r.get("notes", {}).get("researcher_degraded"):
                print(f"  {cond}: researcher stalled "
                      f"({r['notes'].get('researcher_stopped_reason')}), "
                      f"partial evidence passed to writer")

    print("\nANSWERS")
    for cond in ALL_CONDITIONS:
        r = records.get(cond)
        if r is None:
            continue
        ans = (r.get("final_answer") or "(none)").strip().replace("\n", " ")
        print(f"  [{cond}] {ans[:180]}{'...' if len(ans) > 180 else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one task through all four conditions")
    ap.add_argument("--task", help="single task id, e.g. Q001")
    ap.add_argument("--tasks", help="comma-separated task ids")
    ap.add_argument("--first", type=int, help="use the first N tasks")
    ap.add_argument("--tier", type=int, choices=[1, 2, 3],
                    help="restrict --first to one tier")
    ap.add_argument("--conditions", default="A,C,D,B")
    ap.add_argument("--parallel", action="store_true",
                    help="run conditions concurrently (timing becomes invalid)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task-set", default="tasks/tasks.json")
    ap.add_argument("--save", action="store_true",
                    help="also append records to results/raw/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("experiment.output_directory", "results"))
    setup_logging(cfg.get("logging.level", "INFO"),
                  cfg.get("logging.console", True), out_dir / "run_task.log")
    validate_config(cfg).raise_if_failed()

    all_tasks = load_tasks(args.task_set, cfg, tier=args.tier)
    by_id = {t.task_id: t for t in all_tasks}

    if args.task:
        selected = [by_id[args.task]] if args.task in by_id else []
        if not selected:
            print(f"Task '{args.task}' not found. Available: "
                  f"{', '.join(list(by_id)[:10])} ...")
            return 1
    elif args.tasks:
        selected = [by_id[t.strip()] for t in args.tasks.split(",")
                    if t.strip() in by_id]
    elif args.first:
        selected = all_tasks[: args.first]
    else:
        ap.error("Specify --task, --tasks or --first")

    conditions = [c.strip().upper() for c in args.conditions.split(",")]

    client = client_from_config(cfg)
    if not client.is_available():
        print(f"Ollama unreachable at {client.host}. Start it: ollama serve")
        return 2
    if not client.model_present():
        print(f"Model '{client.model}' missing. Run: ollama pull {client.model}")
        return 2

    judge_client = None
    if cfg.get("judge.enabled", True):
        jc = client_from_config(cfg, judge=True)
        judge_client = jc if jc.model_present() else None

    hardware = detect_hardware()
    all_records: Dict[str, Dict[str, Any]] = {}

    for task in selected:
        recs = run_task_all_conditions(cfg, task, conditions, args.parallel,
                                       judge_client, hardware)
        print_comparison(task, recs, cfg, args.parallel)
        all_records[task.task_id] = recs

        if args.save:
            for cond, rec in recs.items():
                path = _raw_path(cfg, cond)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Cross-task summary when more than one task was run.
    if len(selected) > 1:
        print("\n" + "=" * 78)
        print(f"SUMMARY ACROSS {len(selected)} TASKS")
        print("=" * 78)
        print(f"{'Condition':<24}{'ok':>8}{'mean tok':>11}{'mean calls':>12}"
              f"{'mean lat':>11}")
        print("-" * 78)
        for cond in ALL_CONDITIONS:
            recs = [r[cond] for r in all_records.values() if cond in r]
            if not recs:
                continue
            ok = sum(1 for r in recs if r["success_execution"])
            toks = [r["total_tokens"] for r in recs if r["total_tokens"]]
            calls = [r["llm_calls"] for r in recs if r["llm_calls"]]
            lats = [r["latency_seconds"] for r in recs if r["latency_seconds"]]
            print(f"{LABELS[cond]:<24}{f'{ok}/{len(recs)}':>8}"
                  f"{(sum(toks)/len(toks) if toks else 0):>11.0f}"
                  f"{(sum(calls)/len(calls) if calls else 0):>12.1f}"
                  f"{(sum(lats)/len(lats) if lats else 0):>10.1f}s")
        if args.parallel:
            print("\nNOTE: parallel mode -- latency and RAM above are "
                  "contended and must not be reported.")

    out_path = out_dir / "processed" / "per_task_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_records, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
    print(f"\nSaved -> {out_path}")
    if not args.save:
        print("(use --save to also append these to results/raw/ for the "
              "main analysis)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
