"""The experiment runner.

Responsibilities:
  * execute one (task, condition, run) triple
  * measure resources around it
  * write ONE raw JSONL record per execution -- never aggregated only
  * classify and preserve failures rather than discarding them
  * enforce the two-phase compute-matching protocol
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.conditions import (ConditionOutput, load_budget, run_condition_a,
                               run_condition_b, run_condition_c,
                               run_condition_d, write_budget)
from benchmark.tasks import Task
from core.config import detect_hardware, make_experiment_id
from core.llm import OllamaClient, client_from_config
from core.resources import ResourceMonitor
from tools.local_tools import ToolRegistry

LOG = logging.getLogger(__name__)

CONDITIONS = ("A", "B", "C", "D")


def _raw_path(cfg, condition: str) -> Path:
    base = Path(cfg.get("logging.raw_directory", "results/raw"))
    d = base / f"condition_{condition}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"condition_{condition}.jsonl"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one(cfg, condition: str, task: Task, run_index: int,
            client: OllamaClient, registry: ToolRegistry,
            judge_client: Optional[OllamaClient],
            hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one task under one condition and return the raw record."""
    exp_id = make_experiment_id(condition, task.complexity_tier or 0,
                                run_index, task.task_id)
    timeout = int(cfg.get("experiment.timeout_seconds", 600))
    started_wall = time.perf_counter()
    failure_type: Optional[str] = None
    out: Optional[ConditionOutput] = None

    with ResourceMonitor(float(cfg.get("experiment.resource_sample_hz", 5))) as mon:
        try:
            if condition == "A":
                out = run_condition_a(cfg, client, registry, task.to_dict())
            elif condition == "C":
                out = run_condition_c(cfg, client, registry, task.to_dict())
            elif condition == "D":
                out = run_condition_d(cfg, client, registry, task.to_dict())
            elif condition == "B":
                processed = Path(cfg.get("experiment.output_directory",
                                         "results")) / "processed"
                budget = load_budget(processed, task.task_id)
                out = run_condition_b(cfg, client, registry, task.to_dict(),
                                      budget, judge_client)
            else:
                raise ValueError(f"Unknown condition: {condition}")
        except FileNotFoundError:
            raise  # phase-ordering error: must surface, not be swallowed
        except Exception as exc:
            failure_type = "framework_error"
            LOG.exception("Condition %s failed on task %s", condition, task.task_id)
            from core.llm import Ledger
            out = ConditionOutput(None, Ledger(), 0, "exception", {}, [],
                                  failure_type=failure_type,
                                  notes={"error": f"{type(exc).__name__}: {exc}"})

    wall = time.perf_counter() - started_wall
    res = mon.summary()
    include_judge = bool(cfg.get("compute_matching.include_judge_cost", True))
    ledger = out.ledger.summary(include_judge=include_judge)

    if wall > timeout and out.failure_type is None:
        out.failure_type = "timeout"

    record: Dict[str, Any] = {
        "experiment_id": exp_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "task_id": task.task_id,
        "complexity_tier": task.complexity_tier,
        "complexity_score": task.complexity_score,
        "evaluation_type": task.evaluation_type,
        "run_id": run_index,
        "config_hash": cfg.config_hash,
        "model": cfg.require("model.name"),
        "quantization": cfg.get("model.quantization"),
        "temperature": cfg.get("model.temperature"),
        "seed": cfg.get("model.seed"),
        "num_ctx": cfg.get("model.num_ctx"),
        # --- compute accounting ---
        "input_tokens": ledger["input_tokens"],
        "output_tokens": ledger["output_tokens"],
        "total_tokens": ledger["total_tokens"],
        "llm_calls": ledger["llm_calls"],
        "agent_llm_calls": ledger["agent_llm_calls"],
        "judge_llm_calls": ledger["judge_llm_calls"],
        "tool_calls": ledger["tool_calls"],
        "retries": ledger["retries"],
        "iterations": out.iterations,
        # --- timing ---
        "latency_seconds": round(wall, 4),
        "model_latency_seconds": ledger["model_latency_seconds"],
        "tool_latency_seconds": ledger["tool_latency_seconds"],
        "ttft_seconds": ledger["first_ttft_seconds"],
        # --- resources (None when unmeasurable; never fabricated) ---
        **res.to_dict(),
        # --- outcome ---
        "final_answer": out.final_answer,
        "stopped_reason": out.stopped_reason,
        "failure_type": out.failure_type,
        "success_execution": out.final_answer is not None,
        # --- provenance for mechanism analysis ---
        "stage_outputs": out.stage_outputs,
        "handoffs": out.handoffs,
        "notes": out.notes,
        "call_ledger": ledger["call_ledger"],
        "hardware": hardware,
    }
    _append_jsonl(_raw_path(cfg, condition), record)
    return record


def completed_keys(cfg, condition: str) -> set:
    """Which (task_id, run_id) pairs are already on disk for this condition.

    Every finished task is appended to the JSONL immediately, so this set
    is accurate even if the previous run was killed mid-task: the
    interrupted task simply is not in it and will be redone.
    """
    path = _raw_path(cfg, condition)
    done: set = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a hard power loss
            done.add((rec.get("task_id"), rec.get("run_id", 1)))
    return done


def load_existing(cfg, condition: str) -> List[Dict[str, Any]]:
    """Read back every record already written for this condition."""
    path = _raw_path(cfg, condition)
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def run_condition(cfg, condition: str, tasks: List[Task], runs: int,
                  judge_client: Optional[OllamaClient] = None,
                  resume: bool = True,
                  max_hours: Optional[float] = None) -> List[Dict[str, Any]]:
    """Run every task under one condition, `runs` times each.

    resume     skip (task, run) pairs already present in the raw file.
               On by default: rerunning after a crash continues rather
               than duplicating work.
    max_hours  stop cleanly after this many hours. Useful for running in
               short batches to avoid thermal throttling on a laptop.
               Never interrupts a task mid-flight; the check happens
               between tasks so no partial record is written.
    """
    from tools.local_tools import build_registry

    client = client_from_config(cfg)
    registry = build_registry(cfg)
    hardware = detect_hardware()
    records: List[Dict[str, Any]] = []

    already = completed_keys(cfg, condition) if resume else set()
    if already:
        LOG.info("[%s] resuming: %d task-run pair(s) already complete",
                 condition, len(already))

    queue = [(run_index, task)
             for run_index in range(1, runs + 1)
             for task in tasks
             if (task.task_id, run_index) not in already]

    total = len(tasks) * runs
    skipped = total - len(queue)
    if skipped:
        LOG.info("[%s] skipping %d, running %d", condition, skipped, len(queue))

    deadline = (time.time() + max_hours * 3600) if max_hours else None
    stopped_early = False

    for i, (run_index, task) in enumerate(queue, start=1):
        if deadline is not None and time.time() >= deadline:
            LOG.info("[%s] time budget reached; stopping cleanly with %d "
                     "task(s) left. Rerun the same command to continue.",
                     condition, len(queue) - i + 1)
            stopped_early = True
            break

        LOG.info("[%s] %d/%d (%d skipped)  task=%s tier=%s run=%d",
                 condition, i, len(queue), skipped, task.task_id,
                 task.complexity_tier, run_index)
        rec = run_one(cfg, condition, task, run_index, client, registry,
                      judge_client, hardware)
        records.append(rec)
        status = "ok" if rec["success_execution"] else f"FAIL:{rec['failure_type']}"
        LOG.info("      -> %s  tokens=%s calls=%s latency=%.1fs",
                 status, rec["total_tokens"], rec["llm_calls"],
                 rec["latency_seconds"])

    if condition == "D":
        # The budget must cover EVERY completed task, not only the ones
        # run in this batch, or Condition B loses the earlier budgets.
        _emit_budget(cfg, load_existing(cfg, "D"))
        if stopped_early:
            LOG.warning("Condition D is incomplete. Do not start Condition B "
                        "until D has finished every task.")

    return records


def _emit_budget(cfg, records: List[Dict[str, Any]]) -> None:
    """Phase 1 output: Condition D's cost, with NO quality fields.

    This file is the only channel by which Condition B learns anything
    about Condition D, which is what makes the matching unbiased.
    """
    per_task: Dict[str, Dict[str, float]] = {}
    for r in records:
        if r["total_tokens"] is None:
            continue
        prev = per_task.get(r["task_id"])
        entry = {"total_tokens": float(r["total_tokens"]),
                 "llm_calls": float(r["llm_calls"]),
                 "input_tokens": float(r["input_tokens"] or 0),
                 "output_tokens": float(r["output_tokens"] or 0)}
        if prev is None:
            per_task[r["task_id"]] = entry
        else:
            for k in entry:
                prev[k] = (prev[k] + entry[k]) / 2.0

    processed = Path(cfg.get("experiment.output_directory", "results")) / "processed"
    single = _median_single_run_tokens(cfg)
    path = write_budget(processed, per_task, single)
    LOG.info("Compute budget written: %s (%d tasks)", path, len(per_task))


def _median_single_run_tokens(cfg) -> Optional[float]:
    """Median total tokens of one Condition A run, used to size N."""
    path = _raw_path(cfg, "A")
    if not path.exists():
        LOG.warning("Condition A results not found; N will default to min_n. "
                    "Run Condition A before Condition B for correct matching.")
        return None
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("total_tokens"):
                vals.append(float(rec["total_tokens"]))
    return statistics.median(vals) if vals else None
