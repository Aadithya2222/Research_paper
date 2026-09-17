"""Evaluate raw experimental records.

    python evaluate.py

Reads results/raw/condition_*/*.jsonl, scores quality, role fidelity and
failures, and writes results/processed/evaluated.jsonl.

Execution and evaluation are deliberately separate programs so that
evaluation can be re-run (for example with a different judge) without
re-running any inference.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.tasks import load_tasks                      # noqa: E402
from core.config import load_config, setup_logging          # noqa: E402
from core.llm import client_from_config                     # noqa: E402
from evaluation.evaluate import evaluate_file               # noqa: E402
from validation.checks import validate_results              # noqa: E402

LOG = logging.getLogger("evaluate")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate raw benchmark records")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task-set", default="tasks/tasks.json")
    ap.add_argument("--no-judge", action="store_true",
                    help="Deterministic metrics only; skip all judge calls.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("experiment.output_directory", "results"))
    setup_logging(cfg.get("logging.level", "INFO"), True, out_dir / "evaluate.log")

    raw_dir = Path(cfg.get("logging.raw_directory", "results/raw"))
    report = validate_results(raw_dir)
    for w in report.warnings:
        LOG.warning("%s", w)
    if not report.ok:
        for e in report.errors:
            LOG.error("%s", e)
        LOG.error("Raw data failed integrity checks. Fix before evaluating.")
        return 2

    tasks = {t.task_id: t for t in load_tasks(args.task_set, cfg)}

    judge = None
    if not args.no_judge and cfg.get("judge.enabled", True):
        judge = client_from_config(cfg, judge=True)
        if not judge.model_present():
            LOG.warning("Judge model '%s' unavailable; using deterministic "
                        "metrics only. Judge-dependent fields will be null.",
                        judge.model)
            judge = None

    all_rows = []
    for cond in ("A", "B", "C", "D"):
        path = raw_dir / f"condition_{cond}" / f"condition_{cond}.jsonl"
        rows = evaluate_file(cfg, path, tasks, judge)
        LOG.info("Condition %s: evaluated %d records.", cond, len(rows))
        all_rows.extend(rows)

    if not all_rows:
        LOG.error("Nothing to evaluate. Run experiments first.")
        return 3

    processed = out_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    target = processed / "evaluated.jsonl"
    with target.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOG.info("Wrote %d evaluated records to %s", len(all_rows), target)
    LOG.info("Next: python analyze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
