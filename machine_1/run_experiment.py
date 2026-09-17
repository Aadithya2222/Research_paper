"""Main entry point for running the benchmark.

Examples
--------
    python run_experiment.py --condition A
    python run_experiment.py --condition D
    python run_experiment.py --condition B          # needs D first
    python run_experiment.py --all
    python run_experiment.py --condition A --complexity-tier 1 --runs 3

Phase ordering (enforced in code, not by convention):
    1. A   -- establishes the cost of one single-agent run
    2. C   -- pipeline in LangGraph
    3. D   -- pipeline in CrewAI; writes the compute budget
    4. B   -- reads that budget to size N
--all runs them in exactly that order.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.runner import run_condition                       # noqa: E402
from benchmark.tasks import load_tasks, tier_distribution        # noqa: E402
from core.config import load_config, setup_logging               # noqa: E402
from core.llm import client_from_config                          # noqa: E402
from validation.checks import validate_config, validate_tasks    # noqa: E402

LOG = logging.getLogger("run_experiment")

PHASE_ORDER = ["A", "C", "D", "B"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Coordination Tax Benchmark")
    ap.add_argument("--condition", choices=["A", "B", "C", "D"],
                    help="Which condition to run.")
    ap.add_argument("--all", action="store_true",
                    help="Run all conditions in the required phase order.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task-set", default="tasks/tasks.json")
    ap.add_argument("--complexity-tier", type=int, choices=[1, 2, 3])
    ap.add_argument("--runs", type=int, help="Repetitions per task.")
    ap.add_argument("--limit", type=int, help="Use only the first N tasks.")
    ap.add_argument("--output", help="Override output directory.")
    ap.add_argument("--max-hours", type=float,
                    help="Stop cleanly after this many hours. Rerun the same "
                         "command to continue where it stopped.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Redo tasks already on disk (creates duplicates).")
    args = ap.parse_args()

    if not args.condition and not args.all:
        ap.error("Specify --condition {A,B,C,D} or --all")

    cfg = load_config(args.config)
    if args.output:
        cfg.data.setdefault("experiment", {})["output_directory"] = args.output

    out_dir = Path(cfg.get("experiment.output_directory", "results"))
    setup_logging(cfg.get("logging.level", "INFO"),
                  cfg.get("logging.console", True),
                  out_dir / "run.log")

    LOG.info("Config: %s   hash=%s", cfg.path, cfg.config_hash)

    # --- fairness gate: refuse to run an unfair comparison ----------
    validate_config(cfg).raise_if_failed()

    tasks = load_tasks(args.task_set, cfg, tier=args.complexity_tier)
    if args.limit:
        tasks = tasks[: args.limit]
    validate_tasks(tasks).raise_if_failed()
    LOG.info("Tier distribution: %s", tier_distribution(tasks))

    runs = args.runs or int(cfg.get("experiment.runs_per_task", 1))

    # --- model availability -----------------------------------------
    client = client_from_config(cfg)
    if not client.is_available():
        LOG.error("Cannot reach Ollama at %s. Is `ollama serve` running?",
                  client.host)
        return 2
    if not client.model_present():
        LOG.error("Model '%s' not found. Run: ollama pull %s",
                  client.model, client.model)
        return 2

    judge_client = None
    if cfg.get("judge.enabled", True):
        judge_client = client_from_config(cfg, judge=True)
        if not judge_client.model_present():
            LOG.warning("Judge model '%s' not found. Selection falls back to "
                        "self-consistency. Run: ollama pull %s",
                        judge_client.model, judge_client.model)
            judge_client = None

    conditions = PHASE_ORDER if args.all else [args.condition]
    if args.all:
        LOG.info("Running all conditions in phase order: %s",
                 " -> ".join(PHASE_ORDER))

    for cond in conditions:
        LOG.info("=" * 62)
        LOG.info("CONDITION %s  |  %d tasks x %d run(s)", cond, len(tasks), runs)
        LOG.info("=" * 62)
        try:
            run_condition(cfg, cond, tasks, runs, judge_client,
                          resume=not args.no_resume,
                          max_hours=args.max_hours)
        except FileNotFoundError as exc:
            LOG.error("%s", exc)
            return 3
        except RuntimeError as exc:
            LOG.error("Condition %s could not run: %s", cond, exc)
            if not args.all:
                return 4
            LOG.warning("Continuing with remaining conditions.")

    LOG.info("Done. Raw records: %s", out_dir / "raw")
    LOG.info("Next: python evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
