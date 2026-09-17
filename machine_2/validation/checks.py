"""Pre-flight fairness checks and post-hoc data integrity checks.

The benchmark REFUSES TO RUN when a fairness precondition is violated.
A silently unfair comparison produces numbers that look fine and mean
nothing, which is far worse than a crash.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

LOG = logging.getLogger(__name__)


class FairnessError(RuntimeError):
    """Raised when conditions would not be compared on equal terms."""


@dataclass
class CheckReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        for w in self.warnings:
            LOG.warning("VALIDATION WARNING: %s", w)
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise FairnessError(
                f"Configuration failed {len(self.errors)} fairness check(s):"
                f"\n  - {joined}\n"
                "Fix config.yaml and rerun. The benchmark will not proceed."
            )
        LOG.info("Fairness checks passed (%d warnings).", len(self.warnings))


def validate_config(cfg) -> CheckReport:
    """Check that all four conditions will be compared on equal terms."""
    rep = CheckReport()

    # --- model equivalence ------------------------------------------
    if not cfg.get("model.name"):
        rep.errors.append("model.name is not set.")
    temp = cfg.get("model.temperature")
    if temp is None:
        rep.errors.append("model.temperature is not set.")
    elif float(temp) != 0.0:
        rep.warnings.append(
            f"model.temperature={temp}. The main run should be deterministic "
            "(0.0); non-zero belongs to the stochasticity substudy.")
    if cfg.get("model.seed") is None:
        rep.warnings.append("model.seed is not set; runs will not be reproducible.")
    if not cfg.get("model.num_ctx"):
        rep.errors.append("model.num_ctx must be set and identical across conditions.")

    # --- judge independence -----------------------------------------
    if cfg.get("judge.enabled", True):
        if cfg.get("judge.name") == cfg.get("model.name"):
            rep.errors.append(
                "judge.name equals model.name. Judging answers with the model "
                "that produced them introduces self-preference bias. Use a "
                "different model family.")
        if not cfg.get("judge.position_swap", True):
            rep.warnings.append(
                "judge.position_swap is off; position bias will be unmeasured.")

    # --- role charter completeness ----------------------------------
    roles = cfg.get("roles", {})
    for key in ("planner", "researcher", "writer"):
        if key not in roles:
            rep.errors.append(f"roles.{key} is missing.")
        elif not (roles[key].get("charter") or "").strip():
            rep.errors.append(f"roles.{key}.charter is empty.")

    # --- tools --------------------------------------------------------
    if not cfg.get("tools.enabled"):
        rep.errors.append("tools.enabled is empty; all conditions need tools.")

    # --- compute matching --------------------------------------------
    tol = cfg.get("compute_matching.token_tolerance_percent")
    if tol is None:
        rep.errors.append("compute_matching.token_tolerance_percent is not set.")
    elif float(tol) > 25:
        rep.warnings.append(
            f"token tolerance {tol}% is loose; matching will be weak.")
    if not cfg.get("compute_matching.include_judge_cost", True):
        rep.warnings.append(
            "Selection/judge cost is excluded from the budget. Condition B "
            "then gets free compute that Condition D pays for.")
    if not cfg.get("compute_matching.include_retries", True):
        rep.warnings.append("Retries excluded from the budget; costs are hidden.")

    # --- complexity ---------------------------------------------------
    if not cfg.get("complexity.weights"):
        rep.errors.append("complexity.weights is missing (Eq. 1 cannot be applied).")
    cuts = cfg.get("complexity.tier_cutpoints", {})
    if "tier1_max" not in cuts or "tier2_max" not in cuts:
        rep.errors.append("complexity.tier_cutpoints incomplete.")
    elif float(cuts["tier1_max"]) >= float(cuts["tier2_max"]):
        rep.errors.append("tier1_max must be strictly less than tier2_max.")

    # --- statistics ---------------------------------------------------
    mid = cfg.get("statistics.minimal_important_difference", {})
    if "quality_points" not in mid:
        rep.warnings.append(
            "No minimal important difference declared; practical significance "
            "cannot be assessed.")
    return rep


def validate_tasks(tasks: List[Any]) -> CheckReport:
    """Check the task set for duplicates, empties and missing gold data."""
    rep = CheckReport()
    if not tasks:
        rep.errors.append("Task set is empty.")
        return rep

    ids = [t.task_id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        rep.errors.append(f"Duplicate task ids: {sorted(dupes)}")

    for t in tasks:
        if not (t.task_text or "").strip():
            rep.errors.append(f"Task {t.task_id} has empty task_text.")
        if t.evaluation_type == "exact_match" and not t.expected_answer:
            rep.errors.append(
                f"Task {t.task_id} is exact_match but has no expected_answer.")
        if t.complexity_tier not in (1, 2, 3):
            rep.errors.append(
                f"Task {t.task_id} has invalid tier {t.complexity_tier}.")

    dist = {1: 0, 2: 0, 3: 0}
    for t in tasks:
        if t.complexity_tier in dist:
            dist[t.complexity_tier] += 1
    for tier, n in dist.items():
        if n == 0:
            rep.warnings.append(f"Tier {tier} has no tasks.")
    return rep


def validate_results(raw_dir: Path) -> CheckReport:
    """Post-hoc integrity check of the raw JSONL records."""
    rep = CheckReport()
    seen_ids: set[str] = set()
    n = 0

    for cond in ("A", "B", "C", "D"):
        path = Path(raw_dir) / f"condition_{cond}" / f"condition_{cond}.jsonl"
        if not path.exists():
            rep.warnings.append(f"No results for condition {cond}.")
            continue
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                n += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    rep.errors.append(f"{path}:{lineno} is not valid JSON: {exc}")
                    continue

                eid = rec.get("experiment_id")
                if eid in seen_ids:
                    rep.errors.append(f"Duplicate experiment_id: {eid}")
                seen_ids.add(eid)

                lat = rec.get("latency_seconds")
                if lat is not None and lat < 0:
                    rep.errors.append(f"{eid}: negative latency {lat}")
                for k in ("input_tokens", "output_tokens", "total_tokens",
                          "llm_calls", "tool_calls"):
                    v = rec.get(k)
                    if v is not None and v < 0:
                        rep.errors.append(f"{eid}: negative {k}={v}")
                if rec.get("total_tokens") is None:
                    rep.warnings.append(
                        f"{eid}: total_tokens is null (token accounting "
                        "incomplete for this run).")
                if not rec.get("config_hash"):
                    rep.errors.append(f"{eid}: missing config_hash.")
                if not rec.get("model"):
                    rep.errors.append(f"{eid}: missing model metadata.")
                if rec.get("complexity_tier") not in (1, 2, 3):
                    rep.errors.append(f"{eid}: invalid tier.")

    LOG.info("Checked %d raw records.", n)
    return rep
