"""Mechanism analysis: the parts of the paper that had no code behind them.

Produces:
  * Table 9  -- failure stage distribution by MAST category (RQ3)
  * H3 test  -- per-role output quality vs cumulative input context length
  * H4 test  -- role bleed association with downstream task failure
  * Table 10 inputs -- role adherence and bleed, by role and tier

MAST categories are taken from Cemri et al. and are NOT re-derived here.
Stage attribution uses the recorded stopped_reason and stage outputs; any
failure that cannot be attributed from the logs is reported as
"unattributed" rather than being assigned to a stage by guesswork.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

try:
    from scipy import stats as sps
    SCIPY = True
except ImportError:  # pragma: no cover
    SCIPY = False

# MAST top-level categories (Cemri et al.).
SYSTEM_DESIGN = "system_design"
MISALIGNMENT = "inter_agent_misalignment"
VERIFICATION = "task_verification"
UNATTRIBUTED = "unattributed"

# Map our observable failure signals onto MAST categories. Each mapping is
# a defensible reading of the log, not an inference about intent.
FAILURE_TO_MAST = {
    "planner_failure": (SYSTEM_DESIGN, "planner"),
    "researcher_failure": (SYSTEM_DESIGN, "researcher"),
    "writer_failure": (SYSTEM_DESIGN, "writer"),
    "handoff_failure": (MISALIGNMENT, "handoff"),
    "tool_failure": (SYSTEM_DESIGN, "researcher"),
    "timeout": (SYSTEM_DESIGN, "researcher"),
    "framework_error": (SYSTEM_DESIGN, "unattributed"),
    "model_error": (SYSTEM_DESIGN, "unattributed"),
    "evaluation_error": (VERIFICATION, "unattributed"),
}

STOPPED_TO_MAST = {
    "repeat_loop": (SYSTEM_DESIGN, "researcher"),
    "max_iterations": (SYSTEM_DESIGN, "researcher"),
    "planner_empty": (SYSTEM_DESIGN, "planner"),
    "researcher_empty": (SYSTEM_DESIGN, "researcher"),
    "writer_empty": (SYSTEM_DESIGN, "writer"),
}


def _load_raw(raw_dir: Path, conditions=("C", "D")) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cond in conditions:
        p = Path(raw_dir) / f"condition_{cond}" / f"condition_{cond}.jsonl"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def annotate_failure_stages(raw_rows: List[Dict[str, Any]],
                            tiers=(2, 3)) -> Dict[str, Any]:
    """Table 9: where multi-agent failures originate, by MAST category.

    Restricted to Tiers 2-3 as specified in the paper, because Tier 1
    failures are rare and rarely architectural.
    """
    table: Dict[str, Counter] = defaultdict(Counter)
    detail: List[Dict[str, Any]] = []
    n_failed = 0

    for r in raw_rows:
        if r.get("complexity_tier") not in tiers:
            continue
        failure = r.get("failure_type")
        degraded = (r.get("notes") or {}).get("researcher_degraded")
        if not failure and not degraded:
            continue
        n_failed += 1

        if failure and failure in FAILURE_TO_MAST:
            category, stage = FAILURE_TO_MAST[failure]
        else:
            category, stage = STOPPED_TO_MAST.get(
                r.get("stopped_reason", ""), (UNATTRIBUTED, "unattributed"))

        # A stalled researcher whose partial evidence reached the writer is
        # an inter-agent handoff problem, not a hard component failure.
        if degraded and not failure:
            category, stage = MISALIGNMENT, "handoff"

        table[stage][category] += 1
        detail.append({
            "experiment_id": r.get("experiment_id"),
            "condition": r.get("condition"),
            "task_id": r.get("task_id"),
            "tier": r.get("complexity_tier"),
            "failure_type": failure,
            "stopped_reason": r.get("stopped_reason"),
            "degraded": bool(degraded),
            "mast_category": category,
            "originating_stage": stage,
        })

    stages = ["planner", "researcher", "writer", "handoff", "unattributed"]
    cats = [SYSTEM_DESIGN, MISALIGNMENT, VERIFICATION, UNATTRIBUTED]
    matrix = {s: {c: int(table[s][c]) for c in cats} for s in stages}

    return {
        "n_failed_or_degraded": n_failed,
        "tiers_analysed": list(tiers),
        "matrix": matrix,
        "detail": detail,
        "note": ("Stage attribution derived from execution logs. Failures "
                 "that cannot be attributed are reported as unattributed "
                 "rather than assigned by guesswork."),
    }


def context_growth_analysis(raw_rows: List[Dict[str, Any]],
                            evaluated: List[Dict[str, Any]]) -> Dict[str, Any]:
    """H3: does per-role output quality fall as input context accumulates?

    Uses the actual per-stage input length recorded in stage_outputs, and
    the role adherence score for that stage as the quality proxy, because
    per-stage task quality is not separately gradable.
    """
    rfs_by_exp: Dict[str, Dict[str, Any]] = {}
    for e in evaluated:
        if e.get("role_detail"):
            rfs_by_exp[e["experiment_id"]] = e["role_detail"]

    points: List[Tuple[int, float, str]] = []
    for r in raw_rows:
        stages = r.get("stage_outputs") or {}
        detail = rfs_by_exp.get(r.get("experiment_id"), {})
        cumulative = 0
        for role in ("planner", "researcher", "writer"):
            st = stages.get(role) or {}
            in_len = len(str(st.get("input", "")))
            cumulative += in_len
            score = (detail.get(role) or {}).get("score")
            if score is not None and cumulative > 0:
                points.append((cumulative, float(score), role))

    if len(points) < 5:
        return {"n": len(points), "result": None,
                "note": "fewer than 5 usable stage observations; not tested"}

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    out: Dict[str, Any] = {"n": len(points)}
    if SCIPY:
        rho, p = sps.spearmanr(xs, ys)
        out["spearman_rho"] = round(float(rho), 4)
        out["p_value"] = round(float(p), 6)
        out["direction"] = ("negative (consistent with H3)" if rho < 0
                            else "positive or flat (H3 not supported)")
    slope, intercept = np.polyfit(xs, ys, 1)
    out["ols_slope_per_1k_chars"] = round(float(slope) * 1000, 5)

    by_role: Dict[str, Any] = {}
    for role in ("planner", "researcher", "writer"):
        rp = [(x, y) for x, y, r_ in points if r_ == role]
        if len(rp) >= 5 and SCIPY:
            rx = np.array([p[0] for p in rp], float)
            ry = np.array([p[1] for p in rp], float)
            rho, p = sps.spearmanr(rx, ry)
            by_role[role] = {"n": len(rp), "spearman_rho": round(float(rho), 4),
                             "p_value": round(float(p), 6),
                             "mean_context_chars": round(float(rx.mean()), 1)}
        else:
            by_role[role] = {"n": len(rp), "spearman_rho": None,
                             "p_value": None}
    out["by_role"] = by_role
    return out


def bleed_failure_association(evaluated: List[Dict[str, Any]]) -> Dict[str, Any]:
    """H4: do role bleed events predict downstream task failure?

    Reported as a 2x2 contingency table with Fisher's exact test, which is
    appropriate at the sample sizes a single-laptop study produces.
    """
    rows = [e for e in evaluated
            if e.get("condition") in ("C", "D")
            and e.get("role_bleed_rate") is not None
            and e.get("success") is not None]
    if len(rows) < 8:
        return {"n": len(rows), "result": None,
                "note": "fewer than 8 usable observations; not tested"}

    bleed_fail = sum(1 for e in rows if e["role_bleed_rate"] > 0 and not e["success"])
    bleed_ok = sum(1 for e in rows if e["role_bleed_rate"] > 0 and e["success"])
    clean_fail = sum(1 for e in rows if e["role_bleed_rate"] == 0 and not e["success"])
    clean_ok = sum(1 for e in rows if e["role_bleed_rate"] == 0 and e["success"])

    out: Dict[str, Any] = {
        "n": len(rows),
        "contingency": {"bleed_fail": bleed_fail, "bleed_success": bleed_ok,
                        "clean_fail": clean_fail, "clean_success": clean_ok},
        "failure_rate_with_bleed": (round(bleed_fail / (bleed_fail + bleed_ok), 4)
                                    if (bleed_fail + bleed_ok) else None),
        "failure_rate_without_bleed": (round(clean_fail / (clean_fail + clean_ok), 4)
                                       if (clean_fail + clean_ok) else None),
    }
    if SCIPY and min(bleed_fail + bleed_ok, clean_fail + clean_ok) > 0:
        odds, p = sps.fisher_exact([[bleed_fail, bleed_ok],
                                    [clean_fail, clean_ok]])
        out["odds_ratio"] = round(float(odds), 4) if math.isfinite(odds) else None
        out["p_value"] = round(float(p), 6)
        out["test_used"] = "fisher_exact"
        out["note"] = "exploratory; corrected with BH-FDR alongside other exploratory tests"
    return out


def role_fidelity_table(evaluated: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Table 10: role adherence and bleed, by role and tier."""
    acc: Dict[str, Dict[int, List[float]]] = {
        r: defaultdict(list) for r in ("planner", "researcher", "writer")}
    bleed: Dict[str, List[bool]] = defaultdict(list)

    for e in evaluated:
        detail = e.get("role_detail") or {}
        tier = e.get("complexity_tier")
        for role, v in detail.items():
            if v.get("score") is not None and tier in (1, 2, 3):
                acc[role][tier].append(float(v["score"]))
            if v.get("bleed") is not None:
                bleed[role].append(bool(v["bleed"]))

    scale = 4.0
    table: Dict[str, Any] = {}
    for role in ("planner", "researcher", "writer"):
        row: Dict[str, Any] = {}
        for tier in (1, 2, 3):
            vals = acc[role][tier]
            row[f"tier_{tier}_rfs"] = (round(sum(vals) / len(vals) / scale * 100, 2)
                                       if vals else None)
            row[f"tier_{tier}_n"] = len(vals)
        b = bleed[role]
        row["bleed_rate_percent"] = (round(sum(b) / len(b) * 100, 2) if b else None)
        row["bleed_n"] = len(b)
        table[role] = row

    table["_validation"] = {
        "spearman_vs_cras": None,
        "human_agreement_kappa_w": None,
        "note": ("CRAS correlation and blind human agreement must be computed "
                 "from a separately annotated subsample; both are null until "
                 "that annotation exists. Do not report RFS as validated "
                 "before they are filled."),
    }
    return table


def run_mechanism_analysis(cfg, raw_dir: Path, processed_dir: Path,
                           out_dir: Path) -> Dict[str, Any]:
    """Run every mechanism analysis and write the report."""
    raw_rows = _load_raw(raw_dir)
    ev_path = Path(processed_dir) / "evaluated.jsonl"
    evaluated: List[Dict[str, Any]] = []
    if ev_path.exists():
        with ev_path.open("r", encoding="utf-8") as fh:
            evaluated = [json.loads(l) for l in fh if l.strip()]

    report = {
        "n_raw_multiagent_records": len(raw_rows),
        "n_evaluated_records": len(evaluated),
        "failure_stages": annotate_failure_stages(raw_rows),
        "h3_context_growth": context_growth_analysis(raw_rows, evaluated),
        "h4_bleed_failure": bleed_failure_association(evaluated),
        "role_fidelity": role_fidelity_table(evaluated),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mechanism_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Mechanism report written: %s", path)
    return report
