"""Statistical analysis.

Every test is chosen from the data structure, not by habit:

* The design is PAIRED at the task level (the same task is run under
  every condition), so paired non-parametric tests are the default.
* Quality is bounded and non-normal, so we use Wilcoxon signed-rank
  with bootstrap BCa confidence intervals rather than a t-test.
* Three primary hypotheses are corrected with Holm; everything else is
  labelled exploratory and corrected with Benjamini-Hochberg.
* Statistical significance and practical significance are reported
  SEPARATELY, against pre-registered minimal important differences.

Nothing here invents data. If a comparison has too few paired
observations, it returns n and a null result rather than a number.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

try:
    from scipy import stats as sps
    SCIPY = True
except ImportError:  # pragma: no cover
    SCIPY = False
    LOG.warning("scipy not installed: p-values will be null.")


@dataclass
class TestResult:
    name: str
    n_pairs: int
    median_difference: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    effect_size: Optional[float]
    effect_size_name: str
    p_value: Optional[float]
    p_corrected: Optional[float] = None
    test_used: str = ""
    practically_significant: Optional[bool] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42
                 ) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for the median of paired differences."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 3:
        return None, None
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(arr, size=(n_boot, arr.size), replace=True), axis=1)
    return (float(np.percentile(meds, 100 * alpha / 2)),
            float(np.percentile(meds, 100 * (1 - alpha / 2))))


def rank_biserial(differences: Sequence[float]) -> Optional[float]:
    """Matched-pairs rank-biserial correlation, the effect size that
    accompanies a Wilcoxon signed-rank test."""
    d = np.asarray([x for x in differences if x is not None and x != 0], float)
    if d.size == 0:
        return None
    ranks = sps.rankdata(np.abs(d)) if SCIPY else np.argsort(np.argsort(np.abs(d))) + 1
    total = ranks.sum()
    if total == 0:
        return None
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


def paired_wilcoxon(x: Sequence[float], y: Sequence[float], label: str,
                    mid: Optional[float] = None, n_boot: int = 2000,
                    alpha: float = 0.05) -> TestResult:
    """Paired comparison of x versus y (x - y)."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None
             and not (isinstance(a, float) and math.isnan(a))
             and not (isinstance(b, float) and math.isnan(b))]
    if len(pairs) < 3:
        return TestResult(label, len(pairs), None, None, None, None,
                          "rank_biserial", None, test_used="none",
                          note="fewer than 3 complete pairs")

    diffs = [a - b for a, b in pairs]
    med = float(np.median(diffs))
    lo, hi = bootstrap_ci(diffs, n_boot, alpha)
    eff = rank_biserial(diffs)

    p: Optional[float] = None
    used = "wilcoxon_signed_rank"
    if SCIPY:
        nz = [d for d in diffs if d != 0]
        if len(nz) >= 3:
            try:
                p = float(sps.wilcoxon(nz).pvalue)
            except Exception as exc:
                used, p = f"wilcoxon_failed:{exc}", None
        else:
            used, p = "all_differences_zero", None

    practical = None if mid is None else bool(abs(med) >= mid)
    return TestResult(label, len(pairs), round(med, 5), lo, hi, eff,
                      "rank_biserial", p, test_used=used,
                      practically_significant=practical)


def holm(pvals: List[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down correction."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    out: List[Optional[float]] = [None] * len(pvals)
    if not idx:
        return out
    order = sorted(idx, key=lambda i: pvals[i])  # type: ignore[index]
    m = len(order)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * float(pvals[i]))  # type: ignore[arg-type]
        prev = max(prev, adj)
        out[i] = round(prev, 6)
    return out


def benjamini_hochberg(pvals: List[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg FDR correction for exploratory analyses."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    out: List[Optional[float]] = [None] * len(pvals)
    if not idx:
        return out
    order = sorted(idx, key=lambda i: pvals[i])  # type: ignore[index]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        adj = min(prev, float(pvals[i]) * m / (rank + 1))  # type: ignore[arg-type]
        prev = adj
        out[i] = round(adj, 6)
    return out


# ---------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------

def load_evaluated(processed_dir: Path) -> List[Dict[str, Any]]:
    path = Path(processed_dir) / "evaluated.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No evaluated results at {path}. Run: python evaluate.py")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


TIMING_METRICS = {"latency_seconds", "ttft_seconds", "model_latency_seconds",
                  "peak_ram_mb", "mean_ram_mb", "peak_gpu_memory_mb"}


def paired_series(rows: List[Dict[str, Any]], cond_x: str, cond_y: str,
                  metric: str, tier: Optional[int] = None
                  ) -> Tuple[List[float], List[float], List[str]]:
    """Align two conditions on task_id (mean over runs) for one metric.

    Records produced in parallel execution mode are excluded from timing
    and resource metrics, because concurrent conditions contend for the
    same CPU, GPU and memory and those figures are then meaningless.
    Token and quality metrics are unaffected and are retained.
    """
    timing = metric in TIMING_METRICS

    def collect(cond: str) -> Dict[str, List[float]]:
        acc: Dict[str, List[float]] = {}
        for r in rows:
            if r["condition"] != cond:
                continue
            if timing and not r.get("timing_valid", True):
                continue
            if tier is not None and r["complexity_tier"] != tier:
                continue
            v = r.get(metric)
            if v is None:
                continue
            acc.setdefault(r["task_id"], []).append(float(v))
        return acc

    ax, ay = collect(cond_x), collect(cond_y)
    ids = sorted(set(ax) & set(ay))
    return ([float(np.mean(ax[i])) for i in ids],
            [float(np.mean(ay[i])) for i in ids], ids)


# ---------------------------------------------------------------------
# Headline analyses
# ---------------------------------------------------------------------

def compute_matched_delta(rows: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    """Delta_matched = Q(D) - Q(B), per tier. Sign is not assumed."""
    mid = float(cfg.get("statistics.minimal_important_difference.quality_points", 0.05))
    n_boot = int(cfg.get("statistics.n_bootstrap", 2000))
    alpha = float(cfg.get("statistics.alpha", 0.05))
    out: Dict[str, Any] = {}
    for tier in (None, 1, 2, 3):
        x, y, _ = paired_series(rows, "D", "B", "quality", tier)
        label = f"delta_matched_tier_{tier or 'all'}"
        out[label] = paired_wilcoxon(x, y, label, mid, n_boot, alpha).to_dict()
    return out


def coordination_tax(rows: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    """Overhead decomposition.

    O_arch = C - A   (architecture, within one framework)
    O_fw   = D - C   (framework, at fixed architecture)

    These are OPERATIONAL DECOMPOSITIONS of measured differences. They
    support a causal reading only to the extent that the framework and
    architecture were otherwise held constant, which this design does
    but which the manuscript must still argue explicitly.
    """
    n_boot = int(cfg.get("statistics.n_bootstrap", 2000))
    alpha = float(cfg.get("statistics.alpha", 0.05))
    metrics = ["latency_seconds", "total_tokens", "input_tokens",
               "output_tokens", "llm_calls", "tool_calls"]
    out: Dict[str, Any] = {}

    for metric in metrics:
        for label, cx, cy in (("O_arch_C_minus_A", "C", "A"),
                              ("O_fw_D_minus_C", "D", "C"),
                              ("MA_minus_SA_D_minus_A", "D", "A")):
            x, y, _ = paired_series(rows, cx, cy, metric)
            key = f"{label}::{metric}"
            res = paired_wilcoxon(x, y, key, None, n_boot, alpha).to_dict()
            # Relative "tax" percentages, reported ALONGSIDE absolutes.
            base = float(np.median(y)) if y else 0.0
            res["relative_percent"] = (
                round((res["median_difference"] / base) * 100, 2)
                if res["median_difference"] is not None and base > 0 else None
            )
            res["baseline_median"] = round(base, 4) if y else None
            res["interpretation"] = ("operational decomposition, not causal proof "
                                     "unless the design held all else constant")
            out[key] = res
    return out


def tier_interaction(rows: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    """H1: does Delta_matched vary across tiers?

    Reports the per-tier deltas and, when statsmodels is available, a
    mixed-effects model with a condition-by-tier interaction and a
    task-level random intercept.
    """
    result: Dict[str, Any] = {"per_tier": {}, "mixed_effects": None}
    for tier in (1, 2, 3):
        x, y, ids = paired_series(rows, "D", "B", "quality", tier)
        result["per_tier"][f"tier_{tier}"] = {
            "n_pairs": len(ids),
            "median_delta": float(np.median([a - b for a, b in zip(x, y)]))
            if x else None,
        }
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        recs = [r for r in rows
                if r["condition"] in ("B", "D") and r.get("success") is not None]
        if len({r["task_id"] for r in recs}) < 10:
            result["mixed_effects"] = {
                "note": "fewer than 10 paired tasks; model not fitted"}
            return result
        df = pd.DataFrame(recs)
        df["cond"] = (df["condition"] == "D").astype(int)
        df["tier"] = df["complexity_tier"].astype(float)
        model = smf.mixedlm("success ~ cond * tier", df, groups=df["task_id"])
        fit = model.fit(method="lbfgs", disp=False)
        result["mixed_effects"] = {
            "formula": "success ~ cond * tier + (1|task_id)",
            "params": {k: round(float(v), 5) for k, v in fit.params.items()},
            "pvalues": {k: round(float(v), 6) for k, v in fit.pvalues.items()},
            "interaction_term": "cond:tier",
            "note": ("Linear mixed model on a binary outcome. With a larger "
                     "sample prefer a mixed-effects LOGISTIC model "
                     "(statsmodels BinomialBayesMixedGLM or R lme4)."),
        }
    except Exception as exc:
        result["mixed_effects"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def break_even(rows: List[Dict[str, Any]], cfg,
               exchange_rates: Optional[List[float]] = None) -> Dict[str, Any]:
    """Break-even sensitivity surface.

    For each tier and each assumed quality-per-cost exchange rate r,
    report which condition is preferred. Computed from real measured
    data only; if data are absent the cell is null.
    """
    rates = exchange_rates or [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    surface: Dict[str, Any] = {"exchange_rates": rates, "cells": {}}
    for tier in (1, 2, 3):
        qx, qy, _ = paired_series(rows, "D", "B", "quality", tier)
        cx, cy, _ = paired_series(rows, "D", "B", "total_tokens", tier)
        if not qx or not cx:
            surface["cells"][f"tier_{tier}"] = None
            continue
        dq = float(np.median([a - b for a, b in zip(qx, qy)]))
        base = float(np.median(cy)) if cy else 0.0
        dc = float(np.median([a - b for a, b in zip(cx, cy)]))
        dc_norm = dc / base if base > 0 else 0.0
        surface["cells"][f"tier_{tier}"] = {
            "delta_quality": round(dq, 5),
            "delta_cost_normalised": round(dc_norm, 5),
            "preferred": {str(r): ("D" if dq > r * dc_norm else "B") for r in rates},
        }
    return surface


def run_all(cfg, processed_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Execute the full statistical pipeline and write the report."""
    rows = load_evaluated(processed_dir)
    LOG.info("Loaded %d evaluated records.", len(rows))

    report: Dict[str, Any] = {
        "n_records": len(rows),
        "config_alpha": cfg.get("statistics.alpha", 0.05),
        "delta_matched": compute_matched_delta(rows, cfg),
        "coordination_tax": coordination_tax(rows, cfg),
        "tier_interaction": tier_interaction(rows, cfg),
        "break_even": break_even(rows, cfg),
    }

    # Multiplicity: primary (Holm) versus exploratory (BH-FDR).
    primary_keys = ["delta_matched_tier_1", "delta_matched_tier_2",
                    "delta_matched_tier_3"]
    pvals = [report["delta_matched"].get(k, {}).get("p_value") for k in primary_keys]
    for k, adj in zip(primary_keys, holm(pvals)):
        if k in report["delta_matched"]:
            report["delta_matched"][k]["p_corrected"] = adj
            report["delta_matched"][k]["correction"] = "holm (primary)"

    tax_keys = list(report["coordination_tax"])
    tax_p = [report["coordination_tax"][k].get("p_value") for k in tax_keys]
    for k, adj in zip(tax_keys, benjamini_hochberg(tax_p)):
        report["coordination_tax"][k]["p_corrected"] = adj
        report["coordination_tax"][k]["correction"] = "benjamini_hochberg (exploratory)"

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "statistics_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    LOG.info("Statistics written: %s", path)
    return report
