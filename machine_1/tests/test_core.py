"""Unit tests for the measurement and analysis machinery.

    pytest -q

These test the parts where a silent bug would corrupt the scientific
conclusions: token accounting, compute matching, complexity scoring,
effect sizes and multiplicity correction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.statistics_pipeline import (benjamini_hochberg, bootstrap_ci,
                                          holm, paired_wilcoxon, rank_biserial)
from agents.conditions import choose_n, write_budget
from benchmark.tasks import assign_tier, score_complexity
from core.config import load_config, make_experiment_id
from core.llm import CallRecord, Ledger
from evaluation.evaluate import (classify_failure, exact_match,
                                 gold_fact_coverage)
from tools.local_tools import _calculator


@pytest.fixture(scope="module")
def cfg():
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


# ------------------------- token accounting -------------------------

def test_ledger_sums_tokens_including_retries():
    led = Ledger()
    led.add(CallRecord("a", 1, False, 100, 10, 1.0, None))   # failed attempt
    led.add(CallRecord("a", 2, True, 100, 50, 2.0, 0.3))     # retry succeeded
    s = led.summary()
    assert s["input_tokens"] == 200      # retry cost is NOT hidden
    assert s["output_tokens"] == 60
    assert s["total_tokens"] == 260
    assert s["llm_calls"] == 2
    assert s["retries"] == 1


def test_ledger_reports_none_when_counts_missing():
    """Missing token counts must propagate as None, never as zero."""
    led = Ledger()
    led.add(CallRecord("a", 1, True, None, 50, 1.0, None))
    assert led.summary()["input_tokens"] is None
    assert led.summary()["total_tokens"] is None


def test_judge_cost_included_and_excludable():
    led = Ledger()
    led.add(CallRecord("agent", 1, True, 100, 100, 1.0, None))
    led.add_judge(CallRecord("judge", 1, True, 40, 10, 0.5, None))
    assert led.summary(include_judge=True)["total_tokens"] == 250
    assert led.summary(include_judge=False)["total_tokens"] == 200


# ------------------------- compute matching -------------------------

def test_choose_n_scales_with_budget(cfg):
    assert choose_n(3000.0, 1000.0, cfg) == 3
    assert choose_n(1000.0, 1000.0, cfg) == 1


def test_choose_n_respects_bounds(cfg):
    assert choose_n(99999.0, 100.0, cfg) == cfg.get("compute_matching.max_n")
    assert choose_n(None, None, cfg) == cfg.get("compute_matching.min_n")


def test_budget_rejects_quality_fields(tmp_path):
    """The budget file must never carry outcome information."""
    with pytest.raises(ValueError, match="quality fields"):
        write_budget(tmp_path, {"T1": {"total_tokens": 100.0, "success": 1.0}}, 50.0)


def test_budget_accepts_cost_only(tmp_path):
    p = write_budget(tmp_path, {"T1": {"total_tokens": 100.0, "llm_calls": 3.0}}, 50.0)
    assert p.exists()


# ------------------------- complexity -------------------------------

def test_complexity_score_applies_weights(cfg):
    factors = {"n_tool_calls": 2, "n_evidence_sources": 1,
               "dependency_depth": 1, "verification_required": 0,
               "synthesis_required": 1}
    # 2*1.0 + 1*1.0 + 1*1.5 + 0*2.0 + 1*2.0 = 6.5
    assert score_complexity(factors, cfg) == pytest.approx(6.5)


def test_complexity_missing_factor_raises(cfg):
    with pytest.raises(ValueError, match="missing complexity factors"):
        score_complexity({"n_tool_calls": 1}, cfg)


def test_tier_boundaries(cfg):
    assert assign_tier(6.25, cfg) == 1
    assert assign_tier(6.26, cfg) == 2
    assert assign_tier(13.0, cfg) == 2
    assert assign_tier(13.1, cfg) == 3


# ------------------------- quality metrics --------------------------

def test_exact_match_none_when_no_gold():
    assert exact_match("anything", None) is None


def test_exact_match_basic():
    assert exact_match("The answer is 42.", "42") == 1
    assert exact_match("The answer is 41.", "42") == 0
    assert exact_match(None, "42") == 0


def test_gold_fact_coverage():
    assert gold_fact_coverage("contains alpha and beta", ["alpha", "beta"]) == 1.0
    assert gold_fact_coverage("contains alpha only", ["alpha", "beta"]) == 0.5
    assert gold_fact_coverage("x", []) is None


def test_failure_classification_never_silent():
    rec = {"failure_type": None, "final_answer": None,
           "stopped_reason": "max_iterations"}
    assert classify_failure(rec) == "timeout"
    rec2 = {"failure_type": None, "final_answer": "ok", "stopped_reason": "final_answer"}
    assert classify_failure(rec2) is None


# ------------------------- statistics -------------------------------

def test_paired_wilcoxon_detects_difference():
    x = [0.9, 0.8, 0.95, 0.85, 0.9, 0.88, 0.92, 0.87]
    y = [0.5, 0.4, 0.55, 0.45, 0.5, 0.48, 0.52, 0.47]
    r = paired_wilcoxon(x, y, "t", mid=0.05)
    assert r.n_pairs == 8
    assert r.median_difference > 0
    assert r.practically_significant is True


def test_paired_wilcoxon_handles_too_few_pairs():
    r = paired_wilcoxon([1.0], [0.5], "t")
    assert r.median_difference is None
    assert "fewer than 3" in r.note


def test_paired_wilcoxon_ignores_incomplete_pairs():
    r = paired_wilcoxon([1.0, None, 0.5, 0.6], [0.5, 0.2, None, 0.3], "t")
    assert r.n_pairs == 2  # only two complete pairs


def test_rank_biserial_sign():
    assert rank_biserial([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert rank_biserial([-1.0, -2.0, -3.0]) == pytest.approx(-1.0)


def test_bootstrap_ci_brackets_median():
    vals = [0.1] * 20 + [0.2] * 20
    lo, hi = bootstrap_ci(vals, n_boot=500)
    assert lo is not None and lo <= 0.2 and hi >= 0.1


def test_holm_is_monotone_and_conservative():
    adj = holm([0.01, 0.02, 0.03])
    assert adj[0] >= 0.01
    assert adj[0] <= adj[1] <= adj[2]


def test_bh_less_conservative_than_holm():
    ps = [0.01, 0.02, 0.03]
    assert benjamini_hochberg(ps)[-1] <= holm(ps)[-1]


def test_corrections_preserve_none():
    assert holm([None, 0.01])[0] is None
    assert benjamini_hochberg([None, 0.01])[0] is None


# ------------------------- tools and ids ----------------------------

def test_calculator_rejects_code_injection():
    assert _calculator("__import__('os').system('ls')").startswith("ERROR")


def test_calculator_arithmetic():
    assert _calculator("6*7") == "42"


def test_experiment_ids_are_unique():
    a = make_experiment_id("A", 1, 1, "T1")
    b = make_experiment_id("A", 1, 1, "T1")
    assert a != b
