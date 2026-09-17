"""Generate the final PDF report: full A/B/C/D comparison.

    python make_report.py
    python make_report.py --out results/reports/my_report.pdf

Reads only real measured data:
    results/processed/evaluated.jsonl
    results/statistics/statistics_report.json
    results/statistics/mechanism_report.json
    results/figures/*.png

Every value that has not been measured renders as "no data" or "--".
Nothing in this file estimates, simulates or interpolates a result. If a
section has no data behind it, the report says so rather than showing a
plausible-looking number.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import load_config, setup_logging   # noqa: E402

LOG = logging.getLogger("make_report")

from reportlab.lib import colors                                  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY             # noqa: E402
from reportlab.lib.pagesizes import A4                            # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm                                # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)        # noqa: E402

CONDITIONS = ["A", "B", "C", "D"]
LABEL = {
    "A": "A - Single-agent ReAct",
    "B": "B - Compute-matched Best-of-N",
    "C": "C - Pipeline (LangGraph)",
    "D": "D - Pipeline (CrewAI)",
}

INK = colors.HexColor("#1a1a1a")
RULE = colors.HexColor("#999999")
HEAD_BG = colors.HexColor("#33383d")
ALT_BG = colors.HexColor("#f2f4f6")
WARN = colors.HexColor("#b03030")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _styles() -> Dict[str, ParagraphStyle]:
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=17,
                                leading=21, textColor=INK, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=10,
                              leading=14, alignment=TA_CENTER,
                              textColor=colors.HexColor("#555555")),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13,
                             leading=16, textColor=INK, spaceBefore=14,
                             spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11,
                             leading=14, textColor=INK, spaceBefore=10,
                             spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5,
                               leading=13.5, alignment=TA_JUSTIFY,
                               textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=8,
                                leading=11, textColor=colors.HexColor("#555555")),
        "warn": ParagraphStyle("w", parent=ss["Normal"], fontSize=9,
                               leading=12.5, textColor=WARN, spaceAfter=6),
        "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=8,
                              leading=11, alignment=TA_CENTER,
                              textColor=colors.HexColor("#555555"),
                              spaceBefore=2, spaceAfter=10),
    }


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "--"
        return f"{v:.{nd}f}"
    return str(v)


def _table(data: List[List[str]], widths: Optional[List[float]] = None,
           font: float = 8.0, align_right_from: int = 1) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), ALT_BG))
    t.setStyle(TableStyle(style))
    return t


def _agg(rows: List[Dict[str, Any]], cond: str, key: str,
         timing: bool = False) -> Optional[float]:
    vals = []
    for r in rows:
        if r["condition"] != cond or r.get(key) is None:
            continue
        if timing and not r.get("timing_valid", True):
            continue   # parallel-mode runs are contended; excluded
        vals.append(float(r[key]))
    return float(np.mean(vals)) if vals else None


def _median(rows: List[Dict[str, Any]], cond: str, key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows
            if r["condition"] == cond and r.get(key) is not None]
    return float(np.median(vals)) if vals else None


# ---------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------

def sec_header(story, S, cfg, rows, out_name: str) -> None:
    story.append(Paragraph(
        "Single-Agent versus Multi-Agent LLM Orchestration", S["title"]))
    story.append(Paragraph(
        "Compute-Matched, Framework-Controlled Benchmark Report", S["sub"]))
    story.append(Spacer(1, 4 * mm))

    hw = next((r.get("hardware") for r in rows if r.get("hardware")), {}) or {}
    meta = [
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Backbone model", str(cfg.get("model.name"))],
        ["Quantisation", str(cfg.get("model.quantization"))],
        ["Temperature / seed", f"{cfg.get('model.temperature')} / {cfg.get('model.seed')}"],
        ["Context window", str(cfg.get("model.num_ctx"))],
        ["Judge model", str(cfg.get("judge.name"))],
        ["Config hash", str(cfg.config_hash)],
        ["Host OS", f"{hw.get('os', '--')} {hw.get('os_release', '')}"],
        ["System RAM (MB)", _fmt(hw.get("total_ram_mb"), 0)],
        ["GPU", str(hw.get("gpu_name") or "none detected")],
        ["Records analysed", str(len(rows))],
    ]
    story.append(_table([["Field", "Value"]] + meta,
                        widths=[45 * mm, 110 * mm], font=8.5))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This report is generated directly from measured experimental "
        "records. Any quantity that was not measured is shown as \"--\". "
        "No value in this document is estimated, simulated or interpolated.",
        S["small"]))


def sec_conditions(story, S, rows) -> None:
    story.append(Paragraph("1. Conditions and coverage", S["h1"]))
    story.append(Paragraph(
        "Four conditions share the same backbone model, quantisation, "
        "runtime, hardware, tool inventory, task instances and decoding "
        "settings. Only the execution condition varies. The D-versus-B "
        "contrast isolates the benefit of collaboration at equal compute; "
        "C-versus-A isolates architecture within one framework; D-versus-C "
        "isolates the framework at fixed architecture.", S["body"]))

    data = [["Condition", "Records", "Tasks", "Answered", "Failed", "Tiers 1/2/3"]]
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        if not sub:
            data.append([LABEL[c], "0", "0", "--", "--", "--"])
            continue
        tiers = Counter(r.get("complexity_tier") for r in sub)
        ok = sum(1 for r in sub if r.get("success") == 1
                 or (r.get("success") is None and r.get("quality") is not None))
        failed = sum(1 for r in sub if r.get("failure_type"))
        data.append([LABEL[c], str(len(sub)),
                     str(len({r["task_id"] for r in sub})), str(ok), str(failed),
                     f"{tiers.get(1,0)}/{tiers.get(2,0)}/{tiers.get(3,0)}"])
    story.append(_table(data, widths=[52 * mm, 20 * mm, 18 * mm, 22 * mm,
                                      20 * mm, 25 * mm]))
    story.append(Paragraph("Table 1. Coverage by condition.", S["cap"]))

    missing = [c for c in CONDITIONS if not any(r["condition"] == c for r in rows)]
    if missing:
        story.append(Paragraph(
            f"Conditions {', '.join(missing)} have no records. Contrasts that "
            "depend on them cannot be computed and are omitted below.",
            S["warn"]))


def sec_comparison(story, S, rows) -> None:
    story.append(Paragraph("2. Head-to-head comparison (A / B / C / D)", S["h1"]))
    story.append(Paragraph(
        "Means across all tasks. Timing and memory figures exclude any run "
        "executed in parallel mode, because concurrent conditions contend "
        "for CPU, GPU and memory.", S["body"]))

    data = [["Metric", "A", "B", "C", "D"]]
    metrics = [
        ("Quality (mean)", "quality", 3, False),
        ("Success rate", "success", 3, False),
        ("Total tokens", "total_tokens", 0, False),
        ("Input tokens", "input_tokens", 0, False),
        ("Output tokens", "output_tokens", 0, False),
        ("LLM calls", "llm_calls", 2, False),
        ("Tool calls", "tool_calls", 2, False),
        ("Retries", "retries", 2, False),
        ("Latency (s)", "latency_seconds", 1, True),
        ("TTFT (s)", "ttft_seconds", 2, True),
        ("Peak RAM (MB)", "peak_ram_mb", 0, True),
        ("VRAM (MB)", "peak_gpu_memory_mb", 0, True),
    ]
    for label, key, nd, timing in metrics:
        row = [label]
        for c in CONDITIONS:
            row.append(_fmt(_agg(rows, c, key, timing), nd))
        data.append(row)
    story.append(_table(data, widths=[40 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm]))
    story.append(Paragraph("Table 2. Mean values by condition.", S["cap"]))

    # efficiency ratios -- computed only where both parts exist
    story.append(Paragraph("2.1 Efficiency ratios", S["h2"]))
    data = [["Ratio", "A", "B", "C", "D"]]
    for label, num, den, scale in (
            ("Quality per 1k tokens", "quality", "total_tokens", 1000.0),
            ("Quality per second", "quality", "latency_seconds", 1.0),
            ("Quality per LLM call", "quality", "llm_calls", 1.0)):
        row = [label]
        for c in CONDITIONS:
            q = _agg(rows, c, num)
            d = _agg(rows, c, den, timing=(den == "latency_seconds"))
            row.append(_fmt(q / d * scale, 4) if q is not None and d else "--")
        data.append(row)
    story.append(_table(data, widths=[40 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm]))
    story.append(Paragraph(
        "Table 3. Efficiency ratios. These are descriptive; the inferential "
        "comparison is the paired analysis in section 4.", S["cap"]))


def sec_tier(story, S, rows) -> None:
    story.append(Paragraph("3. Results by complexity tier", S["h1"]))
    for metric, label, nd, timing in (("quality", "Quality", 3, False),
                                      ("total_tokens", "Total tokens", 0, False),
                                      ("latency_seconds", "Latency (s)", 1, True)):
        data = [[f"{label} by tier", "A", "B", "C", "D"]]
        any_data = False
        for tier in (1, 2, 3):
            sub = [r for r in rows if r.get("complexity_tier") == tier]
            row = [f"Tier {tier}"]
            for c in CONDITIONS:
                v = _agg(sub, c, metric, timing)
                if v is not None:
                    any_data = True
                row.append(_fmt(v, nd))
            data.append(row)
        if any_data:
            story.append(_table(data, widths=[40 * mm, 27 * mm, 27 * mm,
                                              27 * mm, 27 * mm]))
            story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Tables 4-6. Tier-wise means. Tier assignment follows Eq. (1) with "
        "pre-registered weights and cut points.", S["cap"]))


def sec_stats(story, S, stats) -> None:
    story.append(PageBreak())
    story.append(Paragraph("4. Statistical analysis", S["h1"]))
    if not stats:
        story.append(Paragraph(
            "No statistics report found. Run: python analyze.py", S["warn"]))
        return

    story.append(Paragraph("4.1 Primary endpoint: compute-matched quality "
                           "difference", S["h2"]))
    story.append(Paragraph(
        "The primary quantity is the paired difference in quality between "
        "Condition D and Condition B at matched compute. A value of zero or "
        "below is a valid, reportable result, not a failed experiment.",
        S["body"]))

    dm = stats.get("delta_matched", {})
    data = [["Tier", "n pairs", "Median delta", "95% CI", "Effect size",
             "p (corrected)", "MID met"]]
    for tier in ("1", "2", "3", "all"):
        r = dm.get(f"delta_matched_tier_{tier}", {})
        ci = ("--" if r.get("ci_low") is None
              else f"[{_fmt(r.get('ci_low'), 3)}, {_fmt(r.get('ci_high'), 3)}]")
        data.append([f"Tier {tier}", str(r.get("n_pairs", 0)),
                     _fmt(r.get("median_difference"), 4), ci,
                     _fmt(r.get("effect_size"), 3),
                     _fmt(r.get("p_corrected") or r.get("p_value"), 4),
                     _fmt(r.get("practically_significant"))])
    story.append(_table(data, widths=[18 * mm, 16 * mm, 24 * mm, 32 * mm,
                                      22 * mm, 25 * mm, 18 * mm]))
    story.append(Paragraph(
        "Table 7. Wilcoxon signed-rank with bootstrap confidence intervals. "
        "Primary hypotheses are Holm-corrected.", S["cap"]))

    story.append(Paragraph("4.2 Overhead decomposition", S["h2"]))
    story.append(Paragraph(
        "Architecture-attributable overhead is C minus A, measured within a "
        "single framework. Framework-attributable overhead is D minus C, "
        "measured at fixed architecture. These are operational "
        "decompositions of measured differences.", S["body"]))
    tax = stats.get("coordination_tax", {})
    data = [["Cost dimension", "O_arch (C-A)", "%", "O_fw (D-C)", "%"]]
    for metric, label in (("latency_seconds", "Latency (s)"),
                          ("total_tokens", "Total tokens"),
                          ("input_tokens", "Input tokens"),
                          ("output_tokens", "Output tokens"),
                          ("llm_calls", "LLM calls"),
                          ("tool_calls", "Tool calls")):
        a = tax.get(f"O_arch_C_minus_A::{metric}", {})
        f = tax.get(f"O_fw_D_minus_C::{metric}", {})
        data.append([label, _fmt(a.get("median_difference"), 1),
                     _fmt(a.get("relative_percent"), 1),
                     _fmt(f.get("median_difference"), 1),
                     _fmt(f.get("relative_percent"), 1)])
    story.append(_table(data, widths=[38 * mm, 30 * mm, 22 * mm, 30 * mm,
                                      22 * mm]))
    story.append(Paragraph("Table 8. Overhead decomposition (RQ2).", S["cap"]))

    inter = (stats.get("tier_interaction") or {}).get("mixed_effects") or {}
    if inter.get("params"):
        story.append(Paragraph("4.3 Architecture-by-complexity interaction (H1)",
                               S["h2"]))
        data = [["Term", "Coefficient", "p-value"]]
        for k, v in inter["params"].items():
            data.append([str(k), _fmt(v, 5),
                         _fmt((inter.get("pvalues") or {}).get(k), 5)])
        story.append(_table(data, widths=[50 * mm, 35 * mm, 35 * mm]))
        story.append(Paragraph(
            f"Table 9. {inter.get('formula', '')} "
            f"{inter.get('note', '')}", S["cap"]))
    elif inter.get("note") or inter.get("error"):
        story.append(Paragraph("4.3 Architecture-by-complexity interaction (H1)",
                               S["h2"]))
        story.append(Paragraph(
            f"Model not fitted: {inter.get('note') or inter.get('error')}",
            S["warn"]))


def sec_mechanism(story, S, mech) -> None:
    story.append(Paragraph("5. Mechanism analysis", S["h1"]))
    if not mech:
        story.append(Paragraph(
            "No mechanism report found. Run: python analyze.py", S["warn"]))
        return

    fs = mech.get("failure_stages", {})
    story.append(Paragraph("5.1 Where multi-agent runs fail", S["h2"]))
    story.append(Paragraph(
        f"{fs.get('n_failed_or_degraded', 0)} failed or degraded multi-agent "
        "runs in tiers 2-3, attributed to originating stage and MAST "
        "category. Runs that cannot be attributed from the logs are counted "
        "as unattributed rather than assigned by guesswork.", S["body"]))
    data = [["Originating stage", "System design", "Misalignment",
             "Verification", "Unattributed"]]
    for stage, cats in (fs.get("matrix") or {}).items():
        data.append([stage.capitalize(),
                     str(cats.get("system_design", 0)),
                     str(cats.get("inter_agent_misalignment", 0)),
                     str(cats.get("task_verification", 0)),
                     str(cats.get("unattributed", 0))])
    if len(data) > 1:
        story.append(_table(data, widths=[38 * mm, 30 * mm, 28 * mm, 28 * mm,
                                          28 * mm]))
        story.append(Paragraph("Table 10. Failure stage distribution.", S["cap"]))

    story.append(Paragraph("5.2 Context growth (H3) and role bleed (H4)",
                           S["h2"]))
    h3 = mech.get("h3_context_growth", {})
    h4 = mech.get("h4_bleed_failure", {})
    data = [["Hypothesis", "n", "Statistic", "p-value", "Reading"]]
    data.append(["H3 context vs quality", str(h3.get("n", 0)),
                 _fmt(h3.get("spearman_rho"), 4), _fmt(h3.get("p_value"), 5),
                 str(h3.get("direction") or h3.get("note") or "--")[:38]])
    data.append(["H4 bleed vs failure", str(h4.get("n", 0)),
                 _fmt(h4.get("odds_ratio"), 3), _fmt(h4.get("p_value"), 5),
                 str(h4.get("note") or "--")[:38]])
    story.append(_table(data, widths=[36 * mm, 14 * mm, 24 * mm, 24 * mm,
                                      54 * mm]))
    story.append(Paragraph("Table 11. Mechanism hypotheses.", S["cap"]))

    rf = mech.get("role_fidelity", {})
    if any(k in rf for k in ("planner", "researcher", "writer")):
        story.append(Paragraph("5.3 Role fidelity", S["h2"]))
        data = [["Role", "RFS T1", "RFS T2", "RFS T3", "Bleed rate (%)", "n"]]
        for role in ("planner", "researcher", "writer"):
            r = rf.get(role, {})
            data.append([role.capitalize(), _fmt(r.get("tier_1_rfs"), 1),
                         _fmt(r.get("tier_2_rfs"), 1), _fmt(r.get("tier_3_rfs"), 1),
                         _fmt(r.get("bleed_rate_percent"), 1),
                         str(r.get("bleed_n", 0))])
        story.append(_table(data, widths=[28 * mm, 24 * mm, 24 * mm, 24 * mm,
                                          32 * mm, 16 * mm]))
        val = rf.get("_validation", {})
        story.append(Paragraph(
            f"Table 12. Role fidelity. Spearman vs CRAS: "
            f"{_fmt(val.get('spearman_vs_cras'))}. Human agreement: "
            f"{_fmt(val.get('human_agreement_kappa_w'))}.", S["cap"]))
        if val.get("spearman_vs_cras") is None:
            story.append(Paragraph(
                "RFS is NOT yet validated. Both validation statistics are "
                "null because no annotated subsample exists. Do not report "
                "RFS as a validated measure until they are computed.",
                S["warn"]))


def sec_figures(story, S, fig_dir: Path) -> None:
    story.append(PageBreak())
    story.append(Paragraph("6. Figures", S["h1"]))
    wanted = [
        ("fig_delta_matched.png",
         "Figure 1. Compute-matched quality difference by tier, with "
         "bootstrap confidence intervals. Zero line marks parity."),
        ("fig_overhead_decomposition.png",
         "Figure 2. Architecture- versus framework-attributable overhead."),
        ("fig_quality_by_tier.png",
         "Figure 3. Quality by tier and condition."),
        ("fig_tokens_by_tier.png",
         "Figure 4. Token consumption by tier and condition."),
        ("fig_latency_by_tier.png",
         "Figure 5. Latency by tier and condition."),
        ("fig_resources_by_tier.png",
         "Figure 6. Peak resident memory by tier and condition."),
        ("fig_context_vs_quality.png",
         "Figure 7. Per-role quality against cumulative input context (H3)."),
        ("fig_break_even_surface.png",
         "Figure 8. Break-even surface over complexity and exchange rate."),
    ]
    shown = 0
    for name, caption in wanted:
        p = Path(fig_dir) / name
        if not p.exists():
            continue
        try:
            from PIL import Image as PILImage
            w, h = PILImage.open(p).size
            ratio = h / w
        except Exception:
            ratio = 0.55
        width = 150 * mm
        story.append(Image(str(p), width=width, height=width * ratio))
        story.append(Paragraph(caption, S["cap"]))
        shown += 1
        if shown % 2 == 0:
            story.append(PageBreak())
    if shown == 0:
        story.append(Paragraph(
            "No figures found. Run: python make_figures.py", S["warn"]))


def sec_per_task(story, S, rows, limit: int = 40) -> None:
    story.append(PageBreak())
    story.append(Paragraph("7. Per-task comparison", S["h1"]))
    story.append(Paragraph(
        "Quality and total tokens for every task under each condition. "
        "Tasks are ordered by tier then id. \"--\" means the condition "
        "produced no measurement for that task.", S["body"]))

    by_task: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_task[r["task_id"]][r["condition"]] = r

    ordered = sorted(by_task.items(),
                     key=lambda kv: (min((v.get("complexity_tier") or 9)
                                         for v in kv[1].values()), kv[0]))
    data = [["Task", "Tier", "Q(A)", "Q(B)", "Q(C)", "Q(D)",
             "tok A", "tok B", "tok C", "tok D"]]
    for task_id, per in ordered[:limit]:
        tier = next((v.get("complexity_tier") for v in per.values()
                     if v.get("complexity_tier")), "--")
        row = [task_id, str(tier)]
        for c in CONDITIONS:
            row.append(_fmt((per.get(c) or {}).get("quality"), 2))
        for c in CONDITIONS:
            row.append(_fmt((per.get(c) or {}).get("total_tokens"), 0))
        data.append(row)

    story.append(_table(data, widths=[22 * mm, 12 * mm] + [14 * mm] * 4
                        + [19 * mm] * 4, font=7.0))
    caption = f"Table 13. Per-task results ({min(len(ordered), limit)} of " \
              f"{len(ordered)} tasks shown)."
    story.append(Paragraph(caption, S["cap"]))
    if len(ordered) > limit:
        story.append(Paragraph(
            "The complete per-task record is in "
            "results/processed/evaluated.jsonl.", S["small"]))


def sec_integrity(story, S, rows, stats, mech, cfg) -> None:
    story.append(PageBreak())
    story.append(Paragraph("8. Data integrity and limitations", S["h1"]))

    n_null_tok = sum(1 for r in rows if r.get("total_tokens") is None)
    n_parallel = sum(1 for r in rows if r.get("timing_valid") is False)
    n_judge = sum(1 for r in rows if r.get("judge_rubric") is not None)
    n_gold = sum(1 for r in rows if r.get("exact_match") is not None
                 or r.get("gold_fact_coverage") is not None)
    failures = Counter(r.get("failure_type") for r in rows if r.get("failure_type"))

    data = [["Check", "Value", "Consequence if non-zero"]]
    data.append(["Records with null token counts", str(n_null_tok),
                 "excluded from cost analysis"])
    data.append(["Records run in parallel mode", str(n_parallel),
                 "excluded from timing and memory"])
    data.append(["Records with deterministic gold data", str(n_gold),
                 "higher is better"])
    data.append(["Records scored by LLM judge", str(n_judge),
                 "judge-only quality is weaker evidence"])
    data.append(["Total failures recorded", str(sum(failures.values())),
                 "retained, never discarded"])
    story.append(_table(data, widths=[62 * mm, 24 * mm, 68 * mm]))
    story.append(Paragraph("Table 14. Integrity checks.", S["cap"]))

    if failures:
        data = [["Failure type", "Count"]]
        for k, v in failures.most_common():
            data.append([str(k), str(v)])
        story.append(_table(data, widths=[60 * mm, 24 * mm]))
        story.append(Paragraph("Table 15. Failure types.", S["cap"]))

    story.append(Paragraph("Limitations", S["h2"]))
    limits = [
        "Single backbone model. Conclusions are scoped to small local models "
        "of this class and are not extended to LLM agents generally.",
        "Single hardware platform. The memory budget is a deployment "
        "scenario, not a controlled variable.",
        "Any crossover location is hardware- and model-specific; only its "
        "existence and shape generalise.",
        "Role fidelity is a lightweight proxy and is not a validated metric "
        "until CRAS correlation and human agreement are computed.",
    ]
    if n_gold == 0:
        limits.insert(0, "No deterministic gold data was available, so every "
                         "quality figure rests on a single LLM judge. This is "
                         "the weakest configuration of the quality metric.")
    for text in limits:
        story.append(Paragraph(f"\u2022 {text}", S["body"]))


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

def build_report(cfg, out_path: Path) -> int:
    out_dir = Path(cfg.get("experiment.output_directory", "results"))
    ev = out_dir / "processed" / "evaluated.jsonl"
    if not ev.exists():
        print(f"No evaluated data at {ev}.")
        print("Run this first:  python evaluate.py")
        return 2

    rows = [json.loads(l) for l in ev.open(encoding="utf-8") if l.strip()]
    if not rows:
        print("evaluated.jsonl is empty. Run the experiment first.")
        return 2

    def _read(p: Path) -> Dict[str, Any]:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    stats = _read(out_dir / "statistics" / "statistics_report.json")
    mech = _read(out_dir / "statistics" / "mechanism_report.json")

    S = _styles()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Single vs Multi-Agent Benchmark Report")

    story: List[Any] = []
    sec_header(story, S, cfg, rows, out_path.name)
    sec_conditions(story, S, rows)
    sec_comparison(story, S, rows)
    sec_tier(story, S, rows)
    sec_stats(story, S, stats)
    sec_mechanism(story, S, mech)
    sec_figures(story, S, out_dir / "figures")
    sec_per_task(story, S, rows)
    sec_integrity(story, S, rows, stats, mech, cfg)

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(18 * mm, 10 * mm,
                          "Generated from measured data. Unmeasured values "
                          "appear as \"--\".")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Report written -> {out_path}")
    print(f"  {len(rows)} evaluated records, "
          f"{len({r['task_id'] for r in rows})} tasks, "
          f"{len({r['condition'] for r in rows})} conditions")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the PDF report")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("experiment.output_directory", "results"))
    setup_logging(cfg.get("logging.level", "INFO"), False)

    out = Path(args.out) if args.out else (
        out_dir / "reports" /
        f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    return build_report(cfg, out)


if __name__ == "__main__":
    raise SystemExit(main())
