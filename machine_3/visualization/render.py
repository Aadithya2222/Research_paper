"""Figures and tables.

Nothing is hardcoded: every figure reads the evaluated records or the
statistics report. If a series has no data, the panel is annotated
"no data" rather than being filled with a plausible-looking line.

Figure widths follow IEEE conventions: 3.5 in single column,
7.16 in double column.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

LOG = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:  # pragma: no cover
    MPL = False
    LOG.warning("matplotlib not installed; figures will be skipped.")

SINGLE_COL, DOUBLE_COL = 3.5, 7.16
CONDITION_ORDER = ["A", "B", "C", "D"]
CONDITION_LABEL = {
    "A": "A: single ReAct",
    "B": "B: compute-matched BoN",
    "C": "C: pipeline (LangGraph)",
    "D": "D: pipeline (CrewAI)",
}


def _style() -> None:
    if not MPL:
        return
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    LOG.info("Figure written: %s.{png,pdf,svg}", out_dir / name)


def _by_tier(rows: List[Dict[str, Any]], cond: str, metric: str
             ) -> Dict[int, List[float]]:
    acc: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    for r in rows:
        if r["condition"] != cond:
            continue
        v = r.get(metric)
        t = r.get("complexity_tier")
        if v is not None and t in acc:
            acc[t].append(float(v))
    return acc


def _mean_ci(vals: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    m = float(np.mean(vals))
    if len(vals) < 2:
        return m, 0.0
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return m, 1.96 * se


def metric_by_tier(rows: List[Dict[str, Any]], metric: str, ylabel: str,
                   out_dir: Path, name: str) -> None:
    """Grouped bar chart of one metric across tiers and conditions."""
    if not MPL:
        return
    _style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.6))
    tiers = [1, 2, 3]
    width = 0.2
    any_data = False
    for i, cond in enumerate(CONDITION_ORDER):
        data = _by_tier(rows, cond, metric)
        means, errs = [], []
        for t in tiers:
            m, e = _mean_ci(data[t])
            means.append(m if m is not None else np.nan)
            errs.append(e if e is not None else 0.0)
            if m is not None:
                any_data = True
        xs = [t + (i - 1.5) * width for t in tiers]
        ax.bar(xs, means, width, yerr=errs, capsize=2,
               label=CONDITION_LABEL[cond])
    ax.set_xticks(tiers)
    ax.set_xticklabels([f"Tier {t}" for t in tiers])
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, frameon=False)
    if not any_data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="red")
    _save(fig, out_dir, name)


def delta_matched_figure(stats: Dict[str, Any], out_dir: Path) -> None:
    """The primary result figure: Delta_matched by tier with CIs."""
    if not MPL:
        return
    _style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    tiers, meds, los, his = [], [], [], []
    for t in (1, 2, 3):
        r = stats.get("delta_matched", {}).get(f"delta_matched_tier_{t}", {})
        if r.get("median_difference") is None:
            continue
        tiers.append(t)
        meds.append(r["median_difference"])
        los.append(r["median_difference"] - (r.get("ci_low") or r["median_difference"]))
        his.append((r.get("ci_high") or r["median_difference"]) - r["median_difference"])
    if tiers:
        ax.errorbar(tiers, meds, yerr=[los, his], fmt="o-", capsize=3, color="black")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="red")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Tier 1", "Tier 2", "Tier 3"])
    ax.set_ylabel(r"$\Delta_{\mathrm{matched}} = Q(D) - Q(B)$")
    _save(fig, out_dir, "fig_delta_matched")


def overhead_figure(stats: Dict[str, Any], out_dir: Path) -> None:
    """Architecture- versus framework-attributable overhead."""
    if not MPL:
        return
    _style()
    metrics = ["latency_seconds", "total_tokens", "llm_calls"]
    tax = stats.get("coordination_tax", {})
    arch = [tax.get(f"O_arch_C_minus_A::{m}", {}).get("median_difference")
            for m in metrics]
    fw = [tax.get(f"O_fw_D_minus_C::{m}", {}).get("median_difference")
          for m in metrics]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.4))
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, [a if a is not None else np.nan for a in arch], 0.36,
           label="architecture (C-A)")
    ax.bar(x + 0.18, [f if f is not None else np.nan for f in fw], 0.36,
           label="framework (D-C)")
    ax.set_xticks(x)
    ax.set_xticklabels(["latency (s)", "total tokens", "LLM calls"])
    ax.set_ylabel("median paired difference")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.legend(frameon=False)
    if all(v is None for v in arch + fw):
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="red")
    _save(fig, out_dir, "fig_overhead_decomposition")


def context_quality_figure(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """H3: does per-run quality track cumulative context (proxied by tokens)?"""
    if not MPL:
        return
    _style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    xs = [r["input_tokens"] for r in rows
          if r["condition"] in ("C", "D") and r.get("input_tokens")
          and r.get("quality") is not None]
    ys = [r["quality"] for r in rows
          if r["condition"] in ("C", "D") and r.get("input_tokens")
          and r.get("quality") is not None]
    if xs:
        ax.scatter(xs, ys, s=8, alpha=0.6, color="black")
        if len(xs) >= 3:
            z = np.polyfit(xs, ys, 1)
            gx = np.linspace(min(xs), max(xs), 50)
            ax.plot(gx, np.poly1d(z)(gx), "r--", linewidth=1)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="red")
    ax.set_xlabel("cumulative input tokens")
    ax.set_ylabel("quality")
    _save(fig, out_dir, "fig_context_vs_quality")


def break_even_figure(stats: Dict[str, Any], out_dir: Path) -> None:
    """Break-even sensitivity surface over tier and exchange rate."""
    if not MPL:
        return
    _style()
    be = stats.get("break_even", {})
    rates = be.get("exchange_rates", [])
    cells = be.get("cells", {})
    grid = np.full((3, len(rates)), np.nan)
    for i, t in enumerate((1, 2, 3)):
        cell = cells.get(f"tier_{t}")
        if not cell:
            continue
        for j, r in enumerate(rates):
            grid[i, j] = 1.0 if cell["preferred"][str(r)] == "D" else 0.0
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    if np.isnan(grid).all():
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="red")
    else:
        ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([str(r) for r in rates], rotation=45)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Tier 1", "Tier 2", "Tier 3"])
    ax.set_xlabel("quality-per-cost exchange rate r")
    ax.set_title("blue = B preferred, red = D preferred")
    _save(fig, out_dir, "fig_break_even_surface")


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------

def _write_table(rows: List[List[str]], header: List[str], out_dir: Path,
                 name: str, caption: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.csv").write_text(
        "\n".join([",".join(header)] + [",".join(r) for r in rows]),
        encoding="utf-8")
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out_dir / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    tex = ["\\begin{table}[!t]", f"\\caption{{{caption}}}",
           f"\\label{{tab:{name}}}", "\\centering", "\\footnotesize",
           "\\begin{tabular}{@{}" + "l" * len(header) + "@{}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_dir / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")
    LOG.info("Table written: %s.{csv,md,tex}", out_dir / name)


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "--"          # honest missing-data marker
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table_overall(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    body: List[List[str]] = []
    for cond in CONDITION_ORDER:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            body.append([cond, "0"] + ["--"] * 5)
            continue
        def m(k: str) -> Optional[float]:
            vals = [r[k] for r in sub if r.get(k) is not None]
            return float(np.mean(vals)) if vals else None
        body.append([cond, str(len(sub)), _fmt(m("quality")),
                     _fmt(m("total_tokens"), 1), _fmt(m("llm_calls"), 2),
                     _fmt(m("latency_seconds"), 2), _fmt(m("peak_ram_mb"), 1)])
    _write_table(body, ["Cond", "n", "Quality", "Tokens", "Calls",
                        "Latency (s)", "Peak RAM (MB)"],
                 out_dir, "table_overall", "Overall results by condition.")


def table_statistics(stats: Dict[str, Any], out_dir: Path) -> None:
    body: List[List[str]] = []
    for t in (1, 2, 3):
        r = stats.get("delta_matched", {}).get(f"delta_matched_tier_{t}", {})
        body.append([f"Tier {t}", str(r.get("n_pairs", 0)),
                     _fmt(r.get("median_difference")),
                     f"[{_fmt(r.get('ci_low'))}, {_fmt(r.get('ci_high'))}]",
                     _fmt(r.get("effect_size")), _fmt(r.get("p_corrected"), 4),
                     _fmt(r.get("practically_significant"))])
    _write_table(body, ["Tier", "n", "Median $\\Delta$", "95\\% CI",
                        "Effect", "p (Holm)", "MID met"],
                 out_dir, "table_delta_matched",
                 "Compute-matched quality difference, $Q(D)-Q(B)$.")


def render_all(cfg, processed_dir: Path, stats_path: Path,
               fig_dir: Path, tab_dir: Path) -> None:
    """Regenerate every figure and table from raw evaluated data."""
    ev = Path(processed_dir) / "evaluated.jsonl"
    rows: List[Dict[str, Any]] = []
    if ev.exists():
        with ev.open("r", encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    stats: Dict[str, Any] = {}
    if Path(stats_path).exists():
        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))

    metric_by_tier(rows, "quality", "quality", fig_dir, "fig_quality_by_tier")
    metric_by_tier(rows, "latency_seconds", "latency (s)", fig_dir,
                   "fig_latency_by_tier")
    metric_by_tier(rows, "total_tokens", "total tokens", fig_dir,
                   "fig_tokens_by_tier")
    metric_by_tier(rows, "peak_ram_mb", "peak RAM (MB)", fig_dir,
                   "fig_resources_by_tier")
    delta_matched_figure(stats, fig_dir)
    overhead_figure(stats, fig_dir)
    context_quality_figure(rows, fig_dir)
    break_even_figure(stats, fig_dir)
    table_overall(rows, tab_dir)
    table_statistics(stats, tab_dir)

    # Every numbered table from the manuscript.
    mech_path = Path(stats_path).parent / "mechanism_report.json"
    mech = (json.loads(mech_path.read_text(encoding="utf-8"))
            if mech_path.exists() else {})
    tasks: List[Dict[str, Any]] = []
    for cand in ("tasks/tasks.json", "tasks/tasks_300.json",
                 "tasks/tasks_150.json"):
        tp = Path(cand)
        if tp.exists():
            tasks = json.loads(tp.read_text(encoding="utf-8"))["tasks"]
            break
    paper_tables(cfg, rows, stats, mech, tasks, tab_dir)


# =====================================================================
# Paper tables 1-11: one generator per numbered table in the manuscript.
# Every value comes from real measured data; missing values render "--".
# =====================================================================

def paper_tables(cfg, rows: List[Dict[str, Any]], stats: Dict[str, Any],
                 mech: Dict[str, Any], tasks: List[Dict[str, Any]],
                 out_dir: Path) -> None:
    """Emit every numbered table from the manuscript."""
    import platform

    # --- Table 1: complexity scoring and tier definition ---------------
    w = cfg.get("complexity.weights", {})
    cuts = cfg.get("complexity.tier_cutpoints", {})
    body = [
        ["f1", "Number of required tool invocations", _fmt(w.get("n_tool_calls"), 2)],
        ["f2", "Number of distinct evidence sources", _fmt(w.get("n_evidence_sources"), 2)],
        ["f3", "Dependency depth (chained steps)", _fmt(w.get("dependency_depth"), 2)],
        ["f4", "Verification required (0/1)", _fmt(w.get("verification_required"), 2)],
        ["f5", "Synthesis required (0/1)", _fmt(w.get("synthesis_required"), 2)],
        ["Tier 1", f"C <= {_fmt(cuts.get('tier1_max'), 2)}", ""],
        ["Tier 2", f"{_fmt(cuts.get('tier1_max'), 2)} < C <= "
                   f"{_fmt(cuts.get('tier2_max'), 2)}", ""],
        ["Tier 3", f"C > {_fmt(cuts.get('tier2_max'), 2)}", ""],
    ]
    _write_table(body, ["Factor", "Operationalisation", "Weight"], out_dir,
                 "table1_complexity", "Complexity scoring and tier definition.")

    # --- Table 2: benchmark composition --------------------------------
    tier_counts = {1: 0, 2: 0, 3: 0}
    adapted = 0
    for t in tasks:
        tier = t.get("complexity_tier")
        if tier in tier_counts:
            tier_counts[tier] += 1
        if t.get("provenance") == "adapted":
            adapted += 1
    n = len(tasks)
    body = [["All domains", str(tier_counts[1]), str(tier_counts[2]),
             str(tier_counts[3]), str(n),
             _fmt(100.0 * adapted / n, 1) if n else "--"]]
    _write_table(body, ["Domain", "Tier 1", "Tier 2", "Tier 3", "Total",
                        "Adapted (\\%)"], out_dir, "table2_benchmark",
                 "Benchmark composition and verification.")

    # --- Table 4: configuration and environment ------------------------
    hw = {}
    for r in rows:
        if r.get("hardware"):
            hw = r["hardware"]
            break
    body = [
        ["Backbone model", str(cfg.get("model.name"))],
        ["Quantisation", str(cfg.get("model.quantization"))],
        ["Runtime", "Ollama"],
        ["CPU", str(hw.get("processor") or "--")],
        ["GPU / VRAM", f"{hw.get('gpu_name') or '--'} / "
                       f"{_fmt(hw.get('gpu_memory_total_mb'), 0)} MB"],
        ["System RAM (MB)", _fmt(hw.get("total_ram_mb"), 0)],
        ["Operating system", f"{hw.get('os','--')} {hw.get('os_release','')}"],
        ["Python version", str(hw.get("python_version") or platform.python_version())],
        ["Temperature", _fmt(cfg.get("model.temperature"), 2)],
        ["Seed", str(cfg.get("model.seed"))],
        ["Context window", str(cfg.get("model.num_ctx"))],
        ["Max iterations", str(cfg.get("experiment.max_iterations"))],
        ["Judge model", str(cfg.get("judge.name"))],
        ["Matching tolerance", f"{cfg.get('compute_matching.token_tolerance_percent')}\\%"],
        ["Config hash", str(cfg.config_hash)],
    ]
    _write_table(body, ["Component", "Specification"], out_dir,
                 "table4_config", "Experimental configuration and environment.")

    # --- Table 5: overall paired results -------------------------------
    tax = stats.get("coordination_tax", {})
    dm = stats.get("delta_matched", {})
    body = []
    r = dm.get("delta_matched_tier_all", {})
    body.append(["D vs. B", "Quality", _fmt(r.get("median_difference")),
                 f"[{_fmt(r.get('ci_low'))}, {_fmt(r.get('ci_high'))}]",
                 _fmt(r.get("effect_size")),
                 _fmt(r.get("p_corrected") or r.get("p_value"), 4)])
    k = "MA_minus_SA_D_minus_A::total_tokens"
    r2 = tax.get(k, {})
    body.append(["D vs. A", "Total tokens", _fmt(r2.get("median_difference"), 1),
                 f"[{_fmt(r2.get('ci_low'), 1)}, {_fmt(r2.get('ci_high'), 1)}]",
                 _fmt(r2.get("effect_size")), _fmt(r2.get("p_value"), 4)])
    _write_table(body, ["Contrast", "Metric", "Median $\\Delta$", "95\\% CI",
                        "Effect size", "p"], out_dir, "table5_overall",
                 "Overall paired results with effect sizes.")

    # --- Table 7: overhead decomposition -------------------------------
    body = []
    for metric, label in (("latency_seconds", "Total latency (s)"),
                          ("total_tokens", "Total tokens"),
                          ("input_tokens", "Input tokens"),
                          ("output_tokens", "Output tokens"),
                          ("llm_calls", "LLM calls"),
                          ("tool_calls", "Tool calls")):
        a = tax.get(f"O_arch_C_minus_A::{metric}", {})
        f = tax.get(f"O_fw_D_minus_C::{metric}", {})
        body.append([label, _fmt(a.get("median_difference"), 1),
                     _fmt(a.get("relative_percent"), 1),
                     _fmt(f.get("median_difference"), 1),
                     _fmt(f.get("relative_percent"), 1)])
    _write_table(body, ["Cost dimension", "$O_{arch}$ (C-A)", "\\%",
                        "$O_{fw}$ (D-C)", "\\%"], out_dir, "table7_overhead",
                 "Architecture- and framework-attributable overhead.")

    # --- Table 8: resource utilisation ---------------------------------
    body = []
    for cond in CONDITION_ORDER:
        sub = [r for r in rows if r["condition"] == cond
               and r.get("timing_valid", True)]
        def m(k: str) -> Optional[float]:
            vals = [x[k] for x in sub if x.get(k) is not None]
            return float(np.mean(vals)) if vals else None
        body.append([cond, _fmt(m("peak_ram_mb"), 1), _fmt(m("mean_ram_mb"), 1),
                     _fmt(m("peak_gpu_memory_mb"), 1), _fmt(m("ttft_seconds"), 2)])
    _write_table(body, ["Condition", "Peak RSS (MB)", "Mean RSS (MB)",
                        "VRAM (MB)", "Mean TTFT (s)"], out_dir,
                 "table8_resources", "Resource utilisation by condition.")

    # --- Table 9: failure stage distribution ---------------------------
    matrix = (mech.get("failure_stages") or {}).get("matrix", {})
    body = []
    for stage in ("planner", "researcher", "writer", "handoff", "unattributed"):
        cell = matrix.get(stage, {})
        body.append([stage.capitalize(),
                     str(cell.get("system_design", 0)),
                     str(cell.get("inter_agent_misalignment", 0)),
                     str(cell.get("task_verification", 0))])
    _write_table(body, ["Originating stage", "System design", "Misalignment",
                        "Verification"], out_dir, "table9_failures",
                 "Failure stage distribution (MAST categories), Tiers 2--3.")

    # --- Table 10: role fidelity ---------------------------------------
    rf = mech.get("role_fidelity", {})
    body = []
    for role in ("planner", "researcher", "writer"):
        row = rf.get(role, {})
        body.append([role.capitalize(),
                     _fmt(row.get("tier_1_rfs"), 1), _fmt(row.get("tier_2_rfs"), 1),
                     _fmt(row.get("tier_3_rfs"), 1),
                     _fmt(row.get("bleed_rate_percent"), 1)])
    val = rf.get("_validation", {})
    body.append(["Spearman vs CRAS", _fmt(val.get("spearman_vs_cras")), "", "", ""])
    body.append(["Human agreement", _fmt(val.get("human_agreement_kappa_w")),
                 "", "", ""])
    _write_table(body, ["Role", "RFS T1", "T2", "T3", "Bleed rate (\\%)"],
                 out_dir, "table10_role_fidelity",
                 "Role fidelity and validation statistics.")

    # --- Table 11: statistical test summary ----------------------------
    inter = (stats.get("tier_interaction") or {}).get("mixed_effects") or {}
    h1_p = (inter.get("pvalues") or {}).get("cond:tier")
    h1_b = (inter.get("params") or {}).get("cond:tier")
    fw = tax.get("O_fw_D_minus_C::total_tokens", {})
    h3 = mech.get("h3_context_growth", {})
    h4 = mech.get("h4_bleed_failure", {})
    mid_q = cfg.get("statistics.minimal_important_difference.quality_points")

    body = [
        ["H1", "Mixed-effects interaction", _fmt(h1_b, 4), _fmt(h1_p, 4),
         "--", "--"],
        ["H2", "Wilcoxon ($O_{fw}$ tokens)", _fmt(fw.get("median_difference"), 1),
         _fmt(fw.get("p_corrected") or fw.get("p_value"), 4),
         _fmt(fw.get("effect_size")), "--"],
        ["H3", "Context--quality (Spearman)", _fmt(h3.get("spearman_rho")),
         _fmt(h3.get("p_value"), 4), "--", "--"],
        ["H4", "Bleed--failure (Fisher, exp.)", _fmt(h4.get("odds_ratio")),
         _fmt(h4.get("p_value"), 4), "--", "--"],
    ]
    _write_table(body, ["Hyp.", "Test", "Statistic", "Corr. p", "Effect",
                        "Practical"], out_dir, "table11_statistics",
                 f"Statistical test summary (MID for quality = {mid_q}).")

    LOG.info("Paper tables 1,2,4,5,7,8,9,10,11 written to %s", out_dir)
