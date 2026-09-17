"""Run the statistical pipeline.

    python analyze.py

Reads results/processed/evaluated.jsonl and writes
results/statistics/statistics_report.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.mechanism import run_mechanism_analysis    # noqa: E402
from analysis.statistics_pipeline import run_all          # noqa: E402
from core.config import load_config, setup_logging        # noqa: E402

LOG = logging.getLogger("analyze")


def main() -> int:
    ap = argparse.ArgumentParser(description="Statistical analysis")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("experiment.output_directory", "results"))
    setup_logging(cfg.get("logging.level", "INFO"), True, out_dir / "analyze.log")

    try:
        report = run_all(cfg, out_dir / "processed", out_dir / "statistics")
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 2

    raw_dir = Path(cfg.get("logging.raw_directory", "results/raw"))
    mech = run_mechanism_analysis(cfg, raw_dir, out_dir / "processed",
                                  out_dir / "statistics")

    print("\n" + "=" * 62)
    print("COMPUTE-MATCHED QUALITY DELTA   Q(D) - Q(B)")
    print("=" * 62)
    for tier in ("1", "2", "3", "all"):
        r = report["delta_matched"].get(f"delta_matched_tier_{tier}", {})
        n = r.get("n_pairs", 0)
        med = r.get("median_difference")
        lo, hi = r.get("ci_low"), r.get("ci_high")
        p = r.get("p_corrected", r.get("p_value"))
        if med is None:
            print(f"Tier {tier:>3}: n={n:<4} insufficient paired data")
        else:
            ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "[--, --]"
            ps = f"{p:.4f}" if p is not None else "--"
            print(f"Tier {tier:>3}: n={n:<4} median={med:+.4f}  CI={ci}  p={ps}")
    print("\nNote: a negative or zero delta is a valid scientific result.")

    fs = mech.get("failure_stages", {})
    print("\n" + "=" * 62)
    print("MECHANISM ANALYSIS")
    print("=" * 62)
    print(f"failed or degraded multi-agent runs (tiers 2-3): "
          f"{fs.get('n_failed_or_degraded', 0)}")
    for stage, cats in (fs.get("matrix") or {}).items():
        total = sum(cats.values())
        if total:
            print(f"  {stage:<14} {total:>3}  {cats}")
    h3 = mech.get("h3_context_growth", {})
    if h3.get("spearman_rho") is not None:
        print(f"H3 context vs quality: rho={h3['spearman_rho']} "
              f"p={h3.get('p_value')} ({h3.get('direction')})")
    else:
        print(f"H3: {h3.get('note', 'not tested')}")
    h4 = mech.get("h4_bleed_failure", {})
    if h4.get("p_value") is not None:
        print(f"H4 bleed vs failure: OR={h4.get('odds_ratio')} p={h4['p_value']}")
    else:
        print(f"H4: {h4.get('note', 'not tested')}")
    print(f"\nFull report: {out_dir / 'statistics' / 'statistics_report.json'}")
    print("Next: python make_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
