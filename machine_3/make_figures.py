"""Regenerate every figure and table from the raw evaluated data.

    python make_figures.py

No experimental value is ever hardcoded here: delete results/figures and
results/tables and this script rebuilds them from evaluated.jsonl.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import load_config, setup_logging     # noqa: E402
from visualization.render import render_all            # noqa: E402

LOG = logging.getLogger("make_figures")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate figures and tables")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg.get("experiment.output_directory", "results"))
    setup_logging(cfg.get("logging.level", "INFO"), True, out / "figures.log")

    render_all(cfg, out / "processed", out / "statistics" / "statistics_report.json",
               out / "figures", out / "tables")
    LOG.info("Figures: %s", out / "figures")
    LOG.info("Tables:  %s", out / "tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
