"""Environment and reproducibility verification.

    python reproduce.py

Checks that another researcher's machine can reproduce this study, and
prints the exact environment record that belongs in the paper's
configuration table.
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import detect_hardware, load_config   # noqa: E402

REQUIRED = ["yaml", "requests", "psutil", "numpy"]
OPTIONAL = ["scipy", "pandas", "statsmodels", "matplotlib", "langgraph",
            "crewai", "pytest"]


def main() -> int:
    print("=" * 62)
    print("REPRODUCIBILITY REPORT")
    print("=" * 62)
    problems = 0

    print(f"\nPython: {platform.python_version()}")
    if sys.version_info < (3, 11):
        print("  WARNING: Python 3.11+ is expected.")

    print("\nRequired packages:")
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"  OK       {mod:<14} {getattr(m, '__version__', 'n/a')}")
        except ImportError:
            problems += 1
            print(f"  MISSING  {mod}")

    print("\nOptional packages:")
    for mod in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            print(f"  OK       {mod:<14} {getattr(m, '__version__', 'n/a')}")
        except ImportError:
            print(f"  absent   {mod}")

    print("\nHardware:")
    hw = detect_hardware()
    for k, v in hw.items():
        print(f"  {k:<22} {v}")

    print("\nConfiguration:")
    try:
        cfg = load_config("config.yaml")
        print(f"  config hash            {cfg.config_hash}")
        for key in ("model.name", "model.quantization", "model.temperature",
                    "model.seed", "model.num_ctx", "judge.name",
                    "experiment.max_iterations",
                    "compute_matching.token_tolerance_percent"):
            print(f"  {key:<22} {cfg.get(key)}")
    except Exception as exc:
        problems += 1
        print(f"  ERROR loading config: {exc}")

    print("\nData files:")
    for p in ("tasks/tasks.json", "tasks/corpus.json"):
        path = Path(p)
        print(f"  {'OK      ' if path.exists() else 'MISSING '} {p}")
        if not path.exists():
            problems += 1

    record = {"hardware": hw, "python": platform.python_version()}
    Path("results").mkdir(exist_ok=True)
    Path("results/environment.json").write_text(json.dumps(record, indent=2),
                                                encoding="utf-8")
    print("\nEnvironment record written to results/environment.json")
    print(f"\n{'PASS' if problems == 0 else f'{problems} PROBLEM(S) FOUND'}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
