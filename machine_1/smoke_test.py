"""Smoke test: prove the whole pipeline works before spending hours on it.

    python smoke_test.py

Runs ONE task through every available condition. It deliberately uses a
tiny workload so it costs minutes, not hours. It writes to
results_smoke/ so it never contaminates real experimental data.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.runner import run_condition                   # noqa: E402
from benchmark.tasks import load_tasks                       # noqa: E402
from core.config import detect_hardware, load_config, setup_logging  # noqa: E402
from core.llm import client_from_config                      # noqa: E402
from tools.local_tools import build_registry                 # noqa: E402
from validation.checks import validate_config, validate_tasks  # noqa: E402

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def main() -> int:
    print("=" * 62)
    print("SMOKE TEST -- Coordination Tax Benchmark")
    print("=" * 62)
    failures = 0

    # 1. config
    try:
        cfg = load_config("config.yaml")
        print(f"{OK} config.yaml loaded (hash {cfg.config_hash})")
    except Exception as exc:
        print(f"{BAD} config.yaml: {exc}")
        return 1

    cfg.data["experiment"]["output_directory"] = "results_smoke"
    cfg.data["logging"]["raw_directory"] = "results_smoke/raw"
    setup_logging("WARNING", console=True)

    # 2. hardware
    hw = detect_hardware()
    print(f"{OK} hardware: {hw['os']} | RAM {hw['total_ram_mb']} MB | "
          f"GPU {hw['gpu_name'] or 'none detected'}")
    if hw["total_ram_mb"] and hw["total_ram_mb"] < 15000:
        print(f"{WARN} less than ~15 GB RAM detected; an 8B model may swap.")

    # 3. fairness checks
    rep = validate_config(cfg)
    for w in rep.warnings:
        print(f"{WARN} {w}")
    if not rep.ok:
        for e in rep.errors:
            print(f"{BAD} {e}")
        return 1
    print(f"{OK} fairness checks passed")

    # 4. tools
    try:
        reg = build_registry(cfg)
        out, _, ok = reg.call("calculator", "6*7")
        assert ok and out.strip() == "42", out
        print(f"{OK} tools working ({', '.join(reg.names)})")
    except Exception as exc:
        print(f"{BAD} tools: {exc}")
        return 1

    # 5. tasks
    try:
        tasks = load_tasks("tasks/tasks.json", cfg)
        validate_tasks(tasks).raise_if_failed()
        print(f"{OK} {len(tasks)} tasks loaded and validated")
    except Exception as exc:
        print(f"{BAD} tasks: {exc}")
        return 1

    # 6. Ollama
    client = client_from_config(cfg)
    if not client.is_available():
        print(f"{BAD} Ollama unreachable at {client.host}. Start it: ollama serve")
        return 1
    print(f"{OK} Ollama reachable at {client.host}")
    if not client.model_present():
        print(f"{BAD} model '{client.model}' missing. Run: ollama pull {client.model}")
        return 1
    print(f"{OK} model '{client.model}' available")

    judge = None
    if cfg.get("judge.enabled", True):
        judge = client_from_config(cfg, judge=True)
        if judge.model_present():
            print(f"{OK} judge model '{judge.model}' available")
        else:
            print(f"{WARN} judge model '{judge.model}' missing "
                  f"(run: ollama pull {judge.model})")
            judge = None

    # 7. frameworks
    try:
        import langgraph  # noqa: F401
        print(f"{OK} langgraph importable")
    except Exception:
        print(f"{WARN} langgraph not installed; condition C uses the native runner")
    crewai_ok = True
    try:
        import crewai  # noqa: F401
        print(f"{OK} crewai importable")
    except Exception:
        crewai_ok = False
        print(f"{WARN} crewai not installed; condition D will be skipped "
              f"(pip install crewai)")

    # 8. one task through each condition
    one = tasks[:1]
    print("\nRunning 1 task through each condition (this takes a few minutes)...")
    order = ["A", "C"] + (["D"] if crewai_ok else []) + ["B"]
    for cond in order:
        try:
            recs = run_condition(cfg, cond, one, runs=1, judge_client=judge)
            r = recs[0]
            status = "answered" if r["success_execution"] else f"FAILED:{r['failure_type']}"
            tok = r["total_tokens"] if r["total_tokens"] is not None else "null"
            print(f"{OK} condition {cond}: {status} | tokens={tok} "
                  f"| calls={r['llm_calls']} | {r['latency_seconds']:.1f}s "
                  f"| peak RAM {r['peak_ram_mb']} MB")
            if r["total_tokens"] is None:
                print(f"{WARN}   token accounting returned null for condition {cond}")
        except Exception as exc:
            failures += 1
            print(f"{BAD} condition {cond}: {type(exc).__name__}: {exc}")

    # 9. cleanup
    print()
    if failures:
        print(f"{BAD} smoke test finished with {failures} failing condition(s).")
        print("     Fix these before running the full benchmark.")
        return 1

    print(f"{OK} ALL CHECKS PASSED. Smoke output is in results_smoke/ "
          f"(safe to delete).")
    print("\nNext:  python run_experiment.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
