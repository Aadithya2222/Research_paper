"""Verify Condition D now makes tool calls, BEFORE rerunning 132 tasks.

    python verify_condition_d.py

This answers one question and nothing else:

    Does Condition D, with tools bound, actually invoke them?

Binding tools does not guarantee the model calls them. That is an
empirical question about an 8B model, not a question about code, and it
can only be settled by running it. This script runs three tool-requiring
tasks through Condition D and reports the tool-call count.

It writes to results_verify/ so it cannot contaminate your real data.

INTERPRETING THE RESULT
-----------------------
tool calls > 0 on most tasks
    The fix works. Proceed with the Condition D rerun.

tool calls == 0 with tools bound
    The tools are available and the model is declining to use them.
    That is a genuine finding about CrewAI's prompting of a small
    model, NOT a bug, and it should be reported as such rather than
    engineered around. Do not rerun; report it.

adapter raises
    Your CrewAI version does not expose the expected tool base class.
    Report the version and stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.tasks import load_tasks                    # noqa: E402
from core.config import detect_hardware, load_config, setup_logging  # noqa: E402
from core.llm import client_from_config                   # noqa: E402
from tools.local_tools import build_registry              # noqa: E402

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def main() -> int:
    print("=" * 66)
    print("  CONDITION D TOOL-BINDING VERIFICATION")
    print("=" * 66)

    cfg = load_config("config.yaml")
    cfg.data["experiment"]["output_directory"] = "results_verify"
    cfg.data["logging"]["raw_directory"] = "results_verify/raw"
    setup_logging("WARNING", console=True)

    try:
        import crewai
        print(f"{OK} crewai version {getattr(crewai, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{BAD} crewai not installed: {exc}")
        return 1

    registry = build_registry(cfg)
    print(f"{OK} registry defines {len(registry.names)} tool(s): "
          f"{', '.join(registry.names)}")

    # 1. Does the adapter build tools at all?
    from core.llm import Ledger
    from agents.crewai_tools_adapter import build_crewai_tools
    probe_ledger = Ledger()
    crew_tools = build_crewai_tools(registry, probe_ledger)
    if not crew_tools:
        print(f"{BAD} adapter produced 0 tools. Your CrewAI version does not "
              f"expose crewai.tools.BaseTool. Report the version and stop.")
        return 1
    print(f"{OK} adapter wrapped {len(crew_tools)} tool(s)")

    # 2. Does a wrapped tool actually execute?
    try:
        result = crew_tools[0]._run("6*7") if crew_tools[0].name == "calculator" \
            else crew_tools[0]._run("irrigation")
        print(f"{OK} wrapped tool executes, returned: {str(result)[:60]}")
        print(f"{OK} ledger counted {probe_ledger.tool_calls} tool call(s)")
    except Exception as exc:
        print(f"{BAD} wrapped tool failed to execute: {exc}")
        return 1

    # 3. Ollama reachable?
    client = client_from_config(cfg)
    if not client.is_available():
        print(f"{BAD} Ollama unreachable. Start it: ollama serve")
        return 1
    if not client.model_present():
        print(f"{BAD} model missing. Run: ollama pull {client.model}")
        return 1
    print(f"{OK} model {client.model} available")

    # 4. The real test: run tasks that REQUIRE tools.
    tasks = load_tasks("tasks/tasks.json", cfg)
    tool_tasks = [t for t in tasks if t.required_tools][:3]
    if not tool_tasks:
        tool_tasks = tasks[:3]
    print(f"\nRunning {len(tool_tasks)} tool-requiring task(s) through "
          f"Condition D. This takes several minutes.\n")

    from agents.conditions import run_condition_d
    hardware = detect_hardware()
    results = []
    for t in tool_tasks:
        print(f"  {t.task_id} (tier {t.complexity_tier}) ...", flush=True)
        try:
            out = run_condition_d(cfg, client, registry, t.to_dict())
        except RuntimeError as exc:
            print(f"{BAD} {exc}")
            return 1
        except Exception as exc:
            print(f"{BAD} {t.task_id}: {type(exc).__name__}: {exc}")
            results.append({"task": t.task_id, "tool_calls": None,
                            "error": str(exc)})
            continue
        s = out.ledger.summary()
        results.append({
            "task": t.task_id,
            "tool_calls": s["tool_calls"],
            "tools_bound": out.notes.get("crew_tools_bound"),
            "total_tokens": s["total_tokens"],
            "llm_calls": s["llm_calls"],
            "answered": out.final_answer is not None,
        })
        print(f"      tool calls: {s['tool_calls']}   "
              f"tokens: {s['total_tokens']}   "
              f"answered: {out.final_answer is not None}")

    Path("results_verify").mkdir(exist_ok=True)
    Path("results_verify/verification.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # 5. Verdict
    counts = [r["tool_calls"] for r in results if r.get("tool_calls") is not None]
    total = sum(counts) if counts else 0
    with_calls = sum(1 for c in counts if c > 0)

    print("\n" + "=" * 66)
    if not counts:
        print(f"{BAD} No task completed. Fix the errors above before rerunning.")
        return 1
    if total == 0:
        print(f"{WARN} VERDICT: tools were bound ({crew_tools and len(crew_tools)}) "
              f"but Condition D made 0 tool calls across {len(counts)} task(s).")
        print()
        print("  This is NOT a bug to engineer around. The tools are")
        print("  available and the model is declining to use them under")
        print("  CrewAI's prompting. That is a genuine finding about how a")
        print("  small model behaves in this framework, and it should be")
        print("  REPORTED, not fixed.")
        print()
        print("  Do NOT rerun the full Condition D. Instead, state in the")
        print("  paper that tools were bound and remained unused, which")
        print("  makes the zero tool-call count a framework-behaviour")
        print("  result rather than a configuration defect.")
        return 2

    print(f"{OK} VERDICT: {with_calls}/{len(counts)} task(s) made tool calls, "
          f"{total} in total.")
    print()
    print("  The fix works. The D-C contrast will now compare frameworks")
    print("  at matched tool availability.")
    print()
    print("  Proceed with the Condition D rerun:")
    print("      python run_experiment.py --condition D "
          "--task-set my_tasks.json --no-resume")
    print()
    print("  IMPORTANT: --no-resume is required here, because the old")
    print("  Condition D records must be REPLACED, not skipped. Move the")
    print("  existing results/raw/condition_D/ aside first so the old and")
    print("  new records are never mixed:")
    print("      move results\\raw\\condition_D results\\raw\\condition_D_OLD")
    print()
    print("  Condition B was matched to the OLD Condition D budget, so B")
    print("  must be rerun after D. Conditions A and C are unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
