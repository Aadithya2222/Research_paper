"""Deterministic local tools.

External web APIs are deliberately NOT used: they would make the
benchmark irreproducible and would inject network variance into the
latency measurements, confounding the very quantity under study.

The tool descriptions here are the SINGLE source of truth. Conditions
A, B, C and D all render their prompts from this same registry, so no
condition can receive a richer tool description than another.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOG = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]

    def run(self, arg: str) -> str:
        return self.fn(arg)


class ToolRegistry:
    """Holds the enabled tools and renders their shared description block."""

    def __init__(self, tools: List[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def describe(self) -> str:
        """Identical text for every condition -- fairness depends on this."""
        lines = ["AVAILABLE TOOLS:"]
        for name in self.names:
            lines.append(f"- {name}: {self._tools[name].description}")
        return "\n".join(lines)

    def call(self, name: str, arg: str) -> tuple[str, float, bool]:
        """Run a tool. Returns (output, elapsed_seconds, ok)."""
        tool = self.get(name)
        started = time.perf_counter()
        if tool is None:
            return (f"ERROR: unknown tool '{name}'. Valid tools: "
                    f"{', '.join(self.names)}"), time.perf_counter() - started, False
        try:
            out = tool.run(arg)
            return out, time.perf_counter() - started, True
        except Exception as exc:
            return (f"ERROR: tool '{name}' failed: {type(exc).__name__}: {exc}",
                    time.perf_counter() - started, False)


# ---------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------

_ALLOWED_MATH = re.compile(r"^[0-9\s\.\+\-\*\/\(\)\%\,eE]+$")


def _strip_wrappers(arg: str) -> str:
    """Remove quoting the model commonly adds around tool arguments.

    Observed in live runs: the model emits  Action Input: "25 + 40 / 2"
    and a strict validator rejects the quotes, so the model "simplifies"
    and retries forever until the iteration cap, producing a spurious
    failure that has nothing to do with the architecture under test.
    """
    a = arg.strip()
    for _ in range(3):
        a = a.strip()
        if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'`":
            a = a[1:-1]
        elif a.startswith("```") and a.endswith("```"):
            a = a[3:-3]
        else:
            break
    return a.strip()


def _calculator(expr: str) -> str:
    """Evaluate an arithmetic expression. Deliberately restrictive."""
    cleaned = _strip_wrappers(expr).replace(",", "")
    if not cleaned:
        return "ERROR: empty expression."
    if not _ALLOWED_MATH.match(cleaned):
        return ("ERROR: unsupported characters. Send ONLY the bare "
                "expression with no quotes and no words, "
                "for example:  25 + 40 / 2")
    try:
        value = eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:
        return f"ERROR: could not evaluate: {exc}"
    if isinstance(value, float) and not math.isfinite(value):
        return "ERROR: result is not finite."
    return str(value)


class _Corpus:
    """A tiny local document corpus loaded from JSON."""

    def __init__(self, path: Path) -> None:
        self.docs: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            for doc in payload.get("documents", []):
                self.docs[doc["doc_id"]] = doc
        else:
            LOG.warning("Corpus file not found at %s; corpus tools return empty.",
                        path)

    def search(self, query: str, k: int = 3) -> str:
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
        if not terms:
            return "No results: query too short."
        scored = []
        for doc in self.docs.values():
            hay = (doc.get("title", "") + " " + doc.get("text", "")).lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, doc))
        if not scored:
            return "No results found."
        scored.sort(key=lambda x: (-x[0], x[1]["doc_id"]))
        out = []
        for _, doc in scored[:k]:
            snippet = doc.get("text", "")[:280]
            out.append(f"[{doc['doc_id']}] {doc.get('title','')}: {snippet}")
        return "\n".join(out)

    def lookup(self, doc_id: str) -> str:
        doc = self.docs.get(doc_id.strip())
        if doc is None:
            return (f"ERROR: no document '{doc_id.strip()}'. "
                    f"Known ids: {', '.join(sorted(self.docs)[:10])}")
        return f"[{doc['doc_id']}] {doc.get('title','')}\n{doc.get('text','')}"


def build_registry(cfg) -> ToolRegistry:
    """Construct the registry from config. Same object for all conditions."""
    enabled = set(cfg.get("tools.enabled", []))
    corpus = _Corpus(Path(cfg.get("tools.corpus_path", "tasks/corpus.json")))

    catalogue: Dict[str, Tool] = {
        "calculator": Tool(
            name="calculator",
            description=("evaluate an arithmetic expression. "
                         "Input: a plain expression such as 12*(3+4)."),
            fn=_calculator,
        ),
        "search_corpus": Tool(
            name="search_corpus",
            description=("search the local document corpus by keywords. "
                         "Input: a short keyword query. Returns up to 3 "
                         "matching document ids with snippets."),
            fn=lambda q: corpus.search(_strip_wrappers(q)),
        ),
        "lookup_document": Tool(
            name="lookup_document",
            description=("retrieve the full text of one document. "
                         "Input: a document id such as D003."),
            fn=lambda d: corpus.lookup(_strip_wrappers(d)),
        ),
    }
    tools = [catalogue[n] for n in sorted(enabled) if n in catalogue]
    missing = enabled - set(catalogue)
    if missing:
        raise ValueError(f"Unknown tools in config: {sorted(missing)}")
    return ToolRegistry(tools)
