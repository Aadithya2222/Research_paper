"""Ollama chat client with exact token accounting and timing.

Design notes that matter for experimental validity:

* Token counts come from Ollama's native `prompt_eval_count` and
  `eval_count` fields -- the model's own tokenizer. We never estimate
  tokens from character length. If a field is missing from the
  response, the count is recorded as None, not guessed.
* Retries consume real compute, so every retry attempt is appended to
  the call ledger and counted in tokens, calls and latency.
* TTFT is measured from request dispatch to the first streamed chunk
  that carries non-empty content.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import requests

LOG = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """One LLM invocation, including failed attempts."""

    role_label: str
    attempt: int
    ok: bool
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_seconds: float
    ttft_seconds: Optional[float]
    error: Optional[str] = None

    @property
    def total_tokens(self) -> Optional[int]:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass
class Ledger:
    """Accumulates every unit of compute spent during one task run."""

    calls: List[CallRecord] = field(default_factory=list)
    tool_calls: int = 0
    tool_latency_seconds: float = 0.0
    judge_calls: List[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    def add_judge(self, rec: CallRecord) -> None:
        self.judge_calls.append(rec)

    def _sum(self, records: List[CallRecord], attr: str) -> Optional[int]:
        vals = [getattr(r, attr) for r in records]
        if any(v is None for v in vals):
            return None  # incomplete accounting is reported honestly
        return int(sum(vals))  # type: ignore[arg-type]

    def summary(self, include_judge: bool = True) -> Dict[str, Any]:
        """Aggregate cost. Judge cost is included when configured."""
        records = list(self.calls) + (list(self.judge_calls) if include_judge else [])
        n_retries = sum(1 for r in records if r.attempt > 1)
        return {
            "input_tokens": self._sum(records, "input_tokens"),
            "output_tokens": self._sum(records, "output_tokens"),
            "total_tokens": (
                None
                if self._sum(records, "input_tokens") is None
                or self._sum(records, "output_tokens") is None
                else self._sum(records, "input_tokens") + self._sum(records, "output_tokens")
            ),
            "llm_calls": len(records),
            "agent_llm_calls": len(self.calls),
            "judge_llm_calls": len(self.judge_calls),
            "tool_calls": self.tool_calls,
            "retries": n_retries,
            "model_latency_seconds": round(sum(r.latency_seconds for r in records), 4),
            "tool_latency_seconds": round(self.tool_latency_seconds, 4),
            "first_ttft_seconds": next(
                (r.ttft_seconds for r in self.calls if r.ttft_seconds is not None), None
            ),
            "call_ledger": [asdict(r) for r in records],
        }


class OllamaClient:
    """Thin, explicit wrapper over the Ollama /api/chat endpoint."""

    def __init__(self, host: str, model: str, temperature: float = 0.0,
                 top_p: float = 1.0, seed: Optional[int] = None,
                 max_output_tokens: int = 1024, num_ctx: int = 8192,
                 timeout: int = 300, retry_limit: int = 2) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.retry_limit = retry_limit

    # -- health -------------------------------------------------------
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def model_present(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            if r.status_code != 200:
                return False
            names = {m.get("name", "") for m in r.json().get("models", [])}
            base = self.model.split(":")[0]
            return any(n == self.model or n.startswith(base) for n in names)
        except Exception:
            return False

    # -- generation ---------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], ledger: Ledger,
             role_label: str, is_judge: bool = False) -> str:
        """Send a chat request, streaming to capture TTFT.

        Every attempt (including failures) is written to the ledger, so
        retries are never hidden from the compute accounting.
        """
        last_error: Optional[str] = None
        for attempt in range(1, self.retry_limit + 2):
            started = time.perf_counter()
            ttft: Optional[float] = None
            text_parts: List[str] = []
            in_tok: Optional[int] = None
            out_tok: Optional[int] = None
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_output_tokens,
                        "num_ctx": self.num_ctx,
                    },
                }
                if self.seed is not None:
                    payload["options"]["seed"] = self.seed

                with requests.post(f"{self.host}/api/chat", json=payload,
                                   stream=True, timeout=self.timeout) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line.decode("utf-8"))
                        piece = chunk.get("message", {}).get("content", "")
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - started
                            text_parts.append(piece)
                        if chunk.get("done"):
                            # Native tokenizer counts. Absent => None.
                            in_tok = chunk.get("prompt_eval_count")
                            out_tok = chunk.get("eval_count")

                elapsed = time.perf_counter() - started
                rec = CallRecord(role_label, attempt, True, in_tok, out_tok,
                                 round(elapsed, 4),
                                 round(ttft, 4) if ttft is not None else None)
                (ledger.add_judge if is_judge else ledger.add)(rec)
                return "".join(text_parts)

            except Exception as exc:
                elapsed = time.perf_counter() - started
                last_error = f"{type(exc).__name__}: {exc}"
                rec = CallRecord(role_label, attempt, False, in_tok, out_tok,
                                 round(elapsed, 4), None, last_error)
                (ledger.add_judge if is_judge else ledger.add)(rec)
                LOG.warning("LLM call failed (attempt %d/%d): %s",
                            attempt, self.retry_limit + 1, last_error)

        raise RuntimeError(f"LLM call failed after retries: {last_error}")


def client_from_config(cfg, judge: bool = False) -> OllamaClient:
    """Build a client from the config object."""
    ns = "judge" if judge else "model"
    return OllamaClient(
        host=cfg.get(f"{ns}.host", "http://localhost:11434"),
        model=cfg.require(f"{ns}.name"),
        temperature=float(cfg.get(f"{ns}.temperature", 0.0)),
        top_p=float(cfg.get("model.top_p", 1.0)),
        seed=cfg.get(f"{ns}.seed"),
        max_output_tokens=int(cfg.get(f"{ns}.max_output_tokens", 1024)),
        num_ctx=int(cfg.get("model.num_ctx", 8192)),
        timeout=int(cfg.get(f"{ns}.request_timeout_seconds", 300)),
        retry_limit=int(cfg.get("experiment.retry_limit", 2)),
    )
