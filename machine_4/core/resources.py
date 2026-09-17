"""Cross-platform resource monitoring (Windows / macOS / Linux).

Samples the benchmark process AND, where identifiable, the Ollama
server process -- because on a local setup the model's memory lives in
the server, not in this process. If a quantity cannot be measured on
this machine it is reported as None. Nothing is estimated.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore


def _gpu_memory_used_mb() -> Optional[float]:
    """Return used VRAM in MB, or None when no GPU tooling is present."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


@dataclass
class ResourceSummary:
    peak_ram_mb: Optional[float]
    mean_ram_mb: Optional[float]
    peak_cpu_percent: Optional[float]
    mean_cpu_percent: Optional[float]
    peak_gpu_memory_mb: Optional[float]
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_ram_mb": self.peak_ram_mb,
            "mean_ram_mb": self.mean_ram_mb,
            "peak_cpu_percent": self.peak_cpu_percent,
            "mean_cpu_percent": self.mean_cpu_percent,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "resource_samples": self.n_samples,
        }


class ResourceMonitor:
    """Background sampler; use as a context manager around a task run."""

    def __init__(self, sample_hz: float = 5.0, track_ollama: bool = True) -> None:
        self.interval = 1.0 / max(sample_hz, 0.5)
        self.track_ollama = track_ollama
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ram: List[float] = []
        self._cpu: List[float] = []
        self._gpu: List[float] = []

    def _targets(self) -> List[Any]:
        if psutil is None:
            return []
        procs = [psutil.Process()]
        if self.track_ollama:
            for p in psutil.process_iter(["name"]):
                try:
                    if "ollama" in (p.info.get("name") or "").lower():
                        procs.append(p)
                except Exception:
                    continue
        return procs

    def _loop(self) -> None:
        procs = self._targets()
        while not self._stop.is_set():
            try:
                rss = 0.0
                cpu = 0.0
                for p in procs:
                    try:
                        rss += p.memory_info().rss / (1024 ** 2)
                        cpu += p.cpu_percent(interval=None)
                    except Exception:
                        continue
                if rss > 0:
                    self._ram.append(rss)
                    self._cpu.append(cpu)
                g = _gpu_memory_used_mb()
                if g is not None:
                    self._gpu.append(g)
            except Exception as exc:  # pragma: no cover
                LOG.debug("resource sample failed: %s", exc)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        if psutil is None:
            LOG.warning("psutil not installed; resource metrics will be null.")
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def summary(self) -> ResourceSummary:
        def _peak(xs: List[float]) -> Optional[float]:
            return round(max(xs), 1) if xs else None

        def _mean(xs: List[float]) -> Optional[float]:
            return round(sum(xs) / len(xs), 1) if xs else None

        return ResourceSummary(
            peak_ram_mb=_peak(self._ram),
            mean_ram_mb=_mean(self._ram),
            peak_cpu_percent=_peak(self._cpu),
            mean_cpu_percent=_mean(self._cpu),
            peak_gpu_memory_mb=_peak(self._gpu),
            n_samples=len(self._ram),
        )
