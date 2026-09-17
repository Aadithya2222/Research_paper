"""Configuration loading, hashing and experiment identity.

Every experimental parameter comes from config.yaml. The config hash is
written into every raw record so that any result can be traced back to
the exact configuration that produced it.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

LOG = logging.getLogger(__name__)


@dataclass
class Config:
    """Parsed configuration plus its content hash."""

    data: Dict[str, Any]
    path: Path
    config_hash: str = field(default="", repr=False)

    #: Keys that identify the machine or output location, not the
    #: experimental parameters. Excluded from the hash so that laptops
    #: running the same experiment produce the same hash and their
    #: results can be merged.
    NON_EXPERIMENTAL_KEYS = ("machine", "logging")

    def __post_init__(self) -> None:
        if not self.config_hash:
            experimental = {k: v for k, v in self.data.items()
                            if k not in self.NON_EXPERIMENTAL_KEYS}
            blob = json.dumps(experimental, sort_keys=True).encode("utf-8")
            self.config_hash = hashlib.sha256(blob).hexdigest()[:12]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value with a dotted path, e.g. 'model.name'."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Fetch a nested value, raising if it is absent."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {dotted}")
        return value


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load config.yaml from disk."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} did not parse to a mapping.")
    return Config(data=data, path=p)


def detect_hardware() -> Dict[str, Any]:
    """Cross-platform hardware snapshot.

    Anything that cannot be measured on this machine is recorded as
    None -- never as a guess. GPU detection is optional by design.
    """
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
        "cpu_count_logical": None,
        "cpu_count_physical": None,
        "total_ram_mb": None,
        "gpu_name": None,
        "gpu_memory_total_mb": None,
    }
    try:
        import psutil

        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["total_ram_mb"] = round(psutil.virtual_memory().total / (1024 ** 2), 1)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOG.warning("psutil unavailable, RAM/CPU not recorded: %s", exc)

    # GPU is strictly optional. Absence is recorded as None, not zero.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            first = out.stdout.strip().splitlines()[0]
            name, mem = [x.strip() for x in first.split(",")[:2]]
            info["gpu_name"] = name
            info["gpu_memory_total_mb"] = float(mem)
    except Exception:
        pass  # No GPU, or nvidia-smi absent. Stays None.

    return info


_ID_COUNTER = itertools.count(1)


def make_experiment_id(condition: str, tier: int, run_index: int,
                       task_id: Optional[str] = None) -> str:
    """Build a unique, sortable experiment id.

    Format: YYYYMMDDTHHMMSSmmm_<COND>_T<tier>_R<run>[_<task_id>]_<seq>

    A process-local monotonic counter is appended because two executions
    can start within the same millisecond, which would otherwise produce
    colliding ids and duplicate raw records.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    seq = next(_ID_COUNTER)
    base = f"{stamp}_{condition}_T{tier}_R{run_index:03d}"
    if task_id:
        base = f"{base}_{task_id}"
    return f"{base}_{seq:05d}"


def setup_logging(level: str = "INFO", console: bool = True,
                  logfile: Optional[Path] = None) -> None:
    """Configure root logging once, for the whole process."""
    handlers: list[logging.Handler] = []
    if console:
        handlers.append(logging.StreamHandler())
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        handlers=handlers,
        force=True,
    )
