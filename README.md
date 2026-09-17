# Coordination Tax in Multi-Agent LLM Systems: Empirical Benchmark Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pre-registration Hash](https://img.shields.io/badge/config__hash-e4da2c9abab2-green.svg)](machine_1/config.yaml)
[![Mendeley Data](https://img.shields.io/badge/Mendeley_Data-Dataset_Archive-orange.svg)](#mendeley-data-repository)

This repository contains the complete experimental code, pre-registered configurations, orchestration scripts, and analytical pipelines for the research paper:

> **"The Coordination Tax in Multi-Agent LLM Architectures: An Empirical Benchmark of Task Decomposition Overhead vs. Single-Agent Baselines"**

---

## 🔬 Experimental Architecture & Design

The benchmark investigates the computational, latency, and quality trade-offs across four experimental conditions evaluated on identical hardware baselines:

- **Condition A (Standard Single-Agent ReAct Baseline):** Single-agent iterative reasoning with local deterministic tool invocation.
- **Condition B (Compute-Matched Best-of-$N$ Baseline):** Compute-equalized sampling matching multi-agent token and execution budgets under pre-registered selection rules.
- **Condition C (Uncoordinated Multi-Agent Baseline):** Sequential multi-agent pipeline (Planner $\to$ Researcher $\to$ Writer) without dynamic feedback loops.
- **Condition D (Fully Orchestrated Multi-Agent System):** Hierarchical multi-agent framework utilizing CrewAI orchestration with role charters and tool bindings.

---

## 💻 4-Machine Distributed Execution Matrix

To guarantee environmental control and prevent confounding thermal throttling or resource starvation, the benchmark was distributed across 4 calibrated machines ($N=132$ unique tasks $\times$ 4 conditions = 600 evaluated task-condition instances, including stochasticity runs):

```
                                  [ 132 Benchmark Tasks ]
                                             │
             ┌───────────────────┬───────────┴───────────┬───────────────────┐
             ▼                   ▼                       ▼                   ▼
       ┌───────────┐       ┌───────────┐           ┌───────────┐       ┌───────────┐
       │ Machine 1 │       │ Machine 2 │           │ Machine 3 │       │ Machine 4 │
       │ Tasks 1-33│       │Tasks 34-66│           │Tasks 67-99│       │Tasks 100-132
       └───────────┘       └───────────┘           └───────────┘       └───────────┘
             │                   │                       │                   │
             └───────────────────┴───────────┬───────────┴───────────────────┘
                                             ▼
                             [ Central Analysis & Aggregation ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ GitHub Repository ]                       [ Mendeley Data DOI ]
             (Reproducible Codebase)                     (Raw Datasets & Logs)
```

### Machine Allocation Breakdown

| Directory | Machine ID | Task Shard Range | Primary Role & Responsibilities | Pre-registration Hash |
| :--- | :---: | :---: | :--- | :---: |
| [`machine_1/`](./machine_1/) | **Machine 1** | Tasks `001` – `033` | Calibration, Condition D/B batches, Tool binding diagnostics | `e4da2c9abab2` |
| [`machine_2/`](./machine_2/) | **Machine 2** | Tasks `034` – `066` | Condition D/B batches, Task execution shard 2 | `e4da2c9abab2` |
| [`machine_3/`](./machine_3/) | **Machine 3** | Tasks `067` – `099` | Condition D/B batches, Task execution shard 3 | `e4da2c9abab2` |
| [`machine_4/`](./machine_4/) | **Machine 4** | Tasks `100` – `132` | Condition D/B batches, Tool verification & safety task runs | `e4da2c9abab2` |

---

## 📂 Repository Structure

Each machine directory contains a self-contained, reproducible execution environment configured with identical pre-registered hyperparameters:

```text
.
├── machine_1/                      # Machine 1 benchmark suite (Shard 1/4)
│   ├── config.yaml                 # Pre-registered config (machine.id: 1)
│   ├── rerun_auto.py               # Thermal-managed batch executor
│   ├── send_back.py                # Results packager for central deposit
│   └── ...
├── machine_2/                      # Machine 2 benchmark suite (Shard 2/4)
│   ├── config.yaml                 # Pre-registered config (machine.id: 2)
│   └── ...
├── machine_3/                      # Machine 3 benchmark suite (Shard 3/4)
│   ├── config.yaml                 # Pre-registered config (machine.id: 3)
│   └── ...
├── machine_4/                      # Machine 4 benchmark suite (Shard 4/4)
│   ├── config.yaml                 # Pre-registered config (machine.id: 4)
│   └── ...
└── README.md                       # Root repository documentation
```

### Core Module Layout (Shared Across Machines)

- `agents/` — Implementation of ReAct agents (`react_agent.py`), condition builders (`conditions.py`), and framework adapters (`crewai_tools_adapter.py`).
- `analysis/` — Hypothesis testing (`H1`, `H2`, `H3`), bootstrap confidence intervals, and mechanism decomposition (`statistics_pipeline.py`, `mechanism.py`).
- `benchmark/` — Task runner, execution harness, and state trackers (`runner.py`, `tasks.py`).
- `core/` — Model abstraction layer, Ollama API wrappers, token counting, and configuration validator (`config.py`, `llm.py`, `resources.py`).
- `evaluation/` — LLM-as-a-judge rubric scorer, exact-match evaluator, and gold fact coverage metrics (`evaluate.py`).
- `tools/` — Deterministic local tools (Search Corpus, Document Lookup, Calculator) to eliminate network variance.
- `validation/` — Automated sanity checks, pre-flight assertion checks, and environment validation (`checks.py`).

---

## 🚀 Reproduction & Setup

### 1. Prerequisites

- **OS:** Windows 10/11, macOS, or Ubuntu 20.04+
- **Python:** Version 3.10 or higher
- **Local Inference Engine:** [Ollama](https://ollama.com/) running locally:
  ```bash
  ollama pull llama3.1:8b
  ollama pull phi3:latest
  ```

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Aadithya2222/Research_paper.git
cd Research_paper

# Select target machine folder (e.g., machine_1)
cd machine_1

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Configuration Integrity

Before running any benchmark tasks, verify that the configuration hash matches the pre-registered value (`e4da2c9abab2`):

```bash
python -c "import sys;sys.path.insert(0,'.');from core.config import load_config;print(load_config('config.yaml').config_hash)"
# Expected Output: e4da2c9abab2
```

### 4. Running the Benchmark

```bash
# Execute the automated runner with thermal pacing and recovery
python rerun_auto.py

# Check live progress in a separate terminal
python status.py
```

---

## 📦 Mendeley Data Repository

In accordance with open science practices and journal Data Availability policies, raw execution traces, individual model outputs, and statistical dumps are hosted on Mendeley Data:

- **Mendeley Deposit Structure:**
  - `raw_records.jsonl` — 600 raw execution logs with prompt traces, token timings, and tool call logs.
  - `tasks.json` — 132 structured multi-hop reasoning tasks with gold fact assertions.
  - `corpus.json` — Local document corpus used by retrieval tools.
  - `statistics_report.json` — Summary statistics, bootstrap confidence intervals, and hypothesis outcomes.
  - `DATA_DICTIONARY.md` — Formal schema for all JSONL and evaluation columns.

---

## 📜 Citation & License

If you use this benchmark, code, or experimental protocol in your research, please cite:

```bibtex
@article{coordination_tax_2026,
  title={The Coordination Tax in Multi-Agent LLM Architectures: An Empirical Benchmark of Task Decomposition Overhead vs. Single-Agent Baselines},
  author={Aadithya and Co-authors},
  journal={Journal of Artificial Intelligence Research},
  year={2026}
}
```

This project is licensed under the **MIT License**.
