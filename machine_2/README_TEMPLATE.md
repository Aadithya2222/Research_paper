# Dataset for: "When Does Role-Based Collaboration Beat Compute-Matched Sampling? A Framework-Controlled Study of Multi-Agent LLM Orchestration on Resource-Constrained Local Hardware"

## Dataset Overview

This dataset contains the complete empirical data from 600 task execution runs across four experimental conditions evaluated on resource-constrained hardware across four machines.

- **Total Execution Records:** 600 (150 per condition across 132 tasks)
- **Experimental Conditions:**
  - **Condition A:** Single ReAct agent baseline
  - **Condition B:** Compute-matched Best-of-N sampling baseline
  - **Condition C:** LangGraph multi-agent role-based orchestration
  - **Condition D:** CrewAI multi-agent role-based orchestration
- **Configuration Hash:** `e4da2c9abab2`
- **Models Used:** `llama3.1:8b` (agent) and `qwen2.5:7b-instruct` (judge) via Ollama

---

## Contents of the Deposit

1. `tasks/tasks.json` - Complete benchmark task definitions (132 tasks).
2. `tasks/corpus.json` - Domain retrieval corpus.
3. `tasks/calibration_tasks.json` - Standardization benchmark tasks for hardware calibration.
4. `raw_data/` - Unfiltered JSONL records for conditions A, B, C, and D.
5. `statistics/` - Aggregated statistical analysis, paired difference tests, confidence intervals, and tool diagnostic metrics.
6. `DATA_DICTIONARY.md` - Complete field specifications for all data files.

---

## Data Replication & Usage

To reproduce the statistical analysis reported in the paper using this dataset:

1. Clone the GitHub repository: `https://github.com/YOUR_ORGANIZATION/coordination_tax_benchmark`
2. Install dependencies: `pip install -r requirements.txt`
3. Place this dataset under `results/` in the cloned repository directory.
4. Run statistical synthesis:
   ```bash
   python analyze.py
   python make_figures.py
   python make_report.py
   ```

---

## Hardware Stratification

Data was collected across 4 dedicated machine nodes (Machine 1 through Machine 4) running local Ollama inference under locked configuration parameters (`config.yaml`).
