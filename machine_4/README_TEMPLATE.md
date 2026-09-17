# Dataset for: Quantifying the Coordination Tax in Multi-Agent vs Compute-Matched Single-Agent Systems

## Overview
This repository contains the complete execution traces, benchmark tasks, local corpus, and statistical analysis outputs for the experimental study on multi-agent coordination costs.

## Dataset Contents
- `raw_records/`: 600 raw JSONL execution records collected across 4 hardware stratification nodes.
  - `condition_A_pooled.jsonl`: Single-agent ReAct
  - `condition_B_pooled.jsonl`: Compute-matched Best-of-N single agent
  - `condition_C_pooled.jsonl`: Modular 3-role pipeline (LangGraph)
  - `condition_D_pooled.jsonl`: Multi-agent framework (CrewAI)
- `benchmark_data/`:
  - `tasks.json`: 132 benchmark evaluation tasks with gold facts.
  - `corpus.json`: Local reference corpus for retrieval tasks.
  - `calibration_tasks.json`: Validation tasks used for initial compute matching.
- `statistics/`: Paired statistical tests, effect size computations, and machine stratification reports.
- `DATA_DICTIONARY.md`: Full schema and field definitions for all records.

## Code and Replication
The code pipeline used to generate and evaluate this dataset is available on GitHub:
- **Repository URL**: `[INSERT GITHUB REPOSITORY URL HERE]`
- **Pre-registration Config Hash**: `e4da2c9abab2`

## Data Availability Statement
All raw model responses, tool logs, token counts, and latency measurements are provided unedited in this dataset deposit under CC-BY 4.0 license.
