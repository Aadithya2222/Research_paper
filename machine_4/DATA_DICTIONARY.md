# Data Dictionary: Coordination Tax Benchmark Dataset

This document describes the schema and variables included in the dataset for the study:
**"Quantifying the Coordination Tax: Multi-Agent Orchestration vs. Compute-Matched Single-Agent Inference"**

---

## 1. Raw Execution Records (`raw_records/condition_{A,B,C,D}_pooled.jsonl`)

Each line in a `.jsonl` file represents a single task execution under a specific experimental condition.

| Field | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `string` | Unique identifier for the benchmark task (e.g., `Q001`) |
| `condition` | `string` | Experimental condition (`A`, `B`, `C`, or `D`) |
| `machine_id` | `string` | Machine hardware stratification group (`machine_1`..`machine_4`) |
| `config_hash` | `string` | Pre-registered configuration SHA-256 hash (`e4da2c9abab2`) |
| `final_answer` | `string` | Text answer produced by the agent / multi-agent system |
| `iterations` | `integer` | Number of reasoning/action steps taken during execution |
| `stopped_reason` | `string` | Termination condition (`final_answer`, `max_iterations`, `framework_error`, etc.) |
| `total_tokens` | `integer` | Total input + output LLM tokens consumed |
| `prompt_tokens` | `integer` | Total prompt/input tokens |
| `completion_tokens` | `integer` | Total completion/output tokens |
| `wall_clock_seconds` | `float` | Elapsed execution time in seconds |
| `tool_calls` | `integer` | Number of local tools invoked during execution |
| `tool_call_details` | `array` | Log of tool names, arguments, and return codes |
| `failure_type` | `string` | Classification of execution failure (if any, e.g., `timeout`, `model_error`, `writer_failure`) |

---

## 2. Benchmark Tasks (`benchmark_data/tasks.json`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `string` | Unique identifier matching execution records |
| `task_text` | `string` | Full prompt presented to the agent |
| `evaluation_type` | `string` | Sub-domain / task type classification |
| `gold_facts` | `array` | Reference ground-truth facts for scoring correctness |

---

## 3. Evaluation & Statistics (`statistics/*.json`)

| File | Content Description |
| :--- | :--- |
| `paired_contrasts.json` | McNemar's tests, paired t-tests, Wilcoxon signed-rank tests across conditions |
| `stratified_analysis.json` | Performance metrics stratified across machine hardware classes |
| `robustness_checks.json` | Sensitivity analyses, seed variations, and bootstrap confidence intervals |

---

## 4. Hardware Machine Stratification

- **Machine 1 & 2**: Standard Workstation tier (16GB RAM, RTX 3060/4060 GPU class)
- **Machine 3 & 4**: High-Performance Workstation tier (32GB+ RAM, RTX 4080/4090 GPU class)
