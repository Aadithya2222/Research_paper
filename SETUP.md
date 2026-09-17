# Setup and run — from scratch

Everything below is typed in **one Command Prompt window**, inside the
project folder, with the virtual environment active.

## Part 1 — Install (once)

**1. Python 3.11+** from python.org. Tick **"Add Python to PATH"** on the
first install screen. Verify: `python --version`

**2. Open the project.** Unzip to e.g. `C:\research\ctb`, then:
```
cd C:\research\ctb
```

**3. Virtual environment.**
```
python -m venv .venv
.venv\Scripts\activate
```
Your prompt must show `(.venv)`. Reopen a window later? Run
`.venv\Scripts\activate` again first.

**4. Packages.**
```
pip install -r requirements.txt
```

**5. Ollama** from ollama.com/download. Verify: `ollama --version`

**6. Models** (about 9 GB total, 10–40 min).
```
ollama pull llama3.1:8b
ollama pull qwen2.5:7b-instruct
```
Two models because the judge must not be the model that wrote the
answers. The code refuses to run if they are the same.

## Part 2 — Check everything works

```
python pipeline.py --check
```
Environment, packages, Ollama, models, config, task files. Ends in PASS.

## Part 3 — Pilot (do this before the real run)

```
python pipeline.py --pilot
```
Four tasks through all four conditions, then evaluation, statistics,
figures and a full PDF report. Roughly 20–60 minutes.

**Read the pilot report before going further.** Look for: a sane failure
rate in Condition C and D, non-null token counts for Condition D, and a
compute-matching deviation inside tolerance.

## Part 4 — Full run

```
python pipeline.py --full
```
Hours to days depending on task count. Runs A, C, D, B in that order
(B needs D's budget), then produces everything.

## Part 5 — Re-analyse without re-running inference

```
python pipeline.py --analyse-only
```
Regenerates evaluation, statistics, figures, tables and the PDF from the
raw records already on disk.

---

## Individual commands

| Command | What it does |
|---|---|
| `python reproduce.py` | environment report, writes hardware details for Table 4 |
| `python smoke_test.py` | one task, all conditions |
| `python run_task.py --first 4` | 4 tasks, all conditions, side-by-side |
| `python run_task.py --task Q001 --parallel` | conditions run concurrently |
| `python run_experiment.py --all` | main experiment |
| `python evaluate.py` | grade answers |
| `python analyze.py` | statistics + mechanism analysis |
| `python make_figures.py` | figures and LaTeX tables |
| `python make_report.py` | the PDF comparison report |
| `pytest -q` | 25 unit tests |

## Benchmark preparation (for the 300-question set)

```
python convert_benchmark.py --xlsx agent_benchmark_300_questions.xlsx
python build_corpus.py scaffold --xlsx agent_benchmark_300_questions.xlsx
    ... write the corpus documents ...
python build_corpus.py build
python gold_facts.py template --xlsx agent_benchmark_300_questions.xlsx
    ... fill gold_template.xlsx ...
python gold_facts.py merge --template gold/gold_template.xlsx
python validate_benchmark.py        # must pass before the main run
```

## Freeze the pre-registration first

Before the full run, decide every value marked `# PREREGISTER` in
`config.yaml`, then:
```
git add config.yaml
git commit -m "Pre-registration frozen <date> before main run"
```
This is what proves the thresholds were not chosen after seeing results.

---

# Running on several laptops (team workflow)

## The one rule

**Each laptop runs ALL FOUR conditions on a DIFFERENT SUBSET OF TASKS.**

Do **not** give one condition to each person. Every contrast in this
study is a paired difference on the same task:

| Contrast | What it measures |
|---|---|
| D − B | collaboration benefit at equal compute |
| C − A | architecture-attributable overhead |
| D − C | framework-attributable overhead |

If the two halves of a difference run on different laptops, that
difference contains the hardware gap between the machines as well as the
effect you want, and no amount of later analysis can separate them.
"Same specifications" is not the same machine: thermal behaviour,
background load, driver versions, memory timing and power state all
differ. Sharding by task keeps every pair inside one machine, so the
machine effect cancels within each difference.

`merge_shards.py` enforces this and **refuses to merge** if any task's
conditions are split across machines.

## Coordinator: split the work

```
python shard_tasks.py --machines 4 --task-set tasks/tasks_300.json
```

Produces `shards/machine_1..4/` with each person's `tasks.json`, an
identical `config.yaml`, and a `RUN_ME.txt` with their exact commands.
Tiers are balanced across machines. It also creates
`shards/calibration_tasks.json` — a handful of tasks **everyone** runs,
which is how you measure the leftover difference between machines.

Send each person the whole project folder plus their `machine_N` folder.

## Each person: run your shard

```
python pipeline.py --check
python pipeline.py --full --task-set shards/machine_2/tasks.json
python run_experiment.py --all --task-set shards/calibration_tasks.json
```

Then zip your `results/` folder and send it back as
`machine_2_results.zip`.

**Nobody edits `config.yaml`.** The config hash is checked at merge time;
if it differs, the records were not produced under the same conditions
and cannot be pooled.

Everyone must also install the same package versions and pull the same
two models.

## Coordinator: merge and analyse

```
python merge_shards.py --inputs machine_1_results machine_2_results machine_3_results machine_4_results
python pipeline.py --analyse-only
```

The merge runs five checks and refuses to proceed on a hard failure:

1. **Config hash identical** across machines
2. **Hardware fingerprints** reported (differences warn, not fail)
3. **No task spans machines** — hard failure
4. **No duplicate task-condition-run triples** — hard failure
5. **Calibration comparison** — the same tasks on every machine, so the
   between-machine spread in tokens, latency and calls is measured

A spread above 25% on any metric is flagged. **Report the calibration
spread in the paper** as the residual between-machine effect — that is
the honest way to handle distributed collection, and a reviewer who
notices you used four laptops will look for exactly this.

Merged records carry `machine_id`, so machine can be included as a factor
in the analysis if the calibration shows it matters.

## What this buys you

Four laptops give roughly a 4× wall-clock speedup with no methodological
cost, because the confound is designed out rather than corrected for
afterwards. The 80–120 hour full run becomes 20–30 hours per person.

## What it does not fix

Absolute levels still differ between machines. Paired contrasts are safe;
statements like "Condition D takes 240 seconds" are machine-specific and
should be reported with the calibration spread attached.
