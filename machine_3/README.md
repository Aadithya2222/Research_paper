# Coordination Tax Benchmark

Code for the study *"When Does Role-Based Collaboration Beat Compute-Matched
Sampling? A Framework-Controlled Study of Multi-Agent LLM Orchestration on
Resource-Constrained Local Hardware."*

Everything runs **locally on your own laptop**. No cloud account, no API key,
no cost.

---

## What the four conditions are

| Condition | What it is | What it answers |
|---|---|---|
| **A** | One ReAct agent, does everything itself | naive baseline |
| **B** | The same agent run N times, then pick the best — **budget matched to D** | Is collaboration better than just spending the same compute? (RQ1) |
| **C** | Planner → Researcher → Writer, in LangGraph | architecture cost = C − A (RQ2) |
| **D** | Planner → Researcher → Writer, in CrewAI | framework cost = D − C (RQ2) |

**The order matters.** D must run before B, because B reads D's token budget to
decide how many times to run. The code enforces this — B refuses to start
without D's budget file.

---

## Part 1 — Install everything (once)

### Step 1. Install Python

Download Python **3.11 or newer** from <https://www.python.org/downloads/>.

On the first install screen, tick the box that says
**"Add Python to PATH"**. This matters. If you miss it, nothing below works.

Check it worked. Open **Command Prompt** (press the Windows key, type `cmd`,
press Enter) and type:

```
python --version
```

You should see something like `Python 3.11.9`. If you see an error, reinstall
Python with the PATH box ticked.

### Step 2. Put the project somewhere sensible

Unzip the project folder to, for example, `C:\research\ctb`.

In Command Prompt, go into that folder:

```
cd C:\research\ctb
```

**Every command from here on is typed in this same window, in this folder.**
If you close the window, reopen it and `cd` back here first.

### Step 3. Make a virtual environment

A virtual environment keeps this project's packages separate from the rest of
your computer.

```
python -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`. **You must see `(.venv)` before
running anything else.** If you open a new window later, run
`.venv\Scripts\activate` again.

### Step 4. Install the Python packages

```
pip install -r requirements.txt
```

Takes a few minutes. Some warnings in yellow are normal; red errors are not.

### Step 5. Install Ollama (this runs the AI model on your laptop)

1. Go to <https://ollama.com/download>
2. Download the Windows installer, run it
3. After installing, Ollama runs in the background automatically

Check it works:

```
ollama --version
```

### Step 6. Download the models

The main model is about 4.7 GB, the judge about 4.4 GB. Download both:

```
ollama pull llama3.1:8b
ollama pull qwen2.5:7b-instruct
```

This takes 10–40 minutes depending on your internet. Do it once.

> **Why two models?** The judge grades the answers. If the judge were the same
> model that wrote the answers, it would favour its own writing. That is called
> self-preference bias and it would invalidate your quality results. The code
> **refuses to run** if you set both to the same model.

### Step 7. Check everything is ready

```
python reproduce.py
```

This prints your CPU, RAM, GPU, and which packages are installed. It should end
with `PASS`. It also writes `results/environment.json` — **that file contains
the hardware details that go in Table I of your paper.**

---

## Part 2 — Test run (do this before anything else)

```
python smoke_test.py
```

This runs **one** task through **every** condition. It takes a few minutes and
tells you whether the whole pipeline works. It writes to `results_smoke/`, so it
cannot contaminate your real data.

You want to see `[ OK ]` on every line. Common problems:

| Message | What to do |
|---|---|
| `Ollama unreachable` | Open a second Command Prompt and run `ollama serve`, leave it open |
| `model 'llama3.1:8b' missing` | Run `ollama pull llama3.1:8b` |
| `crewai not installed` | Run `pip install crewai` (Condition D needs it) |
| `judge model missing` | Run `ollama pull qwen2.5:7b-instruct` |

**Do not run the full benchmark until the smoke test passes.** Otherwise you
will discover a bug six hours into a run.

---

## Part 3 — Run the experiment

### The easy way (recommended)

```
python run_experiment.py --all
```

This runs A, then C, then D, then B, in the correct order, automatically.

**This takes hours.** With 12 tasks it is roughly 1–3 hours on a 16 GB laptop.
Leave the laptop plugged in and don't let it sleep.

### One condition at a time

If you'd rather do it in stages (safer — you can stop between them):

```
python run_experiment.py --condition A
python run_experiment.py --condition C
python run_experiment.py --condition D
python run_experiment.py --condition B
```

**Keep this order.** If you run B first you get an error telling you to run D
first. That error is deliberate, not a bug.

### Useful options

```
python run_experiment.py --condition A --limit 2          # only 2 tasks (quick test)
python run_experiment.py --condition A --complexity-tier 1  # only Tier 1 tasks
python run_experiment.py --condition A --runs 3           # 3 repeats per task
```

Use `--limit 2` first. Confirm the output looks sane, then run the full thing.

---

## Part 4 — Get your results

Three commands, in this order:

```
python evaluate.py
python analyze.py
python make_figures.py
```

* `evaluate.py` — grades every answer, scores role fidelity, classifies failures
* `analyze.py` — runs the statistics and **prints your headline result to screen**
* `make_figures.py` — draws all figures and writes all tables

### Where your outputs are

```
results/
  raw/            every single run, one JSON line each  <- your evidence
  processed/      graded results + the compute budget
  statistics/     statistics_report.json                <- all your numbers
  figures/        PNG + PDF + SVG, IEEE column widths   <- paste into the paper
  tables/         CSV + Markdown + LaTeX                <- .tex pastes straight in
```

The `.tex` files in `results/tables/` paste directly into your IEEE manuscript.

---

## Part 5 — Before you run the real experiment

**Freeze your pre-registration first.** Open `config.yaml`, decide the values
marked `# PREREGISTER`, then commit the file to git with a dated message:

```
git add config.yaml
git commit -m "Pre-registration frozen 2026-08-25 before main run"
```

This is what proves you did not choose your complexity thresholds or
significance criteria *after* seeing which way the results went. Without it, a
reviewer is entitled to assume you did.

The values that must be frozen: complexity weights and cut points, the compute
matching tolerance, the minimal important differences, the model temperature and
seed, and the max iteration cap.

---

## Everything you can change

All settings live in **`config.yaml`**. Nothing is hardcoded in the Python files.

The most useful ones:

```yaml
model:
  name: "llama3.1:8b"      # the model under study
  temperature: 0.0         # 0.0 = same answer every time (main run)
  num_ctx: 8192            # context window, identical for all conditions

experiment:
  runs_per_task: 1         # set to 3 for the stochasticity substudy
  max_iterations: 8        # how many ReAct steps before giving up

compute_matching:
  token_tolerance_percent: 10.0   # how close B must be to D's budget
  max_n: 5                        # cap on Best-of-N
```

---

## Adding your own tasks

Tasks live in `tasks/tasks.json`. The 12 supplied are a **starter set to prove
the pipeline works** — your paper needs roughly 210 (70 per tier), with at least
half adapted from established public benchmarks.

Each task looks like this:

```json
{
  "task_id": "T2_005",
  "task_text": "The question the agent must answer.",
  "evaluation_type": "rubric",
  "expected_answer": null,
  "gold_facts": ["fact one", "fact two"],
  "required_tools": ["search_corpus"],
  "complexity_factors": {
    "n_tool_calls": 2,
    "n_evidence_sources": 2,
    "dependency_depth": 1,
    "verification_required": 0,
    "synthesis_required": 1
  },
  "provenance": "adapted",
  "human_verified": true
}
```

You do **not** set the tier. The code computes it from
`complexity_factors` using the weights in `config.yaml` (Eq. 1 in the paper). If
you write a tier that disagrees with the computed one, the code warns you and
uses the computed one.

Documents the agents can search live in `tasks/corpus.json`. Local files only —
no web search, deliberately, so the benchmark is reproducible and network
latency does not pollute your latency measurements.

---

## Running the tests

```
pytest -q
```

25 tests covering token accounting, compute matching, complexity scoring,
effect sizes and multiplicity correction — the places where a silent bug would
quietly corrupt your conclusions.

---

## Design decisions that protect your results

These are deliberate. Please do not "fix" them.

1. **B cannot see D's quality.** The budget file contains only token and call
   counts. `write_budget()` raises an exception if any quality-like field
   appears in it. This is what makes the matching unbiased.

2. **Condition D refuses to fall back.** If CrewAI is not installed, D raises an
   error instead of quietly using the Condition C code. A silent fallback would
   make the D−C framework contrast measure nothing while still producing
   plausible-looking numbers.

3. **Nothing is ever estimated.** Token counts come from Ollama's own
   tokenizer. If a count is unavailable it is recorded as `null`, never guessed
   from character length. Same for GPU metrics on machines without a GPU.

4. **Retries are counted.** Every failed attempt goes into the ledger with its
   tokens and latency. Hiding retries would understate the true cost.

5. **Single-agent conditions get the union of all three role charters.**
   Otherwise the multi-agent conditions would receive more instruction text, and
   you would be measuring prompt length rather than architecture.

6. **Failures are never discarded.** Every failure gets a type and stays in the
   dataset.

7. **The benchmark refuses to run an unfair comparison.** `validation/checks.py`
   raises rather than continuing. A crash costs you an afternoon; a silently
   unfair comparison costs you the paper.

---

## Known limitations (state these in your paper)

* **CrewAI token counts** depend on the installed version exposing
  `usage_metrics`. If your version does not, Condition D's token totals will be
  `null` and you must report that honestly rather than substituting an estimate.
* **TTFT** is measured only for the first streamed call in a run.
* **RAM measurement** samples this process plus any process named `ollama`. On a
  machine where Ollama runs in a container this will undercount.
* **One hardware platform.** Report 16 GB as a deployment scenario, not as a
  controlled variable.
* **The 12 supplied tasks are a pipeline test, not a benchmark.** Scale to ~210
  before drawing conclusions.

---

## Quick reference

```
python reproduce.py                        # check the environment
python smoke_test.py                       # test with 1 task
python run_experiment.py --all             # run everything (hours)
python evaluate.py                         # grade the answers
python analyze.py                          # statistics
python make_figures.py                     # figures and tables
pytest -q                                  # run the tests
```

---

# Per-task evaluation (`run_task.py`)

`run_experiment.py` runs one condition across every task. That is the
right shape for the main experiment, but it gives you no feedback for
hours. `run_task.py` does the opposite: **one task through all four
conditions**, with a side-by-side comparison printed immediately.

```
python run_task.py --task Q001              # one task, all 4 conditions
python run_task.py --tasks Q001,Q002,Q003,Q004
python run_task.py --first 4                # first 4 tasks
python run_task.py --first 4 --tier 3       # first 4 Tier-3 tasks
python run_task.py --task Q001 --parallel   # conditions run concurrently
python run_task.py --first 4 --save         # also append to results/raw/
```

Output per task: a table of success / tokens / calls / tools / iterations
/ latency / peak RAM for A, B, C and D; a compute-matching check against
Eq. (3) with the realised token deviation; the single-task overhead
decomposition; any failures or degraded stages; and the first 180
characters of each condition's answer so you can see what actually
differed.

## Sequential vs parallel

**sequential (default)** — conditions run one after another, so only one
model request is in flight. Latency, TTFT and memory are measured
cleanly. **Use this for any number that goes in the paper.**

**parallel (`--parallel`)** — conditions run concurrently against the
same Ollama server. Faster in wall-clock terms, but they contend for CPU,
GPU and RAM, so those figures are contaminated. Records are tagged
`timing_valid: false` and the statistics pipeline automatically excludes
them from all timing and resource metrics. Token counts and quality are
unaffected and are still used.

On a 16 GB machine, four concurrent 8B requests will likely swap. Parallel
mode is for quickly checking that all four conditions still run, not for
measurement.

---

# What changed in this revision

Two bugs surfaced by the first real run, and three gaps against the paper.

**Bug: tool arguments were rejected when quoted.** The model emits
`Action Input: "25 + 40 / 2"` and the strict validator rejected the
quotes. The agent then "simplified" and retried the same call until it
hit the iteration cap. This caused **14 of 18 Condition C runs to fail**
with `researcher_failure` — a tool-parsing artefact being recorded as an
architecture failure. Tool arguments are now stripped of quotes,
backticks and code fences, and the error message tells the model exactly
what format to use.

**Bug: no loop detection.** An agent repeating an identical failing
action would burn every remaining iteration. The loop now stops after the
same action-plus-input occurs three times, and records `repeat_loop` as
the stopped reason so these are visible in the failure analysis rather
than silently inflating timeouts.

**Partial evidence is now salvaged.** If the researcher stalls but has
gathered observations, those are passed to the writer flagged as
`PARTIAL EVIDENCE` with `researcher_degraded: true` recorded. A real
deployment would pass on what it has, and this stops a tool-level stall
from being counted as an architecture-level failure. It is flagged, not
hidden — the mechanism analysis counts degraded runs separately.

**New: `analysis/mechanism.py`.** Table 9 (failure stage by MAST
category), the H3 test (per-role output quality against cumulative input
context, Spearman plus OLS slope), the H4 test (role bleed against task
failure, Fisher's exact), and the Table 10 role-fidelity breakdown. These
were specified in the manuscript with no code behind them.

**New: every numbered paper table.** `make_figures.py` now emits Tables
1, 2, 4, 5, 7, 8, 9, 10 and 11 as CSV, Markdown and LaTeX. The `.tex`
files paste directly into the manuscript.

**New: compute-matching verification.** Eq. (3) declares a tolerance, but
nothing previously checked whether the realised spend landed inside it.
`run_task.py` now reports the token and call deviation per task and flags
pairs that fall outside tolerance, which is what you need to report the
exclusion rate the paper promises.

## Still not implemented, deliberately

`table10_role_fidelity` leaves the CRAS correlation and human-agreement
cells null. Those require a separately annotated subsample that does not
exist yet. **Do not report RFS as validated until they are filled** — the
code will not fill them for you, because there is nothing to compute them
from.
