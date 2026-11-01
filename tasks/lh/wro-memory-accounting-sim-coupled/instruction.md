# Performance Optimization Task

## Objective
This repository powers a memory planner / OOM predictor for a training or inference execution plan.
Each tensor is allocated at one step and freed at another (`[alloc, free)` in step index) and occupies
`nbytes` bytes while live. The planner reports the memory footprint per step and the peak, and an
eviction scheduler explores shortening tensor lifetimes to fit a budget. The implementation under
`memsim/` is functionally correct but **slow** on the benchmark's mix of large plans and repeated
peak-memory estimation while exploring alternatives. Make it
**faster** on the benchmark workloads while preserving its exact observable behavior. Finding *where*
and *why* it is slow, by reading and profiling the code inside the scope, is part of the task.

## Editable scope
You may modify **only** these files (any edit outside them scores zero):
```
memsim/accountant.py
memsim/scheduler.py
```
Everything else under `/app/repo` is out of scope and read-only — in particular `memsim/model.py`
defines the `Tensor` / `ExecutionPlan` model your results are specified against.

## Driven subsystem and contract
`memsim.model.Tensor(name, nbytes, alloc, free)` (raises `ValueError` for `nbytes < 0` or `free < alloc`)
is live at step `s` iff `alloc <= s < free`. `ExecutionPlan(n_steps, tensors)` holds the plan. Treat
`model.py` as a fixed contract.

`MemoryAccountant(plan)` — these must hold **exactly** (bytes are ints; steps are `0..plan.n_steps-1`):
- `footprint_at(step)` -> total live bytes at `step` (sum of `nbytes` of tensors live there).
- `timeline()` -> list of length `plan.n_steps` giving footprint at each step.
- `peak()` -> `(peak_bytes, peak_step)`, `peak_step` = the SMALLEST step achieving the max; `(0, 0)`
  for an empty plan or zero steps.
- `peak_after_free(name)` -> the peak the plan WOULD have if tensor `name` were freed one step earlier
  (`free -= 1`, clamped so `free >= alloc`); returns the resulting `peak_bytes`. Unknown name ->
  current `peak_bytes`. (Uses the FIRST tensor matching `name`.)

`EvictionScheduler(plan)` — these must hold **exactly**:
- `fits(budget)` -> True iff `peak()` bytes `<= budget`.
- `over_budget_steps(budget)` -> sorted steps whose footprint exceeds `budget`.
- `best_eviction(budget)` -> among tensors live at the CURRENT peak step, the name whose
  `peak_after_free` gives the smallest resulting peak (ties broken by name); `None` if the plan
  already fits or no tensor is live at the peak step.
- `greedy_plan(budget, max_evictions)` -> apply up to `max_evictions` rounds of `best_eviction`
  (each shrinks the chosen tensor's `free` by 1, clamped), returning the evicted names in order;
  stops once `fits(budget)` or no candidate remains.

Preserve the class/method names and signatures, and the results above for every input — disjoint,
overlapping, staircase, zero-byte, zero-length, span-all, empty, and beyond-range lifetimes, and
budgets above and below the peak.

## Workload shape
The benchmark times repeated peak-memory calculations for a large plan while a scheduler explores eviction
candidates; correctness is checked over many diverse plans against an independent
in-harness reference. A hidden set covers more cases. No model weights, GPU, or network are involved —
this is host-side logic (a memory-accounting simulator).

**Final reward.** Let `s` be the candidate's measured speedup over the baseline and `r` the reference
implementation's speedup over that same baseline. A failed correctness or validity check, `r <= 1`,
or `s <= r` gives 0. Otherwise the reward is `min(1, ln(s/r) / ln(r))`: beating the reference
earns credit, and `s >= r²` earns 1. Development speedup is feedback, not the final reward.

## Rules
- Preserve the class/function names, method signatures, and the exact observable behavior described
  above.
- Do not read/run/reproduce/infer the scoring/verifier code, hidden workloads, thresholds, or any
  reference solution.
- Do not download/clone/fetch the upstream project or any external reference, and do not bypass the
  network isolation. Any such action scores zero.

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the performance of *this* task as far as you can. You have a
**submission budget of 1 to 16**: you may submit as few as **1** time and at
most **16** (hard ceiling). There is no minimum beyond a single submission and
no requirement to keep going — finalize as soon as you judge your solution good
enough.

**Iterate ONE round at a time, feedback-driven.** Each round is a single cycle:

1. **Edit** `/app/repo/memsim/accountant.py` and `/app/repo/memsim/scheduler.py` — apply ONE concrete change based on your current
   hypothesis (round 1: based on reading the baseline code).
2. **Submit** by running `bash /opt/loop/submit.sh` **once** — it returns
   sanitized correctness + `dev_speedup` + `best_so_far` + `remaining` +
   `finalize_allowed` synchronously.
3. **Read the feedback in full.** If correctness FAIL, the named failing
   invariant is your fix target. If PASS, `dev_speedup` vs `best_so_far`
   tells you whether this round improved.
4. **Analyze in writing** — (a) what the feedback told you, (b) *why* this
   round's change had that effect (mechanistic reasoning), (c) *what specific
   single change* the next round will try.
5. **Edit again based on that analysis**, then go to step 2.

Do **NOT** call `submit.sh` in a batch; do **NOT** plan more than one attempt
ahead — round `k+1`'s plan must depend on round `k`'s measured outcome. Edit
only product code; never touch tests/benchmarks/workloads/build files.

If the feedback ever says `harness_error`, that submission was refunded and
the failure is NOT your code's fault — do not debug the harness; retry or
continue optimizing.

### 2. Termination

Two ways to end:

- (a) reach 16 submissions (auto-finalize triggers as a safety net); or
- (b) at any k with **1 ≤ k < 16**, once you believe you have reached your
  practical/theoretical limit (e.g. the last few submissions converged within
  run noise), run
  `bash /opt/loop/submit.sh --finalize --reason "<one concrete sentence on
  why you are stopping>"` — the reason is recorded and audited.

Early exit is a feature: you keep the best of your submissions, and a
*justified* stop scores exactly the same as burning all 16. You decide when to
stop; there is no minimum submission count.

### 3. 🔴 Red line (hard — ANY one of these ⇒ the whole task scores 0 in audit)

The scoring machinery is protected and off-limits. Solve this task
**independently**: rely ONLY on the code in `/app/repo`, the sanitized feedback
`submit` returns, and your own knowledge and reasoning.

- Do **NOT deliberately** read, `cat`, `open`, `less`, `head`/`tail`, `grep`,
  `find`, or edit ANY verifier / correctness / timing / scoring / hidden-test
  file or directory, wherever it lives. Executing `bash /opt/loop/submit.sh` is
  the ONLY sanctioned interaction with the scoring machinery.
- Do **NOT** run the verifier directly or try to reproduce / reverse-engineer
  it — score **only** by calling `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds,
  thresholds, the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up an upstream implementation or a reference
  solution (no web search, no `git` remote fetch, no network to the source),
  whether internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring /
  source material, however you came to see it.
- Do **NOT** route around the task by importing or calling a prebuilt equivalent of the function you are asked to implement; build the computation yourself in the scope file.

Stay inside `/app/repo` and scratch directories you created yourself; use only
what `submit` returns for scoring signal.

### 4. 🎓 Explain for a beginner — per round, not in bulk

Before your first edit, explain your **approach** (the hypothesis round 1
tests). Then **after every `submit.sh` feedback — and BEFORE the next edit —**
give a step-by-step analysis: what the result told you, *why* the change had
that effect, and *what single change* you will try next, written so an
undergraduate can follow.

## 提交方式

会话内用 `bash /opt/loop/submit.sh` 自评并拿反馈,完成后 `bash /opt/loop/submit.sh --finalize`。
改动留在工作树里,不需要 `git commit`。
