# Performance Optimization Task

## Objective
This repository powers a privacy-preserving (secure-aggregation) server. Clients additively mask
their model updates and split each value into Shamir secret shares distributed to a committee of
shareholders; the server later reconstructs the aggregated per-coordinate values from the committee's
shares. The implementation under `secureagg/` is functionally correct but **slow** when many values
are reconstructed from the SAME committee (as happens every aggregation round, once per model
coordinate).
Make it **faster** on the benchmark workloads while preserving its exact observable behavior. Finding
*where* and *why* it is slow, by reading and profiling the code inside the scope, is part of the task.

## Editable scope
You may modify **only** these files (any edit outside them scores zero):
```
secureagg/shamir.py
secureagg/aggregator.py
```
Everything else under `/app/repo` is out of scope and read-only — in particular `secureagg/field.py`
defines the prime-field arithmetic your results are specified against and must not be changed.

## Driven subsystem and contract
All values are residues modulo the field prime `secureagg.field.DEFAULT_PRIME`. `field.py` provides
`inv_mod`, `eval_poly`, and `lagrange_at_zero` and is a fixed contract.

`Shamir(prime=DEFAULT_PRIME)` — these must hold **exactly**:
- `split(secret, n, t, rng)` -> list of `n` shares `(x_i, y_i)` with `x_i` in `1..n`, drawing the
  `t-1` random higher coefficients from `rng` (a `random.Random`) in order via `rng.randrange(prime)`
  so results are deterministic given `rng`. Raises `ValueError` unless `1 <= t <= n`.
- `reconstruct(shares)` -> the value at `x=0` of the polynomial through `shares` (Lagrange
  interpolation over the field). Raises `ValueError` on empty input or duplicate `x`-coordinates.
- `reconstruct_many(share_lists)` -> `[reconstruct(s) for s in share_lists]`.

`SecureAggregator(shamir=None)` — these must hold **exactly**:
- `aggregate(coord_shares)` -> `[shamir.reconstruct(s) for s in coord_shares]`, where every share-list
  in `coord_shares` uses the SAME set of `x`-coordinates (the committee).
- `aggregate_rounds(rounds)` -> `[aggregate(r) for r in rounds]`.
- `committee_of(coord_shares)` -> the sorted tuple of `x`-coordinates used (empty tuple when empty).

Preserve the class/method names and signatures, and the results above for every input — thresholds
from 1 to n, exact-threshold and extra shares, multiple committees, multi-round reuse, the secret
values 0 and prime-1, and the documented `ValueError`s.

## Workload shape
The benchmark reconstructs a large model's per-coordinate aggregates over several rounds from one
committee, and times it; correctness is checked over many diverse sharing scenarios against an
independent in-harness reference. A hidden set covers more cases. No model weights, GPU, or network
are involved — this is host-side logic over a prime field.

**Final reward.** Let `s` be the baseline time divided by the candidate time on the final workload;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

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

1. **Edit** `/app/repo/secureagg/shamir.py` and `/app/repo/secureagg/aggregator.py` — apply ONE concrete change based on your current
   hypothesis (round 1: based on reading the baseline code).
2. **Submit** by running `bash /opt/loop/submit.sh` **once** — it returns
   development feedback, including `dev_speedup`, `dev_reward`, `best_so_far` and the remaining budget.
3. **Read the feedback.** If the submission fails a development check, use the reported reason
   to guide your next edit. Otherwise compare `dev_speedup` across rounds; `best_so_far`
   identifies the submission currently retained by the development scorer.
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

There is no penalty for stopping after one or more submissions. Of the submissions that pass
all development checks, the harness selects one with the highest `dev_reward` for final grading.
Further rounds may change which candidate is selected; the development score is not the final reward.

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
