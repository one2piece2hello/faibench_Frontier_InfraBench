# Performance Optimization Task

## Objective

A state-space-model (SSM) sequence-scan subsystem in this repository is
functionally correct but slow. Your task is to **make it faster** on the
benchmark workloads while **preserving its numerical behavior** within the
tolerance stated below. All of your work must stay inside the declared editable
scope.

## Editable scope

You may edit **only** these five files (everything else is out of scope; edits
outside this scope cause the submission to score zero):

```
vllm/model_executor/layers/mamba/ops/ssd_combined.py
vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py
vllm/model_executor/layers/mamba/ops/ssd_state_passing.py
vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py
vllm/model_executor/layers/mamba/ops/ssd_bmm.py
```

Finding *where* the time goes inside this subsystem, and how to compute the same
result faster, is the task. You may add helper functions/kernels within these
files.

## Entry point and contract

The subsystem is driven through its public entry point:

```python
mamba_chunk_scan_combined(
    x, dt, A, B, C, chunk_size,
    D=None, z=None, dt_bias=None,
    dt_softplus=False, out=None, return_final_states=False,
)
```

with tensor shapes:

- `x`:  `(batch, seqlen, nheads, headdim)`
- `dt`: `(batch, seqlen, nheads)`
- `A`:  `(nheads,)`
- `B`, `C`: `(batch, seqlen, ngroups, dstate)`  (`nheads % ngroups == 0`)
- `D`:  `(nheads, headdim)` or `(nheads,)`; `z`: same shape as `x`
- `dt_bias`: `(nheads,)`; `chunk_size`: an integer power of two
- `out`: a preallocated output tensor of `x`'s shape that the call fills

**Correctness requirement.** For every benchmark input, the output written to
`out` must match the reference SSM-scan result within a **2% relative tolerance**
(bf16 inputs, fp32 accumulation; parity is judged by relative error, not exact match). Any change that alters the computed
result beyond this tolerance scores zero.

## Workload shape

The benchmark drives long-sequence **prefill scans**: a batch of sequences with
a multi-thousand `seqlen`, tens of heads, a per-head state dimension, grouped
`B`/`C` shared across heads, and a power-of-two `chunk_size`. Optimize for this
regime; the hidden benchmark also covers additional shapes.

**Final reward.** Let `s` be the median of the paired baseline-time / candidate-time ratios;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

## Rules

- Preserve the public entry point's signature and behavior (within tolerance).
- Do **not** attempt to read, run, reproduce, or infer the scoring/verifier
  code, the hidden test workloads, thresholds, or any reference solution.
- Do **not** download, clone, fetch, or otherwise obtain the upstream project,
  its history, or any external reference implementation, and do **not** bypass
  or disable the environment's network isolation. Solve the task from the code
  present in the workspace. Any such action scores the whole task zero.

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the performance of *this* task as far as you can. You have a
**submission budget of 1 to 16**: you may submit as few as **1** time and at
most **16** (hard ceiling). There is no minimum beyond a single submission and
no requirement to keep going — finalize as soon as you judge your solution good
enough.

**Iterate ONE round at a time, feedback-driven.** Each round is a single cycle:

1. **Edit** one of the five in-scope files (`/app/repo/vllm/model_executor/layers/mamba/ops/ssd_combined.py`, `/app/repo/vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py`, `/app/repo/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py`, `/app/repo/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py`, `/app/repo/vllm/model_executor/layers/mamba/ops/ssd_bmm.py`) — apply ONE concrete change based on your current
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
