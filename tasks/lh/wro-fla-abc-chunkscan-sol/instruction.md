# Optimize a two-stage gated linear-recurrence subsystem

## Objective

`fla/ops/abc/chunk.py` provides a single public operation, `chunk_abc`, exposed
through a `torch.autograd.Function`. It implements a two-stage gated linear-attention
sequence operator with a bounded slot memory. Given per-step queries, keys, values,
and slot logits, it maintains **two** matrix-valued recurrent states with a per-step
forget gate derived from the slot logits, and produces an output stream:

- the slot logits `s` are first turned into a normalized form along the sequence: a
  cumulative log-domain normalizer over time yields both the per-step **forget gate** (as the
  log-domain increment between consecutive steps of that normalizer) and the per-step **slot
  probabilities** (the logits measured against that same normalizer);
- **stage 1** maintains a key/slot state of shape `[K, M]`. Each step decays the previous state
  by the step's forget gate and adds the rank-one contribution of that step's key against its
  slot probabilities; the step's query, scaled by the usual inverse-square-root of the key
  width, is read against the state to produce per-slot scores;
- a softmax over the `M` slots turns those scores into slot weights;
- **stage 2** maintains a slot/value state of shape `[M, V]`, decayed by the same per-step gate
  and accumulating the rank-one contribution of that step's slot probabilities against its
  value; the stage-1 slot weights are read against this state to produce the output.

The current implementation is **functionally correct but slow**.

Your job is to make `chunk_abc` **as fast as possible on the benchmark workload** while
preserving its public behaviour and numerical contract exactly. The reference result is
the two-stage recurrence above evaluated in fp32; your output must match it within the
harness tolerance across a range of shapes.

## Editable scope

You may edit **only** this file:

```
fla/ops/abc/chunk.py
```

Any change to files outside this scope causes the whole task to score zero. The public
entry point and its input/output contract must remain unchanged:

```python
from fla.ops.abc import chunk_abc

o, final_state = chunk_abc(q, k, v, s, initial_state=None, output_final_state=False)
```

- `q`, `k`: shape `[B, T, H, K]`, CUDA (bf16 in the benchmark).
- `v`: shape `[B, T, H, V]`, CUDA, same dtype.
- `s` (slot logits): shape `[B, T, H, M]`, CUDA, same dtype.
- returns `o`: shape `[B, T, H, V]`, equal to the two-stage recurrence above; and, when
  `output_final_state=True`, `final_state`: a pair of states shaped `[B, H, K, M]` and
  `[B, H, M, V]` (the two states as they stand after the last step).
- `T` is an arbitrary positive integer (**not necessarily a power of two**); the
  operation must be correct for all sequence lengths, and for arbitrary batch `B` and
  head count `H`. The feature widths `K`, `V` and the slot count `M` are standard
  head-dimension hyperparameters (e.g. `K, V ∈ {64, 128}`, `M ∈ {32, 64}`).
- `chunk_abc` must remain driven through the same `torch.autograd.Function`; the forward
  maps `(q, k, v, s, initial_state, output_final_state) -> (o, final_state)`. The
  benchmark exercises the forward path only.
- Determinism: given the same inputs, results must be stable across runs.

Helper primitives already present in the surrounding package (outside your scope) are
available for you to call.

## What is measured

- **Correctness** (hard gate): the operation's output must match the independent fp32
  reference computation within a relative-norm tolerance, over a hidden suite of diverse
  shapes (short and long `T`, several non-power-of-two `T`, and varied `B`/`H` and
  feature/slot widths `K`/`V`/`M`).
  Every tested shape must pass; partial correctness earns no reward.
- **Measured speedup**: wall-clock speedup of your implementation over the frozen
  baseline currently in `fla/ops/abc/chunk.py`, measured on a long-sequence workload
  with warmup and repeated timed iterations.

**Final reward.** Let `s` be the baseline time divided by the candidate time on the final workload;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

## Rules (A-only red line)

Solve the task **only** by improving the implementation inside the editable scope. The
following cause the whole task to score zero:

- reading, importing, copying, or reconstructing the hidden verifier, reward script, or
  timing harness, or hard-coding to their internals;
- fetching or reusing any external/upstream implementation of this operation, or
  otherwise obtaining the answer rather than deriving it (no `git clone`/`fetch`/`pull`,
  no adding a remote, no `pip install`/`download` of the same package, no `wget`/`curl`
  of upstream files, no web lookup — whether or not the network appears reachable);
- routing the computation to a prebuilt/turnkey recurrence or state-space kernel from
  another module or package (e.g. a `fused_recurrent` entry point, `mamba_ssm`,
  `selective_scan`, `causal_conv1d`, a library `associative_scan`, or a custom
  `torch.ops` operator) instead of computing the recurrence in your own code;
- editing anything outside the declared scope, or weakening/bypassing the correctness
  check;
- attempting to detect the grading mode or bypass the runtime isolation.

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the performance of *this* task as far as you can. You have a
**submission budget of 1 to 16**: you may submit as few as **1** time and at
most **16** (hard ceiling). There is no minimum beyond a single submission and
no requirement to keep going — finalize as soon as you judge your solution good
enough.

**Iterate ONE round at a time, feedback-driven.** Each round is a single cycle:

1. **Edit** `/app/repo/fla/ops/abc/chunk.py` — apply ONE concrete change based on your current
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
- Do **NOT** call or reference the banned primitives (`fused_recurrent`, `mamba_ssm`, `selective_scan`, `causal_conv1d`, `associative_scan`, `accelerated_scan`, `scan_combined`, `torch.ops`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

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
