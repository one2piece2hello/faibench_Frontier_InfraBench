# Optimize a windowed decay-attention subsystem

## Objective

`fla/ops/wall_attn/parallel.py` provides a single public operation,
`parallel_wall_attn`, a causal **sliding-window attention with a per-channel
multiplicative decay**. Given queries, keys, values, and a per-channel log-decay `g`,
it forms a log-space prefix `P = cumsum(g) * (1/ln2)` and, for query `i` and key `j`,
a logit

```
logit(i, j) = scale * (1/ln2) * sum_n q[i,n] * k[j,n] * exp2(P[i,n] - P[j,n])
```

with `scale = K**-0.5`, restricted to the causal sliding window `0 <= i - j < W`. The
weights are a softmax over the visible keys (in exp2 space) and the output is
`o = weights @ v` (grouped-query aware, `G = HQ // H`).

The current implementation is **functionally correct but slow**: it materializes the
full dense pairwise-score tensor over all key positions and softmaxes densely, so its
work and memory grow with `T^2` while the useful windowed work grows only with `T * W`.

Your job is to make `parallel_wall_attn` **as fast as possible on the benchmark
workload** while preserving its public behaviour and numerical contract exactly. The
reference result is the windowed decay-softmax attention above evaluated in fp32; your
output must match it within the harness tolerance across a range of shapes.

## Editable scope

You may edit **only** this file:

```
fla/ops/wall_attn/parallel.py
```

Any change to files outside this scope causes the whole task to score zero. The public
entry point and its input/output contract must remain unchanged:

```python
from fla.ops.wall_attn import parallel_wall_attn

o = parallel_wall_attn(q, k, v, g, window_size=W)
# q [B,T,HQ,K], k [B,T,H,K], v [B,T,H,V], g [B,T,HQ,K] -> o [B,T,HQ,V]
```

- `q`: `[B, T, HQ, K]`, CUDA (bf16 in the benchmark); `k`: `[B, T, H, K]`; `v`:
  `[B, T, H, V]`; `g` (per-channel log-decay): `[B, T, HQ, K]`, same dtype.
- `window_size = W` is the causal sliding-window width; the scored regime uses
  `g_scalar = sink_bias = None`, `cu_seqlens = None`, `scale = K**-0.5`.
- returns `o`: `[B, T, HQ, V]`, equal to the windowed decay-softmax attention above.
- Behaviour must be preserved across **long context**, **varying window width `W`**
  (including `W >= T`, i.e. full-causal), varying batch `B`, varying head counts with
  grouped-query attention (`HQ >= H`, `G = HQ // H`), and head widths `K`, `V`
  (e.g. `∈ {64, 128}`). The operation must be correct for arbitrary context length `T`,
  not only the benchmark length. Do not special-case the public workload.
- Determinism: given the same inputs, results must be stable across runs.

Helper primitives already present in the surrounding package (outside your scope) are
available for you to call.

## What is measured

- **Correctness** (hard gate): the operation's output must match the independent fp32
  reference within a relative-norm tolerance, over a hidden suite of diverse shapes
  (short and long `T`, several non-power-of-two `T`, varying window width, grouped-query
  head configurations, varying `B`, and both head widths).
  Every tested shape must pass; partial correctness earns no reward.
- **Measured speedup**: wall-clock speedup of your implementation over the frozen
  baseline currently in `fla/ops/wall_attn/parallel.py`, measured on a long-context
  workload with warmup and repeated timed iterations.

**Final reward.** Let `s` be the candidate's measured speedup over the baseline and `r` the reference
implementation's speedup over that same baseline. A failed correctness or validity check, `r <= 1`,
or `s <= r` gives 0. Otherwise the reward is `min(1, ln(s/r) / ln(r))`: beating the reference
earns credit, and `s >= r²` earns 1. Development speedup is feedback, not the final reward.

## Rules (A-only red line)

Solve the task **only** by improving the implementation inside the editable scope. The
following cause the whole task to score zero:

- reading, importing, copying, or reconstructing the hidden verifier, reward script, or
  timing harness, or hard-coding to their internals;
- fetching or reusing any external/upstream implementation of this operation, or
  otherwise obtaining the answer rather than deriving it (no `git clone`/`fetch`/`pull`,
  no adding a remote, no `pip install`/`download` of the same package, no `wget`/`curl`
  of upstream files, no web lookup — whether or not the network appears reachable);
- routing the computation to the package's own eager reference wrapper
  (`naive_wall_attn`) or to a custom `torch.ops` operator instead of computing the
  attention in your own code;
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

1. **Edit** `/app/repo/fla/ops/wall_attn/parallel.py` — apply ONE concrete change based on your current
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
- Do **NOT** call or reference the banned primitives (`naive_wall_attn`, `torch.ops`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

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
