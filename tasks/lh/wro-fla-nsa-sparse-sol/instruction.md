# Optimize a native sparse-attention selection subsystem

## Objective

`fla/ops/nsa/parallel.py` provides a single public operation, `parallel_nsa`, the
selection path of a **native sparse attention** operator. Given per-query selected
key/value blocks, each query attends — under a causal mask — only to the key tokens
that lie inside its selected blocks:

- inputs are grouped-query: with `G = HQ // H` (a power of two, `>= 16`), query head
  `hq` uses kv head `h = hq // G`;
- `block_indices` gives, per query and kv-head, the `S` selected key/value **blocks**
  (each `block_size` contiguous key positions; an entry `< 0` is padding);
- with `scale = K**-0.5`, the output for query position `t` is
  `o_t = softmax_j(scale·⟨q_t, k_j⟩) · v_j` over `{ j : j lies in a selected block of t
  and j <= t }` (causal).

The current implementation is **functionally correct but slow**: it evaluates the
selection strictly one query at a time, so its run time grows with the context length
`T` even though the useful work per query is bounded by the `S · block_size` selected
keys.

Your job is to make `parallel_nsa` **as fast as possible on the benchmark workload**
while preserving its public behaviour and numerical contract exactly. The reference
result is the selection attention above evaluated in fp32; your output must match it
within the harness tolerance across a range of shapes.

## Editable scope

You may edit **only** this file:

```
fla/ops/nsa/parallel.py
```

Any change to files outside this scope causes the whole task to score zero. The public
entry point and its input/output contract must remain unchanged:

```python
from fla.ops.nsa import parallel_nsa

o = parallel_nsa(q, k, v, block_indices=block_indices,
                 block_counts=S, block_size=BS)
# q [B,TQ,HQ,K], k [B,TK,H,K], v [B,TK,H,V], block_indices [B,TQ,H,S] -> o [B,TQ,HQ,V]
```

- `q`: shape `[B, TQ, HQ, K]`, CUDA (bf16 in the benchmark); `k`: `[B, TK, H, K]`;
  `v`: `[B, TK, H, V]`; `block_indices`: `[B, TQ, H, S]` int64 (`< 0` = padding).
- the scored regime uses `block_counts = S` (int), `g_cmp = g_slc = g_swa = None`,
  `window_size = 0`, `cu_seqlens = None`.
- returns `o`: shape `[B, TQ, HQ, V]`, equal to the selection attention above.
- Behaviour must be preserved across **long context**, a **varying number of selected
  blocks `S`**, varying batch `B`, and head widths `K`, `V` (e.g. `∈ {64, 128}`); the
  GQA group size is `G = HQ // H ≥ 16` (a power of two). The operation must be correct
  for arbitrary context length `T` (a multiple of `block_size`), not only the benchmark
  length. Do not special-case the public workload.
- Determinism: given the same inputs, results must be stable across runs.

Helper primitives already present in the surrounding package (outside your scope) are
available for you to call.

## What is measured

- **Correctness** (hard gate): the operation's output must match the independent fp32
  selection-attention reference within a relative-norm tolerance, over a hidden suite of
  diverse shapes (short and long `T`, several non-power-of-two multi-block `T`, varying
  `S`, varying `B`, and both head widths).
  Every tested shape must pass; partial correctness earns no reward.
- **Measured speedup**: wall-clock speedup of your implementation over the frozen
  baseline currently in `fla/ops/nsa/parallel.py`, measured on a long-context workload
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
- routing the computation to the package's own eager reference wrapper (`naive_nsa`) or
  to a custom `torch.ops` operator instead of computing the selection in your own code;
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

1. **Edit** `/app/repo/fla/ops/nsa/parallel.py` — apply ONE concrete change based on your current
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
- Do **NOT** call or reference the banned primitives (`naive_nsa`, `torch.ops`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

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
