# Optimize a structured K-block-sparse weight / fp16-activation matmul subsystem

## Objective

`blocksp` provides a single public operation, `blocksp_matmul(a, w_blocks, k_idx,
block_k)`, which returns `a @ W` for a half-precision activation and a **structured
K-block-sparse** weight `W` (logical shape `[K, N]`, fp16). The `K` rows are partitioned
into `num_blocks = K // block_k` contiguous row-blocks and only a subset (`nnz` of them)
are nonzero — structured input-feature-block pruning, with the same blocks pruned for
every output column. `W` is never stored densely; it is given in COMPRESSED form:

- `w_blocks` (`[nnz * block_k, N]`, `float16`): the nonzero row-blocks stacked in
  ascending block order — stored block `p` occupies rows `p*block_k : (p+1)*block_k`.
- `k_idx` (`[nnz]`, `int32`): the logical block index (in `[0, num_blocks)`, ascending,
  distinct) of each stored block.

The dense weight reconstructed from the compressed form is, for logical block
`kb = k // block_k` and in-block row `r = k % block_k`:

```
W[k, n] = w_blocks[p*block_k + r, n]   if kb == k_idx[p] for some stored block p
        = 0                            if kb is not in k_idx  (a pruned block)
```

and the result is `a @ W` reduced with an fp32 accumulator, returned as `torch.float16`.

The current implementation is **functionally correct but slow**: it does not make
efficient use of the device's memory bandwidth or of the block sparsity. Your job is to
make `blocksp_matmul` **as fast as possible on the benchmark workload** (a small-M,
large-K/N, low-block-density shape) while preserving its public signature and numerical
contract exactly. The reference result is the fp32 product of the activation and the
fully reconstructed dense weight; your output must match it within the harness tolerance
across a range of shapes and densities.

## Editable scope

You may edit **only** this file:

```
blocksp/matmul.py
```

Any change to files outside this scope causes the whole task to score zero. The public
entry point `blocksp_matmul(a, w_blocks, k_idx, block_k)` and its input/output contract
must remain unchanged:

- `a`: `torch.float16`, shape `[M, K]`, CUDA; `K` is a multiple of `block_k`.
- `w_blocks`: `torch.float16`, shape `[nnz * block_k, N]`, CUDA (nonzero K row-blocks
  stacked in ascending block order).
- `k_idx`: `torch.int32`, shape `[nnz]`, CUDA (the logical block index of each stored
  block; ascending, distinct).
- `block_k`: a positive `int` dividing `K`.
- returns: `torch.float16`, shape `[M, N]`, CUDA, equal to `a @ W` where `W` is the dense
  reconstruction above, computed with an fp32 accumulator.
- `M`, `N` are arbitrary positive integers; `K` is a multiple of `block_k`; `nnz` may be
  anything in `[1, K // block_k]`. The operation must be correct for all of them.

## What is measured

- **Correctness** (hard gate): the operation's output must match the independent fp32
  reference within a relative-norm tolerance, over a hidden suite of diverse shapes,
  block sizes, and densities (small, large, rectangular, ragged, non-power-of-two, very
  sparse, nearly dense, single-block). Every stored block must be placed at its correct
  `k_idx` logical position and every pruned block treated as zero.
  Every tested shape must pass; partial correctness earns no reward.
- **Measured speedup**: wall-clock speedup of your implementation over the frozen
  baseline currently in `blocksp/matmul.py`, measured on a small-M, large-K/N,
  low-block-density fp16 workload with warmup and repeated timed iterations.

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
  otherwise obtaining the answer rather than deriving it;
- routing the computation to a prebuilt library matmul (e.g. `torch.matmul`, `torch.mm`,
  `F.linear`, cuBLAS/CUTLASS bindings, the `@` operator) instead of computing the
  product in your own implementation;
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

1. **Edit** `/app/repo/blocksp/matmul.py` — apply ONE concrete change based on your current
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
- Do **NOT** call or reference the banned primitives (`torch.matmul`, `torch.mm`, `torch.bmm`, `F.linear`, `torch.nn.functional.linear`, `torch.einsum`, `cublas`, `cutlass`, `addmm`, `.matmul(`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

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
