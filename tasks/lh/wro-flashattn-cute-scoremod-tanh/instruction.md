# Performance Optimization Task

## Objective
A subsystem in this repository computes a **multi-head attention forward pass with a user-provided
score modification**: given query, key, and value tensors and a callable supplied by the caller, it
produces, for every query position, the softmax-weighted combination of the values according to the
query-key dot products with that callable applied to each (already `softmax_scale`-multiplied) score
before the softmax (optionally restricted to earlier positions for causal attention, and optionally
with fewer key/value heads than query heads). It is functionally correct but **slow**. Make it
**faster** on the benchmark workloads while preserving its numerical behavior (outputs within
tolerance of the reference). Finding *where* and *why* it is slow, by reading and profiling the code
inside the scope, is part of the task.

## Editable scope
You may modify **only** this file (any edit outside it scores zero):
```
flash_attn/cute/interface.py
```
Everything else under `/app/repo` is out of scope.

## Entry point and contract
The verifier drives the subsystem through the public entry point
`flash_attn.cute.flash_attn_func(q, k, v, causal=..., softmax_scale=..., score_mod=...)`. The
`score_mod` argument is a callable the verifier constructs and passes in; it must be honoured on
the scaled scores before the mask and softmax (its concrete effect is described below):

- `q`: `(batch, seqlen_q, num_heads_q, head_dim)` query tensor (fp16 or bf16).
- `k`, `v`: `(batch, seqlen_k, num_heads_kv, head_dim)` key / value tensors (same dtype as `q`).
  `num_heads_q` is an integer multiple of `num_heads_kv` (grouped-query attention; each key/value
  head is shared by `num_heads_q // num_heads_kv` consecutive query heads). `seqlen_q` and
  `seqlen_k` may differ (cross-attention).
- `causal` (bool): if true, query position `i` may only attend to key positions `j <= j0 + i`,
  where `j0 = seqlen_k - seqlen_q` (the standard bottom-right causal alignment); if false, every
  query attends to every key.
- `softmax_scale` (float or None): the multiplier applied to the query-key dot products before the
  score modification and softmax; when `None` it defaults to `1 / sqrt(head_dim)`.
- `score_mod`: a callable the verifier supplies. It receives the (batch, heads, seqlen_q, seqlen_k)
  block of scale-multiplied scores together with per-element batch / head / query-position /
  key-position indices, and returns a tensor of the same shape in which every score has been
  replaced by a scalar-elementwise, position-independent, smooth and monotone gated-tanh
  reshaping of itself. The exact transform is whatever the supplied callable computes; it is
  fixed for this subsystem, and the same transform is what the correctness reference applies.
  The transform is applied to the scaled scores BEFORE the optional causal mask and the softmax.
  The entry point must respect this argument — it must actually apply the transform the callable
  defines, not an identity, a different transform, or an approximation of it.
- returns: the attention output `o` with shape `(batch, seqlen_q, num_heads_q, head_dim)` and the
  same dtype as `q`. (The entry also accepts other optional arguments; the benchmark only exercises
  the arguments described above and leaves the rest at their defaults.)

For each query position the result is the value-mix of the softmax over the score-modified,
`softmax_scale`-multiplied query-key dot products: the `score_mod` transform is applied first,
then the causal restriction on the set of `j` when `causal` is true, then the softmax. The
returned output must match the reference within a **3% relative L2** tolerance (and an 8%
relative max-abs bound) on each hidden case. Preserve the entry point's signature and this
behavior.

## Workload shape
Correctness is checked over many hidden cases spanning different batch sizes, sequence lengths
(including `seqlen_q != seqlen_k`), query/key-value head counts (including grouped-query), head
dimensions, causal and non-causal attention, half-precision dtypes, and softmax scales. The timing
workload runs one forward over a single long-sequence batch with many heads.

**Final reward.** Let `s` be the baseline time divided by the candidate time on the final workload;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

## Rules
- Preserve the entry point's signature and numerical behavior (within tolerance), including the
  gated-tanh score-mod transform applied to the scaled scores, causal masking, the grouped-query
  head sharing, and the `softmax_scale` semantics.
- **Implement the forward yourself within the scope file.** Do not import or call an external /
  prebuilt / compiled attention implementation to stand in for it (the submitted file may not add
  references to `scaled_dot_product_attention`, `flash_attn_2_cuda` / `flash_attn_3_cuda`, the
  compiled `flash_attn_interface`, `xformers`, `sageattention`, or `vllm` — including inside
  comments). Any such occurrence scores zero.
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

1. **Edit** `/app/repo/flash_attn/cute/interface.py` — apply ONE concrete change based on your current
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
- Do **NOT** call or reference the banned primitives (`scaled_dot_product_attention`, `flash_attn_2_cuda`, `flash_attn_3_cuda`, `xformers`, `import flash_attn_interface`, `from flash_attn.flash_attn_interface`, `vllm`, `sageattention`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

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
