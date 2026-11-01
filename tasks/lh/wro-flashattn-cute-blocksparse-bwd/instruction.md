# Performance Optimization Task

## Objective
A subsystem in this repository computes a **differentiable block-sparse multi-head
attention** operation: given query, key, and value tensors and a description of which
pairs of `(query-block, key-block)` are permitted, it produces — for every query
position — the softmax-weighted combination of the values according to the scaled
query-key dot products, but restricted so that each query position can only attend to
key positions that lie inside one of the permitted blocks; on a `.backward(dout)` call
it also produces the exact input gradients `dq`, `dk`, `dv` of that quantity with
respect to `q`, `k`, `v`. It is functionally correct but **slow** on the training step
(forward + backward together). Make it **faster** on the benchmark workloads while
preserving its numerical behavior on both the forward output and the input gradients
(within tolerance of the reference). Finding *where* and *why* it is slow, by reading
and profiling the code inside the scope, is part of the task.

## Editable scope
You may modify **only** this file (any edit outside it scores zero):
```
flash_attn/cute/interface.py
```
Everything else under `/app/repo` is out of scope.

## Entry point and contract
The verifier drives the subsystem through the public entry point
`flash_attn.cute.flash_attn_func(q, k, v, softmax_scale=..., block_sparse_tensors=...,
block_sparse_tensors_bwd=...)` (with `causal=False` and no window / softcap / sink /
score_mod / mask_mod arguments — the block-sparse tensors are the sole masking
mechanism on the benchmark cases). The verifier calls the entry with tensors that have
`requires_grad=True`, then invokes `.backward(dout)` on the returned output.

- `q`: `(batch, seqlen_q, num_heads_q, head_dim)` query tensor (fp16 or bf16),
  `requires_grad=True`.
- `k`, `v`: `(batch, seqlen_k, num_heads_kv, head_dim)` key / value tensors (same
  dtype as `q`), `requires_grad=True`. `num_heads_q` is an integer multiple of
  `num_heads_kv` (grouped-query attention; each key/value head is shared by
  `num_heads_q // num_heads_kv` consecutive query heads). `seqlen_q` and `seqlen_k`
  may differ (cross-attention).
- `softmax_scale` (float or None): the multiplier applied to the query-key dot
  products before the softmax; when `None` it defaults to `1 / sqrt(head_dim)`.
- `block_sparse_tensors`: an instance of
  `flash_attn.cute.block_sparsity.BlockSparseTensorsTorch` naming which
  `(query-block, key-block)` pairs the forward is permitted to include. Only the
  fields relevant to this task are described here — the rest default to `None`:
    - `mask_block_cnt`: `int32` tensor, shape `(1, 1, num_m_blocks)`, giving the
      count of "partial/boundary" key-blocks per query-block. On the benchmark cases
      this is **zero everywhere** (no partial blocks — each permitted block is fully
      covered); the field is still passed to satisfy the API.
    - `mask_block_idx`: `int32` tensor, shape `(1, 1, num_m_blocks, 1)`, dummy since
      `mask_block_cnt` is all zero.
    - `full_block_cnt`: `int32` tensor, shape `(1, 1, num_m_blocks)`, giving the
      number of permitted key-blocks per query-block.
    - `full_block_idx`: `int32` tensor, shape `(1, 1, num_m_blocks, max_full_per_row)`,
      giving — for each query-block `m` — the key-block indices (in
      `[0, num_n_blocks)`) it is allowed to attend to; only the first
      `full_block_cnt[..., m]` entries per row are valid.
    - `block_size`: a `(block_q, block_kv)` tuple with the query-block size (rows of
      the partition) and key-block size (columns).
  Let `num_m_blocks = ceil(seqlen_q / block_q)` and
  `num_n_blocks = ceil(seqlen_k / block_kv)`. The allowed positions for a query row
  `i` are the union — over the first `full_block_cnt[0, 0, m]` entries of
  `full_block_idx[0, 0, m, :]` (where `m = i // block_q`) — of the key ranges
  `[n * block_kv, min((n + 1) * block_kv, seqlen_k))`.
- `block_sparse_tensors_bwd`: an instance of `BlockSparseTensorsTorch` giving the
  same allow-set information from the key-block perspective (for each key-block `n`,
  the list of query-blocks that reference it), plus (optionally) precomputed
  metadata. The verifier passes this argument alongside `block_sparse_tensors` and it
  describes the same allow-set as `block_sparse_tensors` (transposed). Only the
  `mask_block_cnt`, `mask_block_idx`, `full_block_cnt`, `full_block_idx`,
  `dq_write_order`, `dq_write_order_full`, `block_size`, and `spt` fields are
  populated on the benchmark cases; every other field is left at its default
  (`None`).
- returns: a value with either
  - the attention output `o` with shape `(batch, seqlen_q, num_heads_q, head_dim)`
    and the same dtype as `q`, **or**
  - a `(o, lse)` tuple when `return_lse` is set (the benchmark leaves `return_lse`
    at its default; the verifier accepts either form). After
    `o.backward(dout)`, `q.grad`, `k.grad`, `v.grad` must be populated with the
    exact input gradients of the forward output with respect to `q`, `k`, `v` (the
    entry also accepts other optional arguments; the benchmark only exercises the
    arguments described above and leaves the rest at their defaults).

For each query position the forward result is
`sum_j softmax_j( scale * (q · k_j) ) * v_j` taken over the set of key positions `j`
allowed by the block-sparse partition described by `block_sparse_tensors`. A query
whose allowed set is empty produces a zero output vector. `dq`, `dk`, `dv` must be
the corresponding partial derivatives of the sum, per element, of that output with
respect to `q`, `k`, `v`. The returned output and all three input gradients must
match the references within a **3% relative L2** tolerance (and an 8% relative
max-abs bound) on each hidden case. Preserve the entry point's signature and this
behavior.

## Workload shape
Correctness is checked over many hidden cases spanning different batch sizes,
sequence lengths (including `seqlen_q != seqlen_k`), query/key-value head counts
(including grouped-query), head dimensions, block-partition parameters (both
`block_q` and `block_kv` are always `128` on the benchmark; the count of allowed
blocks per query row varies), half-precision dtypes, and softmax scales. The timing
workload runs one forward + backward step over a single long-sequence sparsely-
connected batch.

**Final reward.** Let `s` be the baseline time divided by the candidate time on the final workload;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

## Rules
- Preserve the entry point's signature and numerical behavior (within tolerance),
  including the `block_sparse_tensors` allow-set restriction, the grouped-query head
  sharing, the `softmax_scale` semantics, and the exactness of the backward
  gradients under `.backward(dout)`.
- **Implement the forward and the backward yourself within the scope file.** Do
  not import or call an external / prebuilt / compiled attention implementation to
  stand in for it (the submitted file may not add references to
  `scaled_dot_product_attention`, `flash_attn_2_cuda` / `flash_attn_3_cuda`, the
  compiled `flash_attn_interface`, `xformers`, `sageattention`, or `vllm` —
  including inside comments). Any such occurrence scores zero.
- Do not read/run/reproduce/infer the scoring/verifier code, hidden workloads,
  thresholds, or any reference solution.
- Do not download/clone/fetch the upstream project or any external reference, and
  do not bypass the network isolation. Any such action scores zero.

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
- Do **NOT** call or reference the banned prebuilt/external attention primitives (`scaled_dot_product_attention`, `flash_attn_2_cuda`, `flash_attn_3_cuda`, `xformers`, `sageattention`, `vllm`, the compiled `flash_attn_interface`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

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
