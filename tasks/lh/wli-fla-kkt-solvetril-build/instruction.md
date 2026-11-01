# Task: implement the chunk-local WY / UT transform (two coupled files)

This is a scoped, multi-file implementation task in a real subsystem of
**fla-org/flash-linear-attention** (a flagship open-source LLM-infra kernel library).
The subsystem is the **chunk-local WY / UT transform**

```
T = (I + A)^{-1},   where   A[i,j] = beta_i * 2**(g_i - g_j) * (k_i . k_j)  for i > j (else 0)
```

computed independently for each contiguous chunk of tokens. This transform is the
intra-chunk sequence-mixing primitive of **delta-rule / gated-delta linear-attention**
layers: the chunked kernels build the strictly-lower-triangular matrix `A = beta * K Kᵀ`
and then invert `I + A` per chunk. It is implemented across two files that form a
**producer -> consumer** pipeline.

In the working tree the implementation bodies in the two declared scope files have been
removed, leaving stubs that raise `NotImplementedError` behind the real public signatures
and full docstring contracts. Your job is to implement BOTH from the contracts below so
that the composed transform matches the reference.

## Objective
Implement the two functions so that the composition

```python
A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, chunk_size=BT)   # producer
T = solve_tril(A)                                                  # consumer
```

produces a transform `T` that matches a held-out reference (an independent PyTorch
per-chunk inverse) within tolerance on randomized GPU inputs. This is a correctness-only task;
passing only some cases earns no partial credit.

The two files are **coupled**: `solve_tril`'s input `A` is exactly `chunk_scaled_dot_kkt_fwd`'s
output (this is precisely how `delta_rule/wy_fast.py`, `gated_delta_product/chunk.py`, and
`gated_oja_rule/chunk.py` chain them). A correct result needs both the producer's `A` and the
consumer's `solve_tril`. You must implement both.

## Editable scope (out-of-scope edits are rejected)
```
fla/ops/common/chunk_scaled_dot_kkt.py     # STAGE 1 (producer): A = strict_lower(beta * K Kᵀ)
fla/ops/utils/solve_tril.py                # STAGE 2 (consumer): T = (I + A)^{-1} per chunk
```
Only these two files may be modified. Any diff to another path marks the submission invalid.

## Exact contracts the grader invokes (implement to these precisely)

### STAGE 1 — `fla/ops/common/chunk_scaled_dot_kkt.py`
```python
def chunk_scaled_dot_kkt_fwd(k, g=None, beta=None, cu_seqlens=None, chunk_size=64,
                             output_dtype=torch.float32, chunk_indices=None) -> torch.Tensor:
    # -> A  of shape [B, T, HV, BT]  (BT == chunk_size)
```
Inputs (all CUDA):
- `k`: `[B, T, H, K]` — keys (`H` heads, head dim `K`). fp32 in the graded regime.
- `beta`: `[B, T, HV]` — per-token per-head scalar; `HV == H` in the graded regime.
- `g`: `[B, T, HV]` float32 cumulative gate, **or `None`** (`USE_G`).
- `chunk_size`: int in `{16, 32, 64}` — the chunk length `BT`.
Output `A` `[B, T, HV, BT]` float32 (`output_dtype`): for the chunk containing token `t`
(within-chunk row `i = t % BT`) and each head, `A[b, t, h, j] = beta_i * (k_i . k_j)`,
optionally scaled by `2**(g_i - g_j)` when `g is not None`, for `j < i` within the chunk,
and `0` for `j >= i` (strictly lower-triangular). Tokens past the sequence end are `0`.

### STAGE 2 — `fla/ops/utils/solve_tril.py`
```python
def solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float) -> torch.Tensor:
    # -> (I + A)^{-1}  with the same shape as A  [B, T, H, BT]
```
Input `A` `[B, T, H, BT]` — the per-chunk strictly-lower-triangular matrix from stage 1
(`A.triu() == 0`); `BT == A.shape[-1]` is only ever `16`, `32`, or `64`. Output: for each
contiguous `BT x BT` chunk block, `(I + A_block)^{-1}` (unit lower-triangular, same shape).
A correct, memory-efficient implementation tiles each `BT x BT` block into `16 x 16`
sub-blocks: invert the diagonal `16 x 16` sub-blocks, then combine the off-diagonal
sub-blocks (e.g. `Ai_21 = -Ai_22 @ A_21 @ Ai_11`). The graded regime always passes
`cu_seqlens=None` (fixed-length batch, `T` divisible by `BT`); you may support `cu_seqlens`
for parity but it is not graded.

## The reference (what you are graded against)
The composed `T = solve_tril(chunk_scaled_dot_kkt_fwd(k, g, beta, chunk_size))` is compared
to an independent pure-PyTorch per-chunk inverse of `I + strict_lower(beta * 2**(g_i-g_j) *
k kᵀ)` — the SAME convention that `tests/ops/test_solve_tril.py` asserts these kernels
against. Match it within a relative error ratio `< 0.01` (`||T_ref - T_cand|| / ||T_ref||`,
fp32). A fully correct pipeline scores 1.0.

## What is provided
- `/app/repo` — the full repository tree at the frozen baseline, importable at that path
  (put first on `sys.path`). The two scope files ship hollowed.
- A public smoke you can run: `bash tests/test.sh` (drives the scope and prints a JSON
  verdict). The hidden grading suite uses the same contract with held-out inputs (varied
  `B, T, H, D`, `chunk_size in {16,32,64}`, gated vs ungated).
- The Triton kernels the pristine implementation used are a reasonable target, but ANY
  implementation (Triton or a correct pure-PyTorch one) that meets the contract and
  tolerance scores — you are graded on the output, not the method.

## How you are graded
- Final reward is binary: 1 if the submission builds, imports, matches the reference transform `T`
  within tolerance on every case, and passes all scope and integrity checks; otherwise 0.
  Preserve both public signatures and their return contracts exactly.
- `chunk_size == 16` cases exercise mainly the producer (a single `16x16` block; no
  off-diagonal merge). `chunk_size in {32, 64}` cases require the consumer's off-diagonal
  sub-block merge. A fully correct producer + consumer (incl. the gate and `chunk_size in
  {16,32,64}`) scores 1.0.

## Academic-integrity clause (violations score 0 for the whole task)
- The grader, its reference outputs, and all held-out inputs live outside your workspace.
  Do not read, `cat`, `grep`, import, reconstruct, or modify the grader or any scoring
  asset, and do not reproduce the harness.
- Do not fetch the upstream repository, its git history, PRs, or any external reference
  implementation of these operators in ANY form (git clone/fetch/pull, adding remotes, pip
  download/install, wget/curl, or web lookup).
- Only files in the declared editable scope may be modified; an out-of-scope diff marks the
  submission invalid.
- Do not attempt to reach the network or bypass the sandbox isolation.

## Submission (single submission — this is a correctness task)

This task is graded on **correctness only**: your solution either satisfies every
hidden test case (full credit) or it does not (zero). There is **no speedup to
optimize and no iteration loop** — you get **exactly ONE submission**.

1. **Edit** `/app/repo/fla/ops/common/chunk_scaled_dot_kkt.py` and `/app/repo/fla/ops/utils/solve_tril.py` until you believe they fully satisfy the
   behavioral contract above.
2. **Submit** by leaving your edits in the working tree — no `git commit` is needed
   (grading diffs the working tree against the baked baseline commit; the repo's HEAD
   must stay on that initial commit). Grading runs `tests/test.sh` against your
   `/app/repo` **once**; a correctness task does not reward iteration.

### 🔴 Red line (hard — ANY one of these ⇒ the whole task scores 0 in audit)

The scoring machinery is protected and off-limits. Solve this task
**independently**: rely ONLY on the code in `/app/repo`, your own knowledge, and
your own reasoning.

- Do **NOT deliberately** read, `cat`, `open`, `less`, `head`/`tail`, `grep`,
  `find`, or edit ANY verifier / correctness / scoring / hidden-test file or
  directory, wherever it lives. Leaving your edits in the working tree is the ONLY
  sanctioned way to submit.
- Do **NOT** run the verifier directly or try to reproduce / reverse-engineer
  it — leave your changes in `/app/repo` and let grading score them.
- Do **NOT** search for, print, or infer the hidden test cases, seeds,
  thresholds, or the metric definition from any source.
- Do **NOT** fetch or look up an upstream implementation or a reference
  solution (no web search, no `git` remote fetch, no network to the source),
  whether internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring /
  source material, however you came to see it.
- Do **NOT** route around the task by importing or calling a prebuilt equivalent of the function you are asked to implement; build the computation yourself in the scope files.

Stay inside `/app/repo` and scratch directories you created yourself; use only
what `submit` returns for scoring signal.

### 🎓 Explain for a beginner

Before you edit, explain your **approach** to the contract. After you have
finished editing and before you submit, give a short step-by-step walkthrough of
*why* your implementation satisfies each clause of the behavioral contract,
written so an undergraduate can follow.

## 提交方式

把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `git commit`
(判分读工作树;仓库的 HEAD 必须停在初始基线 commit 上)。
