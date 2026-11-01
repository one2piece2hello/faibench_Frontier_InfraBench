# Performance Optimization Task

You are working on the dense matrix-multiply kernel at the heart of a
high-throughput inference server's linear layers. The file `sgemm_kernel.cu`
implements a single-precision (float32) general matrix multiply. The Python
wrapper `sgemm.py` compiles this kernel and calls it; **you edit only
`sgemm_kernel.cu`.**

## Behavioral contract

Given `A` of shape `(M, K)`, `B` of shape `(K, N)`, `C` of shape `(M, N)`, and
two scalars `alpha` and `beta`, compute

```
D[m, n] = alpha * (sum over k of A[m, k] * B[k, n]) + beta * C[m, n]     # D has shape (M, N)
```

1. `A`, `B`, `C`: 2-D, contiguous, `torch.float32` CUDA tensors. The inner
   dimension matches (`A.shape[1] == B.shape[0] == K`); `C` has shape `(M, N)`.
2. `alpha`, `beta`: python `float`s.
3. Accumulation of the product is performed in **float32**; the returned `D` is a
   **new** `(M, N)` `torch.float32` tensor. The input `C` is read for the
   `beta * C` term but **must not be modified in place**.
4. The result must be correct for arbitrary `M`, `N`, `K` — including sizes that
   are **not** multiples of any tile width, `K = 1`, and large `K`. Row order is
   preserved.

Public entry point (invoked by the fixed wrapper; do NOT change its meaning):

```python
def sgemm(A: torch.Tensor,   # (M, K) float32 CUDA
          B: torch.Tensor,   # (K, N) float32 CUDA
          C: torch.Tensor,   # (M, N) float32 CUDA
          alpha: float,
          beta: float) -> torch.Tensor:   # D = alpha*(A@B) + beta*C, (M, N) float32
    ...
```

Error contract (enforced by the wrapper): non-`float32` or dtype-mismatched
`A` / `B` / `C` → `TypeError`; non-2-D inputs, a mismatched inner dimension, or a
`C` whose shape is not `(M, N)` → `ValueError`.

## Why the current implementation is slow

The current kernel takes the most direct possible approach: it launches **one GPU
thread per output element**, and each thread walks the entire length-`K` inner
dimension on its own, reading one row of `A` and one column of `B` straight from
global memory. Neighbouring threads that need the same rows and columns re-read
them independently, so the same values are fetched from global memory again and
again. The multiply is **compute-bound in principle**, but this kernel spends
almost all of its time moving data through global memory instead of computing —
it sustains only a small fraction of the arithmetic throughput the device is
capable of. Make it **faster on the GPU** while keeping the numerics within the
verifier's tolerance. You may use any GPU technique available in the image (better
data-reuse strategies, on-chip staging of operands, having each thread produce
several outputs, tuning launch dimensions, etc.) as long as the contract above
holds and the product accumulation stays in float32.

**Forbidden:** do not delegate the multiply to a prebuilt matrix-multiply library
or framework primitive — specifically **cuBLAS / cublasLt / cutlass** and
libtorch's matmul (`at::matmul`, `torch::matmul`, `at::mm`, `torch::mm`,
`.matmul(...)`, `.mm(...)`), as well as `torch.matmul` / `torch.mm` / `F.linear` /
the `@` operator on the Python side. The scoring harness scans your submitted
`sgemm_kernel.cu` for those tokens and scores the task 0 if present (do not
reference them even in comments). Implement the multiply yourself.

## Correctness comes first

The verifier compares your `D` against a high-precision reference on multiple
workloads — square and rectangular shapes, sizes that are not tile multiples,
`K = 1`, an identity operand, non-trivial `alpha`/`beta` (including `alpha = 0`),
all-zero inputs, a check that `C` is left unmodified, error-contract probes,
metamorphic checks (scaling `A` by a constant scales the product term; permuting
rows of `A` and `C` permutes rows of `D`), and a large-`K` case that stresses the
float32 accumulation — within a fixed rtol/atol. A faster result outside tolerance
on even one case scores zero.

## Scope

Optimize the kernel in `sgemm_kernel.cu` only. Do **not** edit `sgemm.py`, tests,
benchmark harnesses, workloads, or dependency/build files. The final submitted
diff must contain only changes to `sgemm_kernel.cu`.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can learn from: **submitting ends the
task.**

1. Read the current implementation in `/app/repo/sgemm_kernel.cu` and work out, before you touch
   anything, what will actually make it faster.
2. Make your changes to `/app/repo/sgemm_kernel.cu`. Test them yourself as thoroughly as you can
   with scratch scripts you write in your own scratch directory — that self-testing
   is your only safety net.
3. When — and only when — you are confident the implementation is both **correct**
   and **as good as you can make it**, submit it exactly once:

   ```
   bash /opt/loop/submit.sh
   ```

**Submitting is final and irreversible.**

- The moment `submit.sh` returns, this task is **over**. Stop editing and stop
  working on it.
- The harness finalizes automatically after the one submission. Calling it again
  does not score new edits.
- You will **not** receive iterative feedback to improve against. The receipt
  `submit.sh` prints is a record that your one submission was taken, not a signal
  to try again.

Because you only get one shot: think the design through first, re-read the
behavioural contract above, convince yourself every rule and error case is
handled, and satisfy yourself about performance — **then** submit.

Your score comes from a full, trusted end-of-session verifier (more workloads
than any self-test you can write), so correctness is not optional: a fast but
incorrect submission scores **zero**.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo` and your own knowledge and reasoning. `submit.sh`
is a one-shot submission channel, not a scoring oracle you may probe.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — the ONLY sanctioned scoring action is your single `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself. `submit.sh`
may be called exactly once, and only to submit your finished answer.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
