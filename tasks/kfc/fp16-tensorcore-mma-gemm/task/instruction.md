# Performance Optimization Task

You are working on the dense matrix-multiply kernel at the heart of a
high-throughput inference server's linear layers. The file `gemm_kernel.cu`
implements a half-precision matrix multiply — it computes the product of two
matrices on the GPU. The Python wrapper `gemm.py` compiles this kernel and calls
it; **you edit only `gemm_kernel.cu`.**

## Behavioral contract

Given `A` of shape `(M, K)` and `B` of shape `(K, N)`, compute the matrix product

```
C[m, n] = sum over k of A[m, k] * B[k, n]        # C has shape (M, N)
```

1. `A`, `B`: 2-D, contiguous, `torch.float16` CUDA tensors. The inner dimension
   matches (`A.shape[1] == B.shape[0] == K`).
2. Accumulation is performed in **float32** for numerical stability; the returned
   `C` is `(M, N)`, `torch.float16`.
3. The result must be correct for arbitrary `M`, `N`, `K` — including sizes that
   are **not** multiples of any tile width, `K = 1`, and large `K`. Row order is
   preserved.

Public entry point (invoked by the fixed wrapper; do NOT change its meaning):

```python
def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:  # C = A @ B, fp16, fp32 accumulate
    ...
```

Error contract (enforced by the wrapper): non-`float16` or dtype-mismatched
`A` / `B` → `TypeError`; non-2-D inputs or a mismatched inner dimension →
`ValueError`.

## Why the current implementation is slow

The current kernel is a straightforward tiled multiply on the GPU's general
arithmetic path: it stages tiles of `A` and `B` through shared memory and
accumulates each output element with ordinary fused multiply-adds. It is
**correct** but uses only a fraction of the device's available half-precision
throughput — the multiply is **compute-bound**, and the general FMA path is far
from the peak the hardware can sustain for this data type. Make it **faster on
the GPU** while keeping the numerics within the verifier's tolerance. You may use
any GPU technique available in the image (better tiling, register/shared-memory
reuse, the device's specialized half-precision matrix hardware, etc.) as long as
the contract above holds and accumulation stays in float32.

**Forbidden:** do not delegate the multiply to a prebuilt matrix-multiply
library or framework primitive — specifically **cuBLAS / cublasLt / cutlass** and
libtorch's matmul (`at::matmul`, `torch::matmul`, `at::mm`, `torch::mm`,
`.matmul(...)`), as well as `torch.matmul` / `torch.mm` / `F.linear` / the `@`
operator on the Python side. The scoring harness scans your submitted
`gemm_kernel.cu` for those tokens and scores the task 0 if present (do not
reference them even in comments). Implement the multiply yourself.

## Correctness comes first

The verifier compares your `C` against a high-precision float32 reference on
multiple workloads — square and rectangular shapes, sizes that are not tile
multiples, `K = 1`, an identity operand, error-contract probes, metamorphic
checks (scaling `A` by a constant scales `C`; permuting rows of `A` permutes rows
of `C`), and a large-`K` case that stresses the float32 accumulation — within a
fixed rtol/atol. A faster result outside tolerance on even one case scores zero.

## Scope

Optimize the kernel in `gemm_kernel.cu` only. Do **not** edit `gemm.py`, tests,
benchmark harnesses, workloads, or dependency/build files. The final submitted
diff must contain only changes to `gemm_kernel.cu`.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of ``/app/repo/gemm_kernel.cu``, decide on your
   approach, and make **all** the edits you want in `/app/repo`.
2. Self-test as much as you like with scratch scripts **you** write yourself
   (put them outside the scored file, e.g. under `/tmp`). Verify your output
   against the behavioral contract above on your own inputs, and time your own
   before/after to convince yourself the change is genuinely faster.
3. When — and only when — you are confident, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

**One submission only.** Once the submission is recorded, the harness finalizes it.
A later call does not score new edits. Do your own checks before submitting.

Because of that, everything rides on the work you do **before** you submit:
think the design through, read the baseline carefully, self-test the contract
(including the boundary and error cases described above), and only then submit.
Do not submit a half-finished or untested edit hoping to refine it later — there
is no later.

The grade is produced by a full, trusted end-of-session verifier (more workloads
than anything you can see), so your one submission must be genuinely correct as
well as fast.

**🎓 Explain your work for a beginner.** Before you edit, state your approach:
what makes the current code slow and what you intend to change. After you edit
and before you submit, give a short, concrete, step-by-step account of what you
changed, *why* it should be faster in terms of the actual code path, and how you
convinced yourself it is still correct.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo`, the sanitized feedback that `submit.sh` returns,
and your own knowledge and reasoning.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — score ONLY by calling `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself; use only
what `submit.sh` returns for scoring signal.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
