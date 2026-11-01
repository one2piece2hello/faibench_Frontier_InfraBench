# Performance Optimization Task

You are working on a data-layout kernel used throughout a high-throughput
inference server — it produces the **transpose** of a 2-D matrix (rearranging a
row-major matrix so its rows become columns). The file `transpose_kernel.cu`
implements this on the GPU. The Python wrapper `transpose.py` compiles this
kernel and calls it; **you edit only `transpose_kernel.cu`.**

## Behavioral contract

Given `x` of shape `(M, N)`, produce `y` of shape `(N, M)` such that

```
y[j, i] == x[i, j]        for all 0 <= i < M, 0 <= j < N
```

1. `x`: a 2-D, contiguous, `torch.float32` **or** `torch.float16` CUDA tensor of
   shape `(M, N)`.
2. `y`: shape `(N, M)`, **row-major contiguous**, same dtype as `x`. Elements are
   copied unchanged — this is pure data movement, there is no arithmetic.
3. The result must be correct for arbitrary `M`, `N` — including **non-square**
   shapes, sizes that are **not** a multiple of any block width, and a single row
   (`M = 1`) or single column (`N = 1`).

Public entry point (invoked by the fixed wrapper; do NOT change its meaning):

```python
def transpose(x: torch.Tensor) -> torch.Tensor:  # y[N, M] with y[j, i] == x[i, j]
    ...
```

Error contract (enforced by the wrapper): a non-tensor or non-`float32`/`float16`
`x` → `TypeError`; a non-2-D or non-CUDA `x` → `ValueError`.

## Why the current implementation is slow

The current kernel assigns one GPU thread to each output element: the thread
reads the corresponding source element and writes it to its transposed position.
It is **correct**, but the pattern in which those reads and writes land in global
memory drives the data across the memory bus far below the rate the hardware can
sustain. This operation is **memory-bound** — the run time is dominated entirely
by how efficiently it moves bytes through global memory, not by any computation
(there is none). Make it **faster on the GPU** while producing exactly the same
output. You may use any GPU technique available in the image, as long as the
contract above holds and the result is bit-for-bit identical.

**Forbidden:** do not delegate the reordering to a built-in framework or library
primitive that already produces the transposed layout for you — specifically the
libtorch tensor operations (`at::transpose`, `torch::transpose`, `at::permute`,
`torch::permute`, and the `.t()` / `.transpose(...)` / `.permute(...)` / `.mT`
tensor methods), a `.contiguous()` applied to any of those, and vendor libraries
(**cuBLAS / cutlass**). The scoring harness scans your submitted
`transpose_kernel.cu` for those tokens and scores the task 0 if present (do not
reference them even in comments). Implement the reordering yourself.

## Correctness comes first

The verifier compares your `y` **bit-for-bit** against a reference transpose on
multiple workloads — square and non-square shapes, sizes that are not a block
multiple, a single row and a single column, a 1×1 matrix, float16 as well as
float32, a metamorphic check (transposing twice returns the original), and
error-contract probes. Because this is pure data movement, the match must be
**exact** — a single mismatched element on any one case scores zero.

## Scope

Optimize the kernel in `transpose_kernel.cu` only. Do **not** edit
`transpose.py`, tests, benchmark harnesses, workloads, or dependency/build files.
The final submitted diff must contain only changes to `transpose_kernel.cu`.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of ``/app/repo/transpose_kernel.cu``, decide on your
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
