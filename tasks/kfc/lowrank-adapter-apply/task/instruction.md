# Performance Optimization Task

You are working on a linear layer with a small trainable low-rank correction —
the kind used to adapt a large frozen model cheaply. The file
`lowrank_adapter_apply.py` implements `lowrank_adapter_apply`: it applies a frozen
base linear weight plus a rank-`r` correction built from two small factors.

## Behavioral contract

Given an input `x` (last dimension `K`), a frozen base weight `base_weight`
`[N, K]`, two factors `factor_a` `[r, K]` and `factor_b` `[N, r]` (with the shared
rank `r` much smaller than `N` and `K`), and a python float `scale`, the result is

```
y = x @ base_weight.T + scale * ( (x @ factor_a.T) @ factor_b.T )
```

i.e. the frozen linear output plus a scaled low-rank correction. The op is linear
in `x` (there is no bias).

1. `x`: shape `(..., K)` (at least 2-D; leading dims are token/batch axes), dtype
   `torch.bfloat16` or `torch.float32`, CUDA tensor. `K` is the last (feature)
   dimension.
2. `base_weight`: shape `(N, K)`, same floating dtype and device as `x` — the
   frozen linear weight, laid out row-major `[out, in]` like a standard linear.
3. `factor_a`: shape `(r, K)`, same dtype/device as `x`.
4. `factor_b`: shape `(N, r)`, same dtype/device as `x`. The shared inner dimension
   is `r == factor_a.shape[0] == factor_b.shape[1]`.
5. `scale`: python `float` multiplying the low-rank correction.

Public signature (do NOT change):

```python
def lowrank_adapter_apply(
    x: torch.Tensor,           # (..., K) bf16/fp32 CUDA
    base_weight: torch.Tensor, # (N, K) same dtype/device as x (frozen)
    factor_a: torch.Tensor,    # (r, K) same dtype/device as x
    factor_b: torch.Tensor,    # (N, r) same dtype/device as x
    scale: float,
):  # -> y: (..., N), dtype of x
    ...
```

Error contract: any of `x` / `base_weight` / `factor_a` / `factor_b` not a floating
(bf16/fp32) tensor, or a dtype that differs from `x` → `TypeError`; shape
violations (`x` not at least 2-D; `base_weight` not `(N, K)` with `K` matching `x`;
`factor_a` not `(r, K)`; `factor_b` not `(N, r)` — in particular `factor_a`'s row
count must equal `factor_b`'s column count, the shared rank `r`) → `ValueError`.

## Why the current implementation is slow

The current implementation first builds the full `[N, K]` correction matrix
`delta = scale * (factor_b @ factor_a)`, adds it into the base weight, and then runs
one `[.., K] @ [K, N]` matmul against the combined weight. Forming, storing and
re-reading that full `[N, K]` matrix is exactly the work the low-rank structure lets
you avoid: for `r` much smaller than `N` and `K` the correction can be applied as
**two small matmuls** — the correction then costs on the order of `M · r · (K + N)`
multiply-adds instead of building and consuming an `[N, K]` matrix. Make it
**faster on the GPU** while keeping the numerics within the verifier's tolerance —
you may use any GPU technique available in the image as long as the contract above
holds.

**Forbidden:** materializing the full `[N, K]` weight delta (`factor_b @ factor_a`,
or the combined `base_weight + delta`) and then applying it. That is the slow path
the current code already takes; a faster solution must never form the `[N, K]`
intermediate. Apply the base and the low-rank correction without it.

## Correctness comes first

The verifier compares `y` against a high-precision float32 reference on multiple
workloads — 2-D and 3-D inputs, bf16 and fp32, `r = 1` and `r` near full rank, a
single token row, an out dimension of 1, a zero `scale` (which reduces to the base
linear only), a zero `factor_b` (correction vanishes), error-contract probes, and
metamorphic checks (the correction is linear in `scale`; the whole op is linear in
`x`) — within a per-dtype rtol/atol. A faster result outside tolerance on even one
case scores zero.

## Scope

Optimize the product implementation in `lowrank_adapter_apply.py` only. Do **not**
edit tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of `/app/repo/lowrank_adapter_apply.py`, decide on your
   approach, and make **all** the edits you want in `/app/repo`.
2. Self-test as much as you like with scratch scripts **you** write yourself
   (put them outside the scored file, e.g. under `/tmp`). Verify your output
   against the behavioral contract above on your own inputs, and measure your own
   before/after to convince yourself the change genuinely faster.
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
what limits the current code and what you intend to change. After you edit
and before you submit, give a short, concrete, step-by-step account of what you
changed, *why* it should be better in terms of the actual code path, and how you
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
