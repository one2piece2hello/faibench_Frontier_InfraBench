# Performance Optimization Task

You are given a clean Python workspace with a single editable file:

```
submission/kernel.py
```

This is the **only** file you may edit. Editing any other file (or importing an
implementation from anywhere else) causes the whole task to score 0.

## Objective

`submission/kernel.py` ships a required function `reduce_partials` whose body is **not
implemented** (it raises `NotImplementedError`). Implement it to the contract below so it is
numerically correct, then make it as fast as possible on the hidden benchmark workloads.
Reward is measured on GPU (H20). Correctness is a hard prerequisite: a fast-but-wrong
implementation, or one that still raises `NotImplementedError`, scores 0.

## Interface contract (implement exactly this)

### `reduce_partials(partials, bias) -> out`

The "all-reduce + bias" epilogue of a tensor-parallel **row-parallel linear**: a row-parallel
layer splits its input dimension across `R` ranks, each rank produces a partial output over the
same `[T, D]` shape, and the final result is the sum of those partials across ranks with the bias
added exactly **once**.

- `partials`: `bfloat16` CUDA tensor, shape `[R, T, D]` — `partials[r]` is rank `r`'s partial
  output. `R` is the tensor-parallel world size.
- `bias`: `bfloat16` CUDA tensor, shape `[D]` — added once to the reduced result (broadcast over
  the `T` rows).

Compute (accumulate in **fp32** for numerical stability, then cast back to `bfloat16`):

```
acc = sum over r in [0, R) of partials[r]      # reduce over the rank axis, fp32 [T, D]
out = acc + bias                                # bias added ONCE (broadcast over rows), fp32
return out.to(bfloat16)                         # [T, D]
```

- The bias is added a **single** time to the reduced sum — **not** once per rank.
- For `R == 1` the result is just `partials[0] + bias`.

Return `out`: a `bfloat16` tensor of shape `[T, D]`.

### `custom_kernel(data) -> out`

`data = (partials, bias, config)`, `config = {"R": int, "T": int, "D": int}`. Already wired to call
`reduce_partials` and return the reduced `[T, D]` output. You implement the primitive.

## Correctness

The output is compared against a seeded fp32 reference (sum the partials over the rank axis, add
the bias once) across a hidden set of shapes spanning a single row (`T=1`), odd `T`,
non-power-of-two `D`, the `D=1` edge, `R=1`, and large (`R=16, T=4096, D=4096`) regimes. The
`bfloat16` output must match within `rtol = atol = 2e-2`. Outputs must genuinely depend on the
inputs (the grader calls the primitive on different inputs of the same shape and rejects identical
outputs). A submission that adds the bias once **per rank** (so it is counted `R` times) fails.
Your implementation must be correct across the full domain before latency is measured.

## Latency

Once correct, the function is timed on GPU with CUDA events (block of medians, the input
regenerated every timed iteration; a warmup window is discarded). The benchmark drives many ranks
(`R` up to 16). A fused implementation that reduces the whole `[R, T, D]` stack over the rank axis
in a single bandwidth pass and writes the `[T, D]` result once is the reference point; a version
that accumulates one rank at a time in a Python loop is correct but issues `R` separate kernels and
re-reads/writes the full accumulator `R` times, moving several times the memory traffic (and the
gap grows with `R`).

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Workload shape (what the hidden benchmark covers)

- `[R, T, D]` `bfloat16` partial stacks with `R` up to 16, `T` up to a few thousand, and `D` up to
  4096, on a single GPU, plus a `[D]` bias.
- Single-row, odd, non-power-of-two, `D=1`, and `R=1` shapes as well as large ones.
- The score is dominated by the largest shapes.

## Rules

- Edit only `submission/kernel.py`, and keep the `reduce_partials` and `custom_kernel`
  signatures. Any other change (or importing the answer from another module/file/package)
  scores 0.
- Solve independently. Do **not** download, clone, fetch, or `pip install` any external
  repository, kernel library, or reference implementation to obtain the answer; do **not**
  attempt to bypass or disable the environment's network isolation (e.g. by unsetting proxy
  environment variables or opening raw sockets). Any such action scores the whole task 0,
  regardless of measured speed.
- You may use `torch` and `triton` (both are installed). You do not need any other package.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
