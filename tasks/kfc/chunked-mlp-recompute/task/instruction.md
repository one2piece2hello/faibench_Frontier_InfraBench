# Implementation Task: Gated MLP Forward and Backward

You are given a clean Python workspace with a single editable file:

```
submission/kernel.py
```

This is the **only** file you may edit. Editing any other file (or importing an
implementation from anywhere else) causes the whole task to score 0.

## Objective

`submission/kernel.py` ships a required function `gated_mlp_fwd_bwd` whose body is
**not implemented** (it raises `NotImplementedError`). Implement it to the contract
below so it is numerically correct, then make it use as **little peak GPU memory as
possible** on the hidden benchmark workloads (GPU, H20). Correctness is a hard
prerequisite: an implementation that is memory-light but wrong, or that still raises
`NotImplementedError`, scores 0.

## Background

This is one training step of a gated feed-forward (SwiGLU-style) MLP block: a forward
pass that produces an output `y`, and a manual backward pass that, given the gradient
of the loss with respect to `y`, produces the gradient with respect to the block input
`x`. The forward pass creates several **large intermediate activations** of shape
`[T, I]`; a straightforward implementation that keeps all of them resident in order to
run the backward pass has a high memory high-water mark. Chunking or recomputation can
reduce that footprint without changing the outputs.

## Interface contract (implement exactly this)

Let `silu(z) = z * sigmoid(z)`.

### `gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, chunk_size) -> (y, dx)`

Tensors (all `bfloat16`, on a single CUDA device):

- `x`:        `[T, H]`  — block input
- `w_gate`:   `[H, I]`
- `w_up`:     `[H, I]`
- `w_down`:   `[I, H]`
- `grad_out`: `[T, H]`  — the upstream gradient `dL/dy`
- `chunk_size`: Python `int` — a **suggested row-block granularity** for processing the
  `T` rows in groups. It affects only how the work is scheduled; the returned values
  **must be identical regardless of `chunk_size`** (including `chunk_size >= T`, i.e. a
  single block, and `chunk_size` that does not evenly divide `T`).

**Forward** (rows are independent — row `t` of every tensor below depends only on row `t`
of `x`):

```
g = x @ w_gate                 # [T, I]
u = x @ w_up                   # [T, I]
a = silu(g) * u                # [T, I]
y = a @ w_down                 # [T, H]
```

**Backward** — given `grad_out = dL/dy`, return `dx = dL/dx`:

```
da    = grad_out @ w_down^T                      # [T, I]
du    = da * silu(g)                             # [T, I]
dsilu = da * u                                   # [T, I]
silup = sigmoid(g) * (1 + g * (1 - sigmoid(g)))  # = silu'(g), the activation derivative, [T, I]
dg    = dsilu * silup                            # [T, I]
dx    = dg @ w_gate^T + du @ w_up^T              # [T, H]
```

Note `silup` is the **derivative** of `silu` (not `silu` itself); using `silu(g)` in its
place gives a wrong `dx`. Return `(y, dx)`, both `bfloat16 [T, H]`.

### `custom_kernel(data) -> (y, dx)`

`data = (x, w_gate, w_up, w_down, grad_out, config)` where
`config = {"T": int, "H": int, "I": int, "chunk_size": int}`. It is already wired to call
`gated_mlp_fwd_bwd(x, w_gate, w_up, w_down, grad_out, config["chunk_size"])` and return the
`(y, dx)` pair. You implement `gated_mlp_fwd_bwd`.

## Correctness

Both `y` and `dx` are compared against a seeded `float32` mathematical reference (the contract
math above, computed independently) within `rtol = atol = 2e-2`, across a hidden set of shapes
spanning small (`T=1`), non-power-of-two, `chunk_size` values that do and do not divide `T`,
single-chunk (`chunk_size >= T`), and large (`T=8192, H=4096, I=14336`) regimes. Your
implementation must be correct across the whole domain — including the full backward, in
particular the `silu` derivative term — before memory is measured. Outputs must genuinely depend
on the inputs (the grader runs the op on two different inputs of the same shape and rejects
identical outputs).

## Scoring

Reward is binary: 1 if all correctness cases and validity checks pass, 0 otherwise.
After correctness passes, the grader measures peak GPU memory with
`torch.cuda.max_memory_allocated()`. The measurement must be valid; reducing the peak
does not award partial or extra credit.

## Workload shape (what the hidden benchmark covers)

- Token-major `bfloat16` activations at transformer-MLP scale: `T` up to ~8192, `H` up to 4096,
  `I` up to ~14336, on a single GPU; the large `[T, I]` intermediates dominate the footprint.
- Both small/boundary shapes (size-1 rows, non-tile-multiple dims, `chunk_size` not dividing `T`)
  and large shapes. Every correctness case must pass.

## Rules

- Edit only `submission/kernel.py`, and keep the `gated_mlp_fwd_bwd` and `custom_kernel`
  signatures. Any other change (or importing the answer from another module/file/package)
  scores 0.
- Solve independently. Do **not** download, clone, fetch, or `pip install` any external
  repository, kernel library, or reference implementation to obtain the answer; do **not**
  attempt to bypass or disable the environment's network isolation (e.g. by unsetting proxy
  environment variables or opening raw sockets). Any such action scores the whole task 0,
  regardless of measured memory.
- You may use `torch` and `triton` (both are installed). You do not need any other package.

## 提交方式

本题为**实现类,只有一次评分**。把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `bash /opt/loop/submit.sh`,也不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上;评分由结束后的 `tests/test.sh` 一次性给出)。
