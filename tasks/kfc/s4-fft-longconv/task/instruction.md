# Implementation Task: Causal Long Convolution

You are given a clean Python workspace with a single editable file:

```
submission/kernel.py
```

This is the **only** file you may edit. Editing any other file (or importing an
implementation from anywhere else) causes the whole task to score 0.

## Objective

`submission/kernel.py` ships a required function `causal_conv` whose body is **not implemented**
(it raises `NotImplementedError`). Implement it to the contract below so it is numerically
correct on the hidden GPU (H20) workloads. Reward is binary: 1 if every correctness case and
validity check passes, 0 otherwise. A faster implementation earns no additional credit.

## Interface contract (implement exactly this)

### `causal_conv(u, k) -> y`

A **causal** (non-circular) 1-D convolution of each input sequence with its channel's full-length
kernel — the "convolution mode" of a long-kernel sequence layer.

- `u`: `bfloat16` tensor, shape `[B, H, L]` (`B` sequences, `H` channels, length `L`).
- `k`: `bfloat16` tensor, shape `[H, L]` (a per-channel causal kernel, full length `L`).

For every `(b, h)` and output position `t` in `[0, L)`:

```
y[b, h, t] = sum_{s=0}^{t} k[h, t - s] * u[b, h, s]
```

This is the linear convolution of `u[b,h]` with `k[h]`, **truncated to the first `L` samples**. The
system is causal: output `t` depends only on inputs `s <= t`. There is **no wrap-around** — the tail
of the sequence must not fold back into early outputs (that would be a circular convolution, which
is wrong). Accumulate in **fp32**, then cast the result back to `bf16`.

### `custom_kernel(data) -> y`

`data = (u, k, config)` where `config = {"L": int}`. Already wired to call `causal_conv(u, k)` and
return `y`. You implement the primitive.

## Correctness

Outputs are compared against a seeded fp32 reference within `rtol = atol = 2e-2` across a hidden
set of shapes spanning `L=1` (single tap), tiny `L`, odd / non-power-of-two `L`, odd channel
counts, and a large regime (`L` up to ~2048). The causal boundary must be exact — a submission
that reads future samples (e.g. a circular convolution with no zero-padding) fails the gate.
Outputs must genuinely depend on the inputs (the grader runs the op on two different inputs of the
same shape and rejects identical outputs).

## Latency

Once correct, the function is timed on GPU with CUDA events (block of medians, the `u` input
regenerated every timed iteration; a warmup window is discarded). The primary benchmark uses a
**long** sequence length `L`. Evaluating the convolution directly — an `O(L^2)` sum per output
position, or equivalently a dense lower-triangular Toeplitz matrix apply — is correct but its cost
grows with `L^2`. A full-length convolution can instead be evaluated in `O(L log L)`.
Timing is diagnostic; it does not enter the binary reward.

## Workload shape (what the hidden benchmark covers)

- `bfloat16` `u [B, H, L]` and `k [H, L]`, with `L` up to ~2048, modest `B` and `H`, single GPU.
- `L=1`, tiny, odd, and non-power-of-two shapes as well as large ones.

## Rules

- Edit only `submission/kernel.py`, and keep the `causal_conv` and `custom_kernel` signatures. Any
  other change (or importing the answer from another module/file/package) scores 0.
- Solve independently. Do **not** download, clone, fetch, or `pip install` any external
  repository, kernel library, or reference implementation to obtain the answer; do **not** attempt
  to bypass or disable the environment's network isolation (e.g. by unsetting proxy environment
  variables or opening raw sockets). Any such action scores the whole task 0, regardless of
  measured speed.
- You may use `torch` and `triton` (both are installed). You do not need any other package.

## 提交方式

本题为**实现类,只有一次评分**。把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `bash /opt/loop/submit.sh`,也不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上;评分由结束后的 `tests/test.sh` 一次性给出)。
