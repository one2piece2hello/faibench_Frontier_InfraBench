# Performance Optimization Task

You are given a clean Python workspace with a single editable file:

```
submission/kernel.py
```

This is the **only** file you may edit. Editing any other file (or importing an
implementation from anywhere else) causes the whole task to score 0.

## Objective

`submission/kernel.py` ships with a required routine whose body is **not
implemented** (it raises `NotImplementedError`). Your job is to **implement it to
the contract below so it is numerically correct, then make it as fast as
possible** on the hidden benchmark workloads. Reward is measured on GPU (H20).
Correctness is a hard prerequisite: an implementation that is fast but wrong, or
that still raises `NotImplementedError`, scores 0.

The workload models the **all-to-all redistribution** step used in expert- and
sequence-parallel training. A buffer is partitioned into blocks indexed by the
`(source rank, destination rank)` pair; an all-to-all delivers to every rank the
blocks that were addressed to it, which — laid out as one tensor — is a transpose
of the two rank axes. There is no real cross-device communication (it runs on a
single GPU); the challenge is doing the block reorganization while amortizing
per-block overhead across the whole grid.

## Interface contract (implement exactly this)

The verifier imports `custom_kernel(data)` from `submission/kernel.py` and the
routine it is built from. Every symbol, signature, shape, and dtype below is part
of the contract.

### `all_to_all_redistribute(x, world_size) -> Tensor`

- `x`: a `bfloat16` **CUDA** tensor of shape `[world_size, world_size, chunk, D]`.
  `x[s, d]` is the `[chunk, D]` block that source rank `s` sends to destination
  rank `d`.
- `world_size`: Python `int` `W` (equal to `x.shape[0]` and `x.shape[1]`).

Compute the redistributed buffer `y`, a **contiguous** `bfloat16` tensor of shape
`[world_size, world_size, chunk, D]` with

```
y[d, s] = x[s, d]        for all s, d in [0, world_size)
```

i.e. swap the two leading (source ↔ destination) rank axes and materialize the
result **contiguously**. Each `[chunk, D]` block's contents are copied unchanged;
only its `(source, destination)` position moves. The output dtype (`bfloat16`) and
shape match the input.

### `custom_kernel(data) -> Tensor`

`data = (x, config)` where `config = {"world_size": int, "chunk": int, "D": int}`.
This wrapper is already written; it calls `all_to_all_redistribute(x, config["world_size"])`.
You implement `all_to_all_redistribute`.

## Correctness

Outputs are compared against a seeded fp32 mathematical reference within
`rtol = atol = 2e-2`. The reference is exactly `y[d, s] = x[s, d]` (the rank-axis
swap), materialized contiguously; the grader observes only the returned tensor.
The hidden suite spans:

- a single rank (`W = 1`) and size-1 payloads,
- `chunk != world_size` (a wrong transpose axis is caught by the output **shape**),
- `chunk == world_size` (a wrong transpose axis is caught by the **values**),
- non-power-of-two `chunk` and `D`,
- large grids (`W` up to ~64, i.e. thousands of blocks).

Every returned tensor's **shape and value** are checked. Outputs must genuinely
depend on the inputs (the grader calls the routine on two different inputs of the
same shape and rejects bit-identical outputs).

## Latency

Once correct, the routine is timed on GPU with CUDA events (block of medians, `x`
regenerated with a fresh seed every timed iteration; a warmup window is
discarded). The benchmark drives **large grids** (`W` up to ~64 ⇒ `W*W` blocks),
so the score is dominated by how the `W*W` blocks are moved. Copying the blocks
one `(source, destination)` pair at a time is correct but pays a fixed cost for
**every** block (a cost that grows as `W*W`); driving that per-block cost down —
so the number of launched operations does not scale with the grid — is exactly the
point.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Workload shape (what the hidden benchmark covers)

- `bfloat16` CUDA tensors `x [W, W, chunk, D]`, `W` up to ~64, modest `chunk` and
  `D`, single GPU.
- Both tiny/boundary grids and large ones (including `chunk != W`, `chunk == W`,
  and non-power-of-two `chunk`/`D`).
- The primary grid supplies the timing measurement. All correctness cases still have to pass.

## Rules

- Edit only `submission/kernel.py`. Any other change (or importing the answer
  from another module/file/package) scores 0.
- Solve independently. Do **not** download, clone, fetch, or `pip install` any
  external repository, kernel library, or reference implementation to obtain the
  answer; do **not** attempt to bypass or disable the environment's network
  isolation (e.g. by unsetting proxy environment variables or opening raw
  sockets). Any such action scores the whole task 0, regardless of measured
  speed.
- You may use `torch` (and `triton` if you wish); both are installed. You do not
  need any other package.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
