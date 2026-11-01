# Performance Optimization Task

You are given a clean Python workspace with a single editable file:

```
submission/kernel.py
```

This is the **only** file you may edit. Editing any other file (or importing an
implementation from anywhere else) causes the whole task to score 0.

## Objective

`submission/kernel.py` ships a required function `plan_arena` whose body is **not implemented**
(it raises `NotImplementedError`). Implement it to the contract below so every plan it returns is
**valid**, then make the memory arena it produces as **small as possible** on the hidden
benchmark workloads. This is a pure host-logic task (no GPU, no `torch`). Correctness is a hard
prerequisite: a plan that places two simultaneously-live blocks on top of each other, or one that
still raises `NotImplementedError`, scores 0.

## Background

A runtime services a **time-ordered stream** of memory allocations and frees — think of a small
`malloc`, or an inference runtime's memory arena that packs tensor buffers as a graph executes.
Each block is allocated at some step and freed at a later step; while it is alive it needs its own
bytes, but once it is freed those bytes can be handed to a later allocation. An **online arena
allocator** assigns every block a byte offset into ONE contiguous arena, reusing the bytes of
freed blocks, so the arena high-water is far smaller than the sum of all block sizes.

## Interface contract (implement exactly this)

### `plan_arena(sizes, alloc_step, free_step) -> offsets`

- `sizes`: `list[int]` of length `N`. `sizes[b]` is the byte size of block `b` (`> 0`).
- `alloc_step`: `list[int]` of length `N`. `alloc_step[b]` is the step at which block `b` is
  allocated.
- `free_step`: `list[int]` of length `N`. `free_step[b]` is the step at which block `b` is freed,
  with `alloc_step[b] < free_step[b]`.

Block `b` is **live** across the half-open step interval `[alloc_step[b], free_step[b])` (it is
alive from its allocation step up to, but not including, its free step).

Two blocks `b` and `c` are **simultaneously live** iff their intervals overlap:

```
alloc_step[b] < free_step[c]  and  alloc_step[c] < free_step[b]
```

Return `offsets`: a `list[int]` of length `N` with `offsets[b] >= 0`. Block `b` occupies the byte
range `[offsets[b], offsets[b] + sizes[b])`.

**Validity (hard gate, checked before the arena is scored):** for every pair `(b, c)` of
simultaneously-live blocks, their byte ranges must be disjoint:

```
NOT (offsets[b] < offsets[c] + sizes[c]  and  offsets[c] < offsets[b] + sizes[b])
```

Blocks whose lifetimes do **not** overlap may reuse (share) the same bytes — that is exactly
where the savings come from.

**Arena size (the quantity to minimize):** `max_b (offsets[b] + sizes[b])`. Smaller is better.

### `custom_kernel(data) -> offsets`

`data = (sizes, alloc_step, free_step, config)`, `config = {"N": int}`. Already wired to call
`plan_arena(sizes, alloc_step, free_step)` and return the offsets list. You implement the
allocator.

## Correctness

Your plan is validated on a hidden set of workloads spanning single-block (`N=1`), tiny/odd,
non-power-of-two, medium and large (`N=400`) regimes, and workloads dominated by a few long-lived
blocks plus many short-lived ones. For each workload the grader checks that every block is placed
(`len(offsets) == N`, all `>= 0`) and that **no two simultaneously-live blocks occupy overlapping
bytes** (verified by sweeping the alloc/free timeline). The grader also runs your allocator on two
different workloads and requires each plan to be valid for its own workload (a fixed/cached plan
that ignores the input fails).

A valid plan is required before the arena is scored.

## Scoring (arena size — lower is better)

Once valid, the arena high-water your plan achieves on the hidden primary workload is measured:
`arena = max_b (offsets[b] + sizes[b])`. Let `s` be the naive arena size divided by yours, and
`r` the same ratio for the reference allocator. The final reward is 0 if any hard check fails,
`r <= 1`, or `s <= r`; otherwise it is `min(1, ln(s/r) / ln(r))`. You must use less arena space
than the reference to score. Large workloads dominate the measured arena size, so a tight plan
on a tiny workload cannot offset a wasteful one on a large workload.
The naive plan that gives every block its own private slab (arena = sum of all sizes) is correct
but leaves most of the reuse on the table.

## Workload shape (what the hidden benchmark covers)

- `N` up to a few hundred blocks; block sizes from ~256 KiB to ~8 MiB.
- A small number of blocks live almost the entire stream; most are live for a short window and
  can be overlaid on each other once freed.
- Single-block, tiny, odd and non-power-of-two workloads as well as large ones.

## Rules

- Edit only `submission/kernel.py`, and keep the `plan_arena` and `custom_kernel` signatures.
  Any other change (or importing the answer from another module/file/package) scores 0.
- Solve independently. Do **not** download, clone, fetch, or `pip install` any external
  repository or reference implementation to obtain the answer; do **not** attempt to bypass or
  disable the environment's network isolation (e.g. by unsetting proxy environment variables or
  opening raw sockets). Any such action scores the whole task 0, regardless of measured arena.
- Pure Python standard library is sufficient; you do not need any third-party package.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
