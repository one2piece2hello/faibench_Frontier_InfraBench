# Performance Optimization Task

You are given a clean Python workspace with a single editable file:

```
submission/kernel.py
```

This is the **only** file you may edit. Editing any other file (or importing an implementation
from anywhere else) causes the whole task to score 0.

## Objective

`submission/kernel.py` ships a required function `balance_chunks` whose body is **not implemented**
(it raises `NotImplementedError`). Implement it to the contract below so every partition is valid,
then make the **bottleneck (largest) chunk's byte total** it produces as **small** as possible on
the hidden workloads. This is a pure host-logic task (no GPU, no `torch`). Correctness is a hard
prerequisite: an invalid partition, or leaving `NotImplementedError` in place, scores 0.

## Background

In decentralized / data-parallel training (a DiLoCo outer step, a ring all-reduce, a
reduce-scatter), the flattened gradient is exchanged as a small number of **contiguous chunks** in
parameter order. A ring all-reduce runs in lock-step rounds, so its wall-time is bounded by the
**largest chunk** — the bottleneck link must move that many bytes each round. With at most
`num_chunks` chunks available (one per ring slot / communication buffer), you place the chunk
boundaries so the largest chunk is as small as possible, i.e. you balance the chunks by bytes. Real
gradient tensors are highly skewed — a few huge tensors (embeddings, the output projection) among
many small ones — so where you cut matters.

## Interface contract (implement exactly this)

### `balance_chunks(sizes, num_chunks) -> boundaries`

- `sizes`: `list[int]` of length `N` (`N >= 1`); `sizes[i] >= 1` is gradient tensor `i`'s byte
  count, given in **parameter order** (chunks must be contiguous in this order).
- `num_chunks`: int `P` (`>= 1`) — the maximum number of contiguous chunks.

Return `boundaries`: `list[int]`. The **exclusive end index** of each contiguous chunk, in strictly
increasing order. Chunk `k` spans `[boundaries[k-1], boundaries[k])` (chunk 0 starts at index 0), so
`boundaries[-1]` must equal `N`. **Validity (hard, checked first):** the list is non-empty, strictly
increasing, every value is in `1..N`, the last value is exactly `N` (all tensors covered), and
`len(boundaries) <= P`.

**Bottleneck chunk bytes (the quantity to minimize):** the `max` over chunks of each chunk's summed sizes —
minimize it.

### `custom_kernel(data) -> boundaries`

`data = (sizes, num_chunks, config)`, `config = {"N": int, "num_chunks": int}`. Already wired to
call `balance_chunks(sizes, num_chunks)` and return `boundaries`. You implement the partition.

## Correctness

Your partition is checked on a hidden set of streams spanning a single tensor, the `P == 1` edge
(one chunk over everything), the `P >= N` case (per-tensor chunks allowed), odd counts,
non-power-of-two counts, and large streams, against the validity rule above (strictly increasing
exclusive end indices, all in `1..N`, last equal to `N`, at most `P` chunks). The kernel is also run
on two different streams (of different lengths) and each partition must be valid (a fixed boundary
list that ignores the input fails). An invalid partition scores 0.

## Scoring (bottleneck chunk bytes — lower is better)

Once valid, the total bottleneck bytes over the hidden primary suite (the sum of each stream's
largest-chunk byte total) is measured. Let `s` be baseline bottleneck bytes divided by your
bottleneck bytes, and `r` the same ratio for the reference balancer. The final reward is 0 if any
hard check fails, `r <= 1`, or `s <= r`; otherwise it is `min(1, ln(s/r) / ln(r))`. You must use
fewer bottleneck bytes than the reference to score. Cutting into equal-count chunks is valid but a chunk
that happens to hold a huge tensor blows the bottleneck; placing boundaries so the byte totals are
balanced minimizes the largest chunk. The score is dominated by the largest streams.

## Workload shape (what the hidden benchmark covers)

- Gradient-size streams (`N` up to ~420) in parameter order: many small norm/bias tensors, medium
  linear weights, and a few huge embedding/output tensors — so balancing has real leverage.
- `P` a moderate chunk budget, well below `N`.
- Single-tensor, `P == 1`, `P >= N`, odd, non-power-of-two, and large cases.

## Rules

- Edit only `submission/kernel.py`, and keep the `balance_chunks` and `custom_kernel` signatures.
  Any other change (or importing the answer from another module/file/package) scores 0.
- Solve independently. Do **not** download, clone, fetch, or `pip install` any external repository
  or reference implementation to obtain the answer; do **not** attempt to bypass or disable the
  environment's network isolation (e.g. by unsetting proxy environment variables or opening raw
  sockets). Any such action scores the whole task 0, regardless of measured bytes.
- Pure Python standard library is sufficient; you do not need any third-party package.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
