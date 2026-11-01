# Performance Optimization Task

## What you are given
A clean workspace whose single editable file ships a required function `merge_shard_extents` that
is **not implemented** (it raises `NotImplementedError`). It is the metadata-merge step of a
**distributed checkpoint**: many ranks each write shard-metadata records for the logical tensors
they own, and to assemble the global checkpoint metadata you must compute, per tensor, the total
flattened size = the largest byte range any shard reaches. **Implement it to the contract below so
it is correct, then make it as fast as possible.** Correctness is a hard prerequisite: a
fast-but-wrong result, or a body still raising `NotImplementedError`, scores 0.

## Editable scope (out-of-scope edits fail the task)
Edit **only**:
```
submission/kernel.py
```

## Entry point and contract
`merge_shard_extents(entries, num_tensors) -> numpy.ndarray[int64]`:
- `entries`: a 2-D `numpy` int64 array of shape `(N, 3)`; each row is one shard record
  `[tensor_id, offset, size]` with `0 <= tensor_id < num_tensors`, `offset >= 0`, `size >= 1`.
  Rows are in **arbitrary order** (ranks interleave their records).
- Each shard's **end** offset is `offset + size`.
- `num_tensors` (int `G`): every tensor id in `[0, G)` appears in at least one row.
- For each tensor id `t` in `[0, G)`, its global extent is the **maximum** `offset + size` over all
  rows with `tensor_id == t`.
- Return a 1-D `numpy` `int64` array of length `G`: the global extent per tensor id, indexed by
  tensor id (`out[t]` = extent of tensor `t`).

`custom_kernel(data)` with `data = (entries, num_tensors)` is already wired to call
`merge_shard_extents` and return its result. Keep both public signatures.

## Required behavioral property (checked)
The verifier drives `custom_kernel` on hidden inputs across a range of tensor counts `G` and shard
counts `N`, and compares the returned per-tensor extent array for **exact equality** against an
independent reference implementing the contract. Outputs must genuinely depend on the input (a
constant/cached output is rejected). A submission that leaves the function `NotImplementedError`,
takes the max of the raw `offset` (forgetting to add `size`), or otherwise mis-reduces fails and
scores 0.

## Performance
The grader times your implementation on a large shard set with many tensors. A naive approach
that, for each tensor id, scans all `N` shard rows to find its maximum end offset does work that
grows as `O(G * N)`. Computing every shard's end once and scatter-reducing the maximum into a
per-tensor array in a single vectorized pass is `O(N)`. The gap grows with the number of tensors
`G`.

Let `s` be your speedup over the naive baseline and `r` the reference implementation's
speedup over that same baseline. A failed validity check, `r <= 1`, or `s <= r` gives
zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`: matching the
reference earns 0, and reaching `s = r²` earns 1.

## Rules
- Change only `submission/kernel.py`. Implement the complete contract; keep the public signatures.
- Pure Python / NumPy within the scope file; no new third-party dependencies.
- Deterministic given inputs; no reliance on wall-clock, randomness, or external state.
- Do not read, run, reproduce, or infer the scoring/verifier code, the hidden workloads, the
  threshold, the metric, or any reference solution.
- Do not download, clone, fetch, or `pip install` any external repository or reference
  implementation, and do not bypass the environment's network isolation. Any such action scores
  the whole task 0.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `bash /opt/loop/submit.sh`,也不需要 `git commit`(判分读工作树,HEAD 停在初始基线 commit;评分由结束后的 `tests/test.sh` 一次性给出)。
