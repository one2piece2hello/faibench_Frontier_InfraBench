# Performance Optimization Task

## Scope

You may modify **only** this file:

```
varlen_cu_seqlens.py
```

Everything else is **out of scope**. Any change to a file outside the scope above
causes the submission to score zero. Find where the slowness is *inside the scope*
by reading and profiling the code — that is part of the task.

## Objective

`varlen_cu_seqlens.py` builds the variable-length-attention metadata for a batch of
**packed documents**. Modern trainers pack several short documents end-to-end into
each fixed-length sequence row and mark the packing with a per-token `positions`
tensor of shape `[batch, seq_len]` whose values **reset to 0 at each document start**
and increase by 1 within a document. A variable-length (FlashAttention-style)
attention kernel does not want the `[batch, seq_len]` mask — it wants a compact
`cu_seqlens` index: the cumulative sequence-length boundaries of every document,
expressed over the whole batch flattened row-major (token `(b, col)` has global index
`b * seq_len + col`), plus `max_seqlen`, the longest document length.

The current implementation is **functionally correct but slow**. It:

1. **scans every token in Python** (`for b in range(batch): for i in range(seq_len)`)
   to find the per-row document starts;
2. **assembles the index lists element by element** in Python; and
3. **finds `max_seqlen` with a Python max loop**.

Its cost is O(`batch * seq_len`) in the interpreter and dominates for realistic
batches with many packed documents.

Your job: **make `build_varlen_cu_seqlens` faster on the benchmark workload, while
producing exactly the same `(cu_seqlens, max_seqlen)`.** You may reorganize the logic
within the scope file however you like (vectorize with numpy, change data
structures), as long as the observable output below is preserved.

## Behavioral contract (what the grader checks)

The grader calls the public entry point:

```python
build_varlen_cu_seqlens(positions) -> (cu_seqlens: list[int], max_seqlen: int)
```

- `positions`: a 2-D array-like `[batch, seq_len]` of non-negative ints; values reset
  to 0 at each document start (so `positions[b][0] == 0` for every row).
- returns `cu_seqlens`: a `list[int]` of the batch-flattened global indices
  (`b * seq_len + col`) of every document start, in row-major order, with the total
  token count `batch * seq_len` appended as the final entry.
- returns `max_seqlen`: the largest document length (the largest consecutive
  difference of `cu_seqlens`).

The returned `cu_seqlens` (full list) and `max_seqlen` must equal the independent
reference the grader computes from the same document layout. Any deviation scores
zero. The `build_varlen_cu_seqlens(positions)` signature must remain unchanged.

The grader times `build_varlen_cu_seqlens` on the benchmark workload.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Notes

- `numpy` is available.
- The benchmark workload uses a large batch with many short documents, so the
  per-token Python scan and per-boundary Python assembly dominate — the headroom
  grows with `batch * seq_len` and with the number of documents.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
