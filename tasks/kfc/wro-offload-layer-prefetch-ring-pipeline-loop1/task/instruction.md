# Layer-Wise Weight Prefetch over a Pinned Staging Ring — Performance Task

## Objective

`prefetch_ring.py` is the scheduler that lets a model whose weights do not fit in
device memory run anyway: while layer `l` computes, the weights of layers `l + 1`
and beyond are streamed in over a link into a small pool of **pinned staging
slots**. Planning that stream means answering, in this order:

1. how each layer's weight bytes split into fixed-size **transfer chunks**;
2. how far ahead of the layer being computed the prefetcher may run — the
   **lookahead window**, bounded both by a layer count and by the number of
   staging slots;
3. which **ring slot** each chunk lands in, which chunk it evicts, and how often
   each slot is reused;
4. when each chunk **arrives**, given that all transfers share one link with a
   fixed per-transfer overhead;
5. the per-layer **stall profile**: when a layer can start, when it ends, and how
   long compute sat idle waiting for weights;
6. the whole **pipeline report** — the bubble budget, and the slots that were
   overwritten before their previous occupant had been consumed.

The current implementation is **functionally correct but slow**: it is a literal
transcription of the specification, so every stage runs in the Python interpreter
one chunk at a time:

1. the **chunk table** is built by appending to Python lists inside a
   `while rem > 0` loop per layer — one interpreter step per chunk;
2. the **lookahead window** is a per-layer loop with scalar numpy indexing;
3. the **ring assignment** walks every chunk to compute `i % ring_slots`, its
   evicted predecessor and a per-slot counter;
4. the **link queue** is walked one transfer at a time, accumulating a running
   clock — one of the two dominant costs;
5. the **stall profile** re-scans every chunk of a layer to find the layer's last
   arrival, and then carries a running `end` forward — the other dominant cost;
6. the **driver** just calls all five in order.

Your job: **make the planner faster on the benchmark workload while producing
exactly the same output.** You may reorganize everything inside the scope file as
long as the observable contract below is preserved.

There is real structure to exploit, and finding it is the task. Two of the stages
*look* sequential and are not: the link queue is a running clock, but its cost per
chunk does not depend on the clock, and the stall profile is a running maximum of
`ready[k]` offset by the compute time between `k` and `l`. Both have closed forms
over the whole array. The chunk table looks like an unbounded `while` loop, but the
number of chunks per layer is known before the loop starts.

## Editable scope (only this file may change)

```
prefetch_ring.py
```

Everything else is **out of scope**. Any change to a file outside the scope above
causes the submission to score zero. Find where the slowness is *inside the scope*
by reading and profiling the code — that is part of the task.

## Behavioral contract (what the grader checks)

All six public entry points must keep their signatures and their behaviour:

```python
layer_chunk_table(layer_bytes, chunk_bytes)      -> (int64 (C,), int64 (C,), int64 (L+1,))
prefetch_window(layer_ptr, lookahead, ring_slots) -> int64 (L,)
ring_assign(chunk_size, ring_slots, slot_bytes)   -> (int64 (C,), int64 (C,), int64 (ring_slots,))
chunk_arrivals(chunk_size, bw_bytes_per_us, fixed_us, issue_us) -> (int64 (C,), int64 (C,))
stall_profile(layer_ptr, chunk_done, compute_us)  -> (int64 (L,), int64 (L,), int64 (L,))
run_pipeline(layer_bytes, chunk_bytes, ring_slots, slot_bytes, lookahead,
             bw_bytes_per_us, fixed_us, compute_us, issue_us=0) -> dict
```

The per-function docstrings in the scope file are normative. The load-bearing
clauses, restated because they are the ones that break:

1. **Everything is integer.** Every array is `int64`, every division is integer
   division, `ceil` is `(a + b - 1) // b` on non-negative integers, times are
   microseconds and sizes are bytes. Every comparison the grader makes is exact.
2. **Chunking is per layer, and the remainder is last.** Layer `l` contributes
   `ceil(layer_bytes[l] / chunk_bytes)` chunks, all of size `chunk_bytes` except
   the final one, which carries `layer_bytes[l] % chunk_bytes` when that is
   non-zero. Chunks are numbered globally in ascending `(layer, offset)` order,
   which is also the order they are issued on the link. **A zero-byte layer
   contributes no chunks** and `layer_ptr` still records its (empty) range.
3. **The window bound is `min(layer_ptr[min(l + 1 + lookahead, L)], layer_ptr[l] +
   ring_slots)`** — `lookahead` whole layers past the layer *after* the current
   one, and never more than `ring_slots` chunks past the current layer's first
   chunk. It is an **exclusive** bound.
4. **The ring is plain round-robin:** chunk `i` occupies slot `i % ring_slots` and
   evicts chunk `i - ring_slots`, or `-1` on the first pass. `reuse[s]` counts the
   chunks assigned to slot `s`, and its length is `ring_slots` even when no chunk
   exists.
5. **The link is serial and its per-chunk cost is fixed:**
   `cost[i] = ceil(chunk_size[i] / bw_bytes_per_us) + fixed_us`, the first transfer
   starts at `issue_us`, and `start[i] = done[i-1]`. The overhead is added *after*
   the ceiling division, not before.
6. **A layer's readiness is the maximum arrival over its own chunks**, and `0` when
   it owns none. Compute is serialised on one device, so
   `start[l] = max(ready[l], end[l-1])` with `end[-1] = 0`,
   `end[l] = start[l] + compute_us[l]` and `stall[l] = start[l] - end[l-1]`.
   Dropping the `end[l-1]` term is the classic mistake — it is what a naive
   vectorisation produces, and it is checked.
7. **A conflict is an early eviction:** chunk `i` conflicts when
   `ev[i] >= 0 and start[i] < end[chunk_layer[ev[i]]]` — its transfer began before
   the chunk it overwrites had been consumed, and a chunk is consumed when the
   layer owning it finishes computing. `conflict_chunks` lists those indices in
   ascending order.
8. **Shapes, dtypes and keys are fixed.** `run_pipeline` returns exactly the python
   ints `n_chunks`, `n_layers`, `bytes_total`, `total_us` (the last layer's `end`,
   or `0` with no layers), `stall_us`, `bubbles` (how many layers stalled at all),
   `conflicts`, `link_busy_us` (the sum of the per-chunk costs) and
   `last_arrival_us`, plus the arrays `window`, `layer_stall`, `layer_start`,
   `slot_reuse` and `conflict_chunks`.
9. **No input may be modified**, and every documented input-validation
   `ValueError` must still fire: a non-1-D array, a negative byte count or compute
   time, a non-positive `chunk_bytes` / `ring_slots` / `slot_bytes` /
   `bw_bytes_per_us`, a negative `lookahead` / `fixed_us` / `issue_us`, a chunk
   larger than a staging slot, a `layer_ptr` that is empty, does not start at zero,
   is not non-decreasing, or does not index `chunk_done` exactly, and a
   `compute_us` whose length is not `len(layer_ptr) - 1`.

The grader compares every returned value against an independent reference over
curated cases that between them exercise a plain model, layers with zero bytes, an
all-zero model, free compute, compute-bound compute, one huge layer, a chunk size
larger than any layer, a tiny chunk size, a one-slot ring, a ring larger than the
model, `lookahead` of 0 and 99, a one-byte-per-microsecond link, an effectively
infinite link, zero and large per-transfer overhead, a late issue time, slots with
slack, a single layer, no layers at all, and two deep models. Any deviation scores
zero.

## Correctness & how you are scored

- Correctness is a hard gate: any mismatch against the reference, any mutated
  input, any missing error contract, or any edit outside the scope file scores
  **0**.
- Let `s` be the whole planner's speedup over the frozen slow baseline and `r` the reference
  planner's speedup over that baseline. The final reward is 0 if any hard check fails, `r <= 1`,
  or `s <= r`; otherwise it is `min(1, ln(s/r) / ln(r))`. Matching the reference earns 0.
- The benchmark workload plans **768 layers chunked at 16 KiB (about 150 thousand
  chunks) over a 192-slot ring**, so every per-chunk stage matters and the
  per-layer stages do not.
- Timing is CPU time (`time.process_time`), min of 3 runs, taken after a fixed CPU
  warm-up, so the score is robust to machine state.

## Notes

- `numpy` is available. There is no GPU and no torch in this task.
- Because every quantity is an integer, there is no tolerance to hide behind: a
  reassociated sum is either the same integer or a failure.
- Nothing forbids computing an intermediate once and sharing it between the driver
  and its callees, as long as every function is still correct when called on its
  own and no input array is written to.

## Solve independently — prohibited actions (any one => the whole task scores 0)

- Do **not** read, modify, or reverse-engineer the hidden verifier / grader /
  timing harness, and do not fabricate a reward.
- Do **not** fetch an upstream reference implementation (`accelerate`,
  DeepSpeed ZeRO-Inference, vLLM weight streaming or any other offload stack) for
  the scoped planner, in any form.
- Do **not** edit anything outside the declared scope `prefetch_ring.py`, and do
  not bypass the scope or network isolation.
- Only legitimate optimization of the in-scope code counts.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
