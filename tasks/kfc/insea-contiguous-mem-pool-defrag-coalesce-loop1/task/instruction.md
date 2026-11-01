# Performance Optimization Task

You are working on the memory pool that a training runtime uses to hand out many
variable-size buffers from a single, pre-reserved block of memory (an "arena"). Each
buffer must be backed by a **contiguous** run of cells inside the arena. As buffers
are allocated and released the arena fragments, so the pool also merges neighbouring
free space and, when needed, compacts live buffers toward the front to reopen a large
contiguous run. The file `mem_pool.py` implements this as a container, `MemoryPool`.
It is correct but slow.

## Behavioral contract

A `MemoryPool` is created with a positive integer `size` (the arena length in cells).
Buffers are requested with `allocate` and returned with `release`.

```python
class MemoryPool:
    def __init__(self, size: int): ...

    def allocate(self, size: int):
        """-> handle (an int) on success, or None if the request cannot be satisfied."""

    def release(self, handle) -> None:
        """Free the run backing `handle` and merge it with adjacent free space."""

    def offset_of(self, handle) -> int:
        """The current base offset of a live handle."""

    def largest_free(self) -> int:
        """Size of the largest single contiguous free run (0 if the arena is full)."""

    def total_free(self) -> int:
        """Total number of free cells."""

    def relocated_blocks(self) -> int:
        """Cumulative count of live runs physically moved by compaction."""
```

- **`MemoryPool(size)`**: `size` is a positive `int`. `TypeError` if it is not an
  `int` (bools rejected); `ValueError` if `size < 1`.
- **`allocate(size)`**: `size` is an `int` `>= 0` (`TypeError` if not an int / is a
  bool; `ValueError` if `size < 0`).
  - `size == 0` always succeeds and returns a handle that occupies no cells; its
    offset is `0`.
  - If `size` exceeds the current total free space, the allocation **fails** and
    returns `None` (a normal decision, not an error).
  - Otherwise the pool returns a handle backed by a contiguous run of `size` cells,
    placed at the **lowest start address** among the free runs large enough to hold
    it (first-fit, lowest address). If no single free run is large enough but the
    total free space is, the pool first **compacts** the live runs toward the front
    of the arena — relocating them while preserving their relative order — so the
    free space becomes one contiguous run, then places the allocation there.
- **`release(handle)`**: free the run backing `handle` and merge it with any
  immediately adjacent free run on either side, so two touching free runs never
  remain separate. `KeyError` if the handle is unknown or already released.
- **`offset_of(handle)`**: the current base offset of a live handle (`KeyError` if
  unknown / released). An offset can change after a compaction.
- **`largest_free()` / `total_free()`**: the largest single free run, and the total
  free cells (`size` minus the sum of live run sizes).
- **`relocated_blocks()`**: the cumulative number of live runs whose offset was
  physically changed by a compaction — `> 0` exactly when a compaction has moved
  something, and `0` while every allocation is satisfied without compaction.

Invariants that must hold after every operation: live runs are non-overlapping and
lie within `[0, size)`; `total_free() == size - sum(live run sizes)`; after any
`release` no two adjacent free runs remain un-merged; and `allocate(n)` returns
`None` iff `n > total_free()` (for `n > 0`).

Public API (do **not** change the names or the return shapes): `MemoryPool`,
`allocate`, `release`, `offset_of`, `largest_free`, `total_free`, `relocated_blocks`.

## Why the current implementation is slow

The current code keeps the free runs in a plain list. On **every** `release` it
appends the freed run and then re-sorts the whole list and re-scans it to merge
neighbours; on **every** `allocate` it re-sorts the free list again to find the
lowest-address fit; and it recomputes the free totals by scanning all free runs each
time. When the arena is fragmented into many small free runs (the common case after
a long mix of allocations and releases), those repeated full re-sorts and scans
dominate. Make the pool **faster** — do the same work with fewer element operations,
producing the **same placements and the same success/fail decisions** — for example
by keeping the free runs organized so a release merges its neighbours and an
allocation finds its fit without re-sorting the whole list every time, and by keeping
the free totals up to date incrementally.

**Forbidden:** delegating the allocation/bookkeeping to a real device or OS allocator,
or to an array library. The scoring harness scans your submitted file for those and
scores the task 0 (do not reference them even in comments). Build the free-run
bookkeeping, the neighbour merge, the first-fit search, and the compaction relocation
yourself.

## Correctness comes first

The verifier compares your pool's behaviour against an independent reference on many
workloads — interleaved allocate/release sequences, first-fit lowest-address
placement, allocating exactly the whole arena and a zero-size buffer, freeing every
buffer (which must coalesce to one free run of the whole arena), filling the arena
completely, releasing an unknown handle (a defined error) and requesting more than
the free space (a defined failure), a fragmentation pattern that only fits after
compaction, free-order independence, allocate-then-release round-tripping the exact
free layout, the non-overlap / free-total invariant after every op, and hidden and
work-evidence checks (the reported relocation count is `0` with no compaction and
positive when compaction runs). A faster result that is wrong on even one case scores
zero.

## Scope

Optimize the product implementation in `mem_pool.py` only. Do **not** edit tests,
benchmark harnesses, workloads, or dependency/build files. The final submitted diff
must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares the CPU instructions executed by your implementation with the baseline
instruction count, not elapsed time.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can learn from: **submitting ends the
task.**

1. Read the current implementation in `/app/repo/mem_pool.py` and work out, before you touch
   anything, what will actually make it faster.
2. Make your changes to `/app/repo/mem_pool.py`. Test them yourself as thoroughly as you can
   with scratch scripts you write in your own scratch directory — that self-testing
   is your only safety net.
3. When — and only when — you are confident the implementation is both **correct**
   and **as good as you can make it**, submit it exactly once:

   ```
   bash /opt/loop/submit.sh
   ```

**Submitting is final and irreversible.**

- The moment `submit.sh` returns, this task is **over**. Stop editing and stop
  working on it.
- The harness finalizes automatically after the one submission. Calling it again
  does not score new edits.
- You will **not** receive iterative feedback to improve against. The receipt
  `submit.sh` prints is a record that your one submission was taken, not a signal
  to try again.

Because you only get one shot: think the design through first, re-read the
behavioural contract above, convince yourself every rule and error case is
handled, and satisfy yourself about performance — **then** submit.

Your score comes from a full, trusted end-of-session verifier (more workloads
than any self-test you can write), so correctness is not optional: a fast but
incorrect submission scores **zero**.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo` and your own knowledge and reasoning. `submit.sh`
is a one-shot submission channel, not a scoring oracle you may probe.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — the ONLY sanctioned scoring action is your single `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself. `submit.sh`
may be called exactly once, and only to submit your finished answer.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
