# Performance Optimization Task

You are working on the caching allocator inside a training runtime. Device memory
allocation and release are expensive, so the allocator keeps freed buffers in a
pool and hands one back on the next request of the same size instead of allocating
a fresh one. The file `caching_allocator.py` implements this pool as a class,
`CachingAllocator`. It is correct but slow.

## Behavioral contract

A `CachingAllocator` is created with a positive integer `capacity` (the hard byte
capacity of a simulated device). Buffers are handed out by `alloc` and returned by
`free`. The "device" is a stub whose allocate/free operations only bump counters
and track a running byte total — that counter is the cost being modelled.

```python
class CachingAllocator:
    def __init__(self, capacity: int = 1_000_000_000): ...

    def alloc(self, size: int):
        """-> an opaque integer handle for a buffer of `size` cells."""

    def free(self, handle, cacheable: bool = True) -> None:
        """Return a live buffer to the pool (or release it to the device)."""
```

- **capacity**: a positive `int` — the total bytes the device can hold *resident*
  at once (live buffers **plus** pooled/cached buffers). `TypeError` if not an
  `int` (bools rejected); `ValueError` if `< 1`.
- **alloc(size) -> handle**: `size` is a positive `int` (`TypeError` if not an int
  / bools rejected; `ValueError` if `< 1`).
  - If a cached (previously freed) buffer of **exactly** `size` is available, reuse
    it — no new device allocation is performed.
  - Otherwise a new device buffer is required. If creating it would push the
    resident bytes over `capacity`, first **evict the entire cache** (return every
    pooled buffer to the device, reclaiming its bytes) and retry once; if it still
    does not fit, raise `MemoryError`. Otherwise create it.
  - Returns an opaque integer `handle`.
- **free(handle, cacheable=True)**: `handle` must be a currently-live buffer
  (`KeyError` otherwise — this covers freeing an unknown handle and double-free).
  When `cacheable` is true the buffer is returned to the pool for its size (kept
  resident, reusable by a later same-size `alloc`); when false it is released to
  the device immediately (its bytes are reclaimed and it is **not** pooled).

Invariants: reuse is **size-exact** (a request never reuses a buffer of a different
size); a freed-then-alloc'd same-size request performs **zero** new device
allocations; the bytes resident on the device (live + cached) never exceed
`capacity`; a whole-cache eviction happens **only** when a new allocation would
otherwise exceed `capacity`.

### Observable state (the verifier reads these — keep the names and meanings)

- `decisions`: a `list`, one entry per `alloc`, each the string `"reuse"` (served
  from the pool) or `"new"` (a new device buffer was created).
- `device_alloc_count`: number of new device buffers created (the expensive op).
- `device_free_count`: number of buffers released back to the device.
- `eviction_count`: number of whole-cache evictions performed.
- `reuse_count`: number of allocs served from the pool.
- `live_sizes()`: sorted list of the sizes of currently-live buffers.
- `cached_sizes()`: sorted list of the sizes of currently-pooled buffers.

Public API (do **not** change the names or the return shapes): `CachingAllocator`,
`__init__(capacity)`, `alloc(size) -> handle`, `free(handle, cacheable=True)`, and
the observable attributes/methods listed above.

## Why the current implementation is slow

Every freed buffer is dropped into a **single flat list**. To serve `alloc(size)`
the current code **linear-scans that whole list** looking for a buffer whose size
equals the request. When many buffers of many different sizes are pooled at once,
each alloc walks past a long run of wrong-size buffers before it finds a match (or
gives up and allocates a new one), so the lookup cost grows with the number of
pooled buffers. Make the lookup **faster** — do fewer per-buffer comparisons for
the same decisions — for example by organizing the pooled buffers so a request of a
given size goes straight to the buffers of that size instead of scanning every
pooled buffer.

**Forbidden:** delegating the pooling or the reuse decision to a real memory
allocator, an array/tensor library, or an external device/runtime framework. The
scoring harness scans your submitted file for those and scores the task 0 (do not
reference them even in comments). Build the pool, the size-exact reuse decision,
and the evict-and-retry control flow yourself.

## Correctness comes first

The verifier compares your allocator's **observable output** (the `decisions`
sequence, the `device_alloc_count` / `device_free_count` / `eviction_count` /
`reuse_count` counters, and the final `live_sizes()` / `cached_sizes()` multisets)
against an independent reference on many workloads — same-size reuse, distinct
sizes that never reuse, a first alloc on an empty cache, a single-size churn, all
distinct sizes, many concurrent same-size buffers, a non-cacheable release, an
unknown/double free, bad `size`/`capacity`, an allocation that fits only after an
eviction and one that cannot fit even then, plus metamorphic and hidden checks
(the reuse-decision sequence must equal the reference; free order within a size
must not change any observable; long interleaved streams with mid-stream
evictions). A faster result that is wrong on even one case scores zero.

## Scope

Optimize the product implementation in `caching_allocator.py` only. Do **not** edit
tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single submission (READ CAREFULLY)

The grader compares the CPU instructions executed by your implementation with the baseline
instruction count, not elapsed time.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** submission. There is no iterative loop, no budget of
retries, and no per-round feedback to improve against: the moment you submit, the
task is over.

1. Read and profile the current implementation in `/app/repo/caching_allocator.py`, work out
   what is slow, and make it faster — taking as long as you need. Test your work
   yourself as much as you like *before* submitting: write your own scratch
   benchmarks and your own correctness checks in a scratch directory you create.
2. When — and only when — you are confident the solution is both correct and as
   fast as you can make it, submit it **once**:

   ```
   bash /opt/loop/submit.sh
   ```

3. **Submitting ends the task.** The harness records your submission and finalizes
   automatically. Later calls do not score new edits.

Because there is no second chance, spend your effort *before* the submission:
convince yourself the contract holds on every case you can think of (the shapes,
dtypes, boundary and error cases described above), and verify your own timing
shows a real improvement over the shipped implementation. A fast but incorrect
submission scores zero, and there is no round in which to repair it.

The grade is produced by a full, trusted end-of-session verifier over more
workloads than anything you can see, so make the one submission you have a
genuinely correct one.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo` and your own knowledge and reasoning.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — score ONLY by calling `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
