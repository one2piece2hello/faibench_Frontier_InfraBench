# Performance Optimization Task

Before an RDMA transport can put a byte on the wire it has to answer two questions, and it has to
answer both of them on the hot path. **Which registered region does this address live in, and
which block should I hand out for a payload of this size?** — the NIC can only read memory that
has been registered with it, registration is far too expensive to do per message, so a transport
registers a handful of large slabs up front and carves fixed-size blocks (8 KiB / 64 KiB / 2 MiB)
out of them, keeps each tier's idle blocks in several independent buckets so concurrent
allocators do not all pile onto one list, and gives every worker a small thread-local cache of
the smallest tier that is refilled and drained in bulk. And **given a queue of outbound messages,
what work requests should I post?** — a reliable-connected queue pair may only have as many sends
outstanding as the peer has receives posted, so the sender runs a credit window, piggybacks the
acknowledgement in the 32-bit immediate field of the next send, solicits a completion event from
the peer only once enough unsolicited traffic has piled up, and signals its own send completions
only once every quarter window because reaping a completion queue entry costs more than the send
did. That planning core — the registration table, the tiered block pool, and the work-request
cutter with its credit accounting — is this module.

It is functionally correct but **slow**: a region's usable size is found by **adding one whole
stripe at a time** until the request is used up; a new region is checked for overlap against
**every** region already registered and then walked back into place by an **insertion sort**;
resolving an address to its region **scans the whole table**, and this happens at least once for
every block allocated, freed or posted; `bucket_index` resolves **the same address three times**;
the pool **re-counts the thread-local cache and re-sums a whole bucket** on every allocation, and
its bulk refill **recounts the cache inside the refill loop**; a free that overflows the cache
finds the tail of the batch it is about to splice out by **restarting the walk from the head for
every position, twice**; `pool_stats` makes **five separate walks of the same lists per tier**;
the work-request cutter **rebuilds the current request's byte count and the current message's
offset from scratch for every scatter/gather entry it adds, and re-materialises both lists by
concatenation**, so filling one request costs the square of the number of entries in it, and it
**materialises a list of every message index still behind the cursor** just to ask whether any
are left; completion reaping makes **one pass per event kind**, resolves each receive's address
**three times**, and collects the distinct slots in a **linearly scanned list**; and
`window_report` makes **eight passes over the same plan** and dedups its slots in a list too.
Make it **faster** on the benchmark workload while **preserving its output exactly** — every
returned dict, list, tuple and scalar must match the reference element for element, every
mutation it makes to the pool, the region table and the endpoint state must be the same, and
every documented `ValueError` must still be raised.

## Editable scope

Edit **only** this file (any edit outside this scope scores the whole task zero):

```
brpc_rdma_pool_window.py
```

## The subsystem

`BLOCK_SIZES` is `(8192, 65536, 2 * BYTES_IN_MB)` and the tier ids `BLOCK_DEFAULT`,
`BLOCK_LARGE`, `BLOCK_HUGE` are `0, 1, 2`; `TIER_NAMES` is `("default", "large", "huge")`. A
*region table* is `{"regions": [...], "next_lkey": int}` whose `regions` list is kept **ascending
by `start`** and whose entries are `{"start", "size", "block_type", "lkey"}`. A *pool* is a dict
whose `idle[tier][bucket]` and `expansion[tier][bucket]` are singly linked lists of
`{"start", "len", "next"}` nodes, plus `idle_size` / `expansion_size` byte totals, `region_num`
per tier, the thread-local cache `tls` / `tls_num`, the deterministic bucket picker `rng`, the
next region base `next_base`, and the `allocated` / `freed` / `extends` counters. An *endpoint
state* holds the four negotiated depths, `local_cap` and `remote_cap`, the two windows
`sq_window` and `remote_rq_window`, the immediate-only reserve `sq_imm_window`, the ring cursor
`sq_current`, the batching counters `sq_unsignaled` / `unsolicited` / `unsolicited_bytes` /
`accumulated_ack`, the pending credit `new_rq_wrs`, and the `posted` / `imm_sent` /
`acks_flushed` totals. `RESERVED_WR_NUM` is `3`, `MAX_REGIONS` is `16`, `ACK_FLUSH_SHIFT` is `1`,
`UNSOLICITED_BYTE_LIMIT` is `1048576`, and every number the module produces is a plain Python
`int` — there is no float arithmetic anywhere.

Twenty-two entry points, all of them graded — `run_transfer_tick` chains most of the others, and
each of the others is also graded on its own:

1. `block_type_for(size)` — the smallest tier that holds `size` bytes. A non-int, a `size` below
   1 or a `size` above the largest block raises `ValueError`.
2. `regularize_region_size(size_mb, block_type, buckets)` — the request truncated to a whole
   multiple of `block_size * buckets`, because a region is split evenly across the buckets and
   every sub-slab must be a whole number of blocks. A `size_mb` outside
   `[MIN_POOL_MB, MAX_POOL_MB]`, a bad tier, a `buckets` outside `[MIN_BUCKETS, MAX_BUCKETS]`, or
   a request that holds no block at all raises `ValueError`.
3. `region_table_new()` — an empty table, `next_lkey` `1`.
4. `region_table_add(table, start, size, block_type)` → the new `lkey`, and the table stays
   sorted by `start`. Regions may not overlap. A table already holding `MAX_REGIONS` regions, a
   `start` that is not a positive multiple of `4096`, a non-positive `size`, a `size` that is not
   a multiple of the tier's block size, a bad tier, or any overlap raises `ValueError`.
5. `region_of(table, addr)` → the entry containing `addr`, or `None`. A non-int `addr` or a
   malformed table raises `ValueError`.
6. `region_lkey(table, addr)` → that entry's `lkey`, or `0` when the address is unregistered.
7. `bucket_index(table, addr, buckets)` → `((addr - start) * buckets) // size` for the containing
   region. An unregistered address or a bad `buckets` raises `ValueError`.
8. `pool_create(buckets, tls_cache_num, base_addr=0x200000, seed=1)` — an empty pool.
   `tls_cache_num` must be an **even** int in `[2, 4096]`; `base_addr` must be a positive
   4096-aligned int; `buckets` must be in `[MIN_BUCKETS, MAX_BUCKETS]`.
9. `pool_extend(pool, table, size_mb, block_type)` → `{"base", "size", "lkey", "stripe"}`.
   Registers one more region at `next_base`, stripes it into `buckets` equal sub-slabs pushed
   onto the tier's **expansion** lists (not the idle lists), bumps `region_num` and `extends`,
   and advances `next_base` past a guard gap of one largest block, rounded **up** to a multiple
   of the largest block so that a block address is always aligned to its own tier.
10. `pool_alloc(pool, table, size, grow_mb=MIN_POOL_MB)` → the address of one block. The smallest
    tier is served from the thread-local cache first. Otherwise the bucket is
    `rng_next % buckets`; an empty idle list is filled from that bucket's expansion list, and if
    that is empty too the pool extends by `grow_mb` first. The bucket's head chunk is shaved by
    one block, or unlinked when it held exactly one. For the smallest tier the rest of that
    bucket's whole blocks are then pulled into the thread-local cache in bulk, stopping at the
    first chunk larger than one block or once the cache is more than half full.
11. `pool_recycle_tls(pool, table)` → how many blocks were flushed from the thread-local cache
    back onto the shared buckets.
12. `pool_dealloc(pool, table, addr)` → `0`. Smallest-tier blocks go to the thread-local cache;
    once it is full, `tls_cache_num // 2` of them plus the block being freed are spliced back
    onto the bucket derived from the address in a single batch. An unregistered address, or one
    that is not on a block boundary of its region's tier, raises `ValueError`.
13. `pool_stats(pool)` → per tier the `chunks`, `idle_bytes`, `expansion_chunks`,
    `expansion_bytes`, `max_chunk` and `regions`, plus `tls_blocks`, `tls_num`, `allocated`,
    `freed` and `extends`.
14. `window_capacities(sq_size, rq_size, remote_sq_size, remote_rq_size)` — each depth clamped
    into `[MIN_QP_SIZE, MAX_QP_SIZE]`, then `local_cap = min(sq, remote_rq) - RESERVED_WR_NUM`
    and `remote_cap = min(rq, remote_sq) - RESERVED_WR_NUM`. A non-int depth, or depths that
    leave either window below `1`, raises `ValueError`.
15. `endpoint_init(sq_size, rq_size, remote_sq_size, remote_rq_size, remote_recv_block, max_sge)`
    — the capacities plus both windows opened to `local_cap`, `sq_imm_window` at
    `RESERVED_WR_NUM` and every counter at zero. A `remote_recv_block` that is not one of
    `BLOCK_SIZES`, or a `max_sge` outside `[1, 64]`, raises `ValueError`.
16. `plan_send_wrs(state, table, msgs)` → `{"wrs", "total_len", "eagain", "consumed",
    "messages"}`. Cuts a non-empty queue of `{"addr", "len", "block"}` descriptors into work
    requests. Each request carries at most `max_sge` scatter/gather entries and at most one peer
    receive block worth of bytes, and an entry never straddles a source block. Every request
    takes the whole outstanding receive credit into its `imm` field (zeroing `new_rq_wrs`), gets
    `slot` from the ring cursor `sq_current` (which wraps at `sq_size - RESERVED_WR_NUM`), and
    carries the `solicited` and `signaled` flags the batching policies dictate; a signalled
    request's `wr_id` is the number of unsignalled requests it closes out, and `0` otherwise.
    Running out of either window stops the cut early; running out before a single request could
    be built returns `eagain` with an empty plan. A malformed `state`, an empty or malformed
    `msgs`, a `len` above its `block`, an `addr` not aligned to its `block`, or an address that
    is not registered raises `ValueError`.
17. `send_imm(state, imm)` → `1` if a bare immediate-only request was posted, else `0`. A
    negative or non-int `imm` raises `ValueError`.
18. `send_ack(state, num)` → whatever `send_imm` returned, accumulating `num` reposted receives
    into `new_rq_wrs` and flushing only when the count **before** this call was already past half
    the remote window. A negative or non-int `num` raises `ValueError`.
19. `handle_completion(state, table, events)` → `{"send_events", "recv_events",
    "credits_returned", "reposted", "acks", "bad_lkey", "distinct_slots"}`. A send event returns
    its `wr_id` worth of send credit, clamped to the room left in the window; a receive event
    returns its `imm` worth of remote receive credit the same way, and then either counts as a
    bad key (its address is unregistered) or as a repost that accumulates one ack. A
    non-sequence, an event without a `kind`, a `kind` outside `EVENT_NAMES`, a negative `wr_id`
    or a negative `imm` raises `ValueError`.
20. `window_report(wrs)` → `{"wr_count", "total_bytes", "sge_count", "max_sge", "solicited",
    "signaled", "signal_bytes", "imm_total", "distinct_slots", "max_bytes", "first_slot",
    "last_slot"}`; an empty plan reports zeros with both slots `-1`. A non-sequence raises
    `ValueError`.
21. `run_transfer_tick(cfg, seed=1)` — pre-registers `cfg["pre_regions"]` regions round-robin
    over the tiers, then per step allocates, cuts, reports, reaps the completions those requests
    would raise, frees back down to `cfg["live_blocks"]`, and folds a 48-bit digest. Returns the
    digest, the plan and byte totals, the endpoint counters, the region and pool counters, and
    `pool_stats` before and after a final `pool_recycle_tls`. Every key of `CFG_KEYS` must be
    present and a plain `int`; `steps` below 1, any of `alloc_per_step` / `msgs_per_step` /
    `recv_per_step` / `max_alloc` below 1, a `live_blocks` below `alloc_per_step`, a `max_alloc`
    above the largest block, or a `pre_regions` outside `[0, MAX_REGIONS - 1]` raises
    `ValueError`.
22. `transfer_sweep(cfgs, seed=1)` → `{"rows", "guard", "bytes_total", "count"}` over a non-empty
    sequence of configurations, tick `i` run with `seed + i`.

Contract notes that are easy to break:

* `regularize_region_size` truncates **down**; a request that does not hold one whole block per
  bucket is an error, not a zero-sized region.
* The region table is sorted by `start` **and** the regions are disjoint — but only because
  `region_table_add` enforces both. `region_of` must agree with a linear scan on every address,
  including one exactly on a region's `start` and one exactly on its end (which is *outside*).
* `pool_extend` puts the new memory on the **expansion** list; it only reaches the idle list when
  an allocator finds its bucket empty, and promotion requires the idle list to be empty.
* `pool_alloc`'s bulk refill leaves `idle_size` deliberately unadjusted for the blocks it moves
  into the thread-local cache — the next allocation from that bucket recomputes it. `pool_stats`
  reports the byte totals from the lists themselves, never from `idle_size`.
* The bulk refill stops **before** a chunk that is larger than one block, and its half-full test
  is `tls_num > tls_cache_num // 2` — strictly greater.
* `pool_dealloc`'s batch splices the **first** `tls_cache_num // 2` cached blocks, in cache
  order, followed by the block being freed, onto the front of the bucket's idle list, and the
  bucket comes from the freed address, not from the cached ones.
* `send_ack` compares the value of `new_rq_wrs` **before** the accumulation against
  `remote_cap >> ACK_FLUSH_SHIFT`, so the flush always happens one call late.
* A work request's `solicited` flag is forced when either window is down to its last credit or
  when no message is left behind the cursor; only otherwise do the three accumulator thresholds
  get a say, and every one of them is a strict `>`.
* `plan_send_wrs` returns `eagain` **only** when not a single request could be built; a partial
  cut is a normal, non-`eagain` result with `consumed` below the queue's total length.
* `handle_completion` clamps a returned credit to the room actually left in the window, so an
  over-large `wr_id` or `imm` is silently truncated rather than rejected.
* `window_report`'s `distinct_slots` counts distinct `slot` values, and `first_slot` /
  `last_slot` come from the ends of the plan, not from the smallest and largest slot.
* Every emitted number is a plain Python `int` and every scatter/gather entry is a plain
  `(addr, len, lkey)` 3-tuple — there is no float arithmetic anywhere and there must not be any
  after your change.

## Constraints

* Pure Python standard library. No new dependencies, no C extensions, no subprocesses, no
  threads.
* Do not weaken, special-case or precompute for the benchmark inputs: the verifier runs a
  separate correctness suite (the constants themselves; `block_type_for` and
  `regularize_region_size` swept over their boundaries; a registration table built **out of
  order** with on-edge, ±1 and fully-inside overlap probes and address probes on every region
  boundary; four pool shapes driven through 400 interleaved allocate/free operations with
  `pool_stats` and the whole region table compared throughout; the credit windows over seven
  depth shapes including clamping at both ends; `plan_send_wrs`, `window_report` and
  `handle_completion` over six window shapes, six rounds each, with structural invariants on
  every emitted request; `run_transfer_tick` over five configurations × two seeds plus the sweep;
  and roughly seventy documented `ValueError` cases) against an **independent** reference written
  from the contract rather than from this code — its registration table is an interval list keyed
  through `bisect`, its pool keeps Python lists instead of linked nodes, and its cutter is driven
  by a per-message cursor array — and any mismatch scores **zero**.
* Do not touch the verifier, the tests directory, or anything outside `brpc_rdma_pool_window.py`.

## How you are scored

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

Exact correctness is required: one mismatched element scores 0 regardless of speed.
The benchmark drives three transfer configurations —
several hundred allocations, a work-request cut of several hundred messages at up to 64
scatter/gather entries each, and several hundred completion events per step — across the three
block tiers and fourteen registered regions. There is a large amount of headroom and it **grows
with the width of the work request and with the size of the thread-local cache**, so removing a
quadratic beats shaving a constant by a wide margin. A partially optimised reference that fixes
only the passes and the repeated lookups — the region sizing loop, the triple address
resolution, the five-walk `pool_stats`, the restarted batch walk in `pool_dealloc`, the one-pass-
per-event-kind completion reaper and the eight-pass report — but leaves the table scan, the
pool's recomputed counters and the structure of the work-request cutter alone reaches about
**1.23x**.

## Where to start

Read the module docstring, then each function's docstring: they pin the exact contract. Then ask
what each routine really needs from the data it walks — whether a truncation to a multiple has to
be found by repeated addition, whether a sorted disjoint interval list has to be scanned to find
the interval containing an address or checked against every one of its members to find an
overlap, whether an address that has already been resolved has to be resolved again, whether a
counter the pool maintains exactly has to be recomputed by walking the list it counts, whether
the tail and the byte total of a fixed-length prefix of a linked list need more than one walk of
it, whether the byte count of a work request under construction has to be re-summed for every
entry appended to it, whether asking "is any message left?" requires building the list of them,
and how many times a finished plan or a batch of completions actually has to be read.

## How you are scored (ONE single graded submission)

- You get **exactly one** graded submission. Submit it with
  `bash /opt/loop/submit.sh`.
- **Submitting ends the task.** The moment `submit.sh` returns you are done: stop
  editing and stop working on this task.
- A second call to `submit.sh` is **refused** and exits non-zero. There is no
  retry, no budget of attempts, no "best of several submissions", and no separate
  finalization step — your one submission *is* the final answer.
- You will **not** get iterative feedback to improve against. Nothing is measured
  for you round by round, so there is no measured signal to chase.
- Whatever state the in-scope file is in when you submit is exactly what gets
  graded. Nothing is restored, re-selected or rolled back for you.
- Therefore: read the code, settle the design, and **self-test thoroughly with
  your own scratch scripts** before you submit. Correctness is a hard gate — a
  fast but incorrect submission scores **zero**.
- Explain your approach and your reasoning in writing before you submit.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
