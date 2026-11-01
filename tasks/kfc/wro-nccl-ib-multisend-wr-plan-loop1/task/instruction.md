# Performance Optimization Task

Before an InfiniBand transport can hand a batch of sends to the NIC it has to turn a
*receiver-published credit window* into an actual chain of RDMA work requests: match each
posted send against the peer's clear-to-send FIFO, cut every request into per-queue-pair
chunks that respect the 128-byte write alignment the LL/LL128 protocols depend on, pick the
right local `lkey` and remote `rkey` for the device that each queue pair rides, advance the
local and remote cursors by exactly the bytes that were posted, and cap the chain with a
single signalled `RDMA_WRITE_WITH_IMM` that also gathers the per-request sizes into the peer's
completion-record array. That planning stage is this module.

It is functionally correct but **slow**: the plan is built one work request at a time. Each
record is materialised as a fresh dictionary, the per-request cursors live in a
**string-keyed** state dict that is re-formatted on every access, the `wr_id` packing table and
the queue-pair ring are **re-derived from scratch inside the hot loop**, the alignment rounding
walks up to the 128-byte boundary **one byte at a time**, and the aggregate report **rescans
the finished plan once per queue pair and once per request**. Make it **faster** on the
benchmark workload while **preserving its output exactly** — every returned list, dict and
scalar must match the reference element for element, and every documented `ValueError` must
still be raised.

## Editable scope

Edit **only** this file (any edit outside this scope scores the whole task zero):

```
ib_multisend_planner.py
```

## The subsystem

A *CTS batch* is `nreqs` FIFO slots published together by the receiver; they share one `idx`
generation counter, and the sender may not consume the batch until `slots[0]['idx']` equals the
generation it expects. Each slot carries the peer's `tag`, the number of bytes it is willing to
receive, the remote `addr`, and one `rkey` per **remote** device. Sends are matched against
slots by tag, in slot order, skipping slots already claimed. A request is then striped across
`nqps` queue pairs; each queue pair fixes a local device (which selects the `lkey`) and a
remote device index (which selects the `rkey`).

Seven entry points, all of them graded — `drain_cts_fifo` chains the others, and each of the
others is also graded on its own:

1. `chunk_span(size, nqps)` — the nominal per-queue-pair chunk,
   `DIVUP(DIVUP(size, nqps), 128) * 128`. A zero-byte send has a zero span. `nqps < 1` or
   `size < 0` raise `ValueError`.
2. `pack_wr_ids(slot, nreqs)` — the running `wr_id` table. The transport accumulates
   `wr_id += (slot & 0xff) << (r * 8)` as it walks the requests, so request `r` carries the
   **sum of the first `r + 1` bytes**, and the trailing signalled work request reuses the final
   value. `slot < 0`, `nreqs < 1` and `nreqs > 8` raise `ValueError`.
3. `match_send_slot(slots, expected_idx, tag, size, taken)` → `(r, clipped_size)` for the first
   unclaimed slot whose tag matches, with `clipped_size = min(size, slots[r]['size'])`.
   Returns `(-1, -1)` when the batch is not yet visible or when nothing matches. An **empty**
   batch, an out-of-range `nreqs`, and a matched slot carrying malformed receive info
   (`size < 0`, `addr == 0`, or `rkeys[0] == 0`) all raise `ValueError`.
4. `qp_ring(nqps, req_id, qp_dev, qp_remdev)` — the ordered
   `(qp_index, dev_index, rem_dev_index)` schedule; step `i` rides queue pair
   `(req_id + i) % nqps`. Ragged device tables, `nqps < 1` and `req_id < 0` raise `ValueError`.
5. `last_wr_plan(...)` — the trailing descriptor. An **extra** work request is appended when
   `nreqs > 1`, or when adaptive routing must be split off
   (`not (rem_ooo and local_ooo) and ar and req0_size > ar_threshold`). Only a multi-recv
   extra work request gathers sizes: `num_sge = 1`, `sizes_len = nreqs * 4`,
   `remote_addr = cmpls_addr + slot * 32`, and the per-device rkey table is copied. `imm_data`
   is `req0_id % 0xFFFFFFFF` under `MATCH_BY_ID` and `req0_size` under `MATCH_BY_INDEX`, masked
   to 32 bits, and `imm_be` is that value byte-swapped (`htobe32`). `chain_len` counts the data
   work requests plus the extra one. Bad `nreqs` or an unknown scheme raise `ValueError`.
6. `plan_multisend(...)` — the whole plan, **QP-major**: `for step in qp_ring(...): for r in
   range(nreqs)`, so index `j = step * nreqs + r`. For every request the posted length follows
   `len_i = min(size - offset_i, len_{i-1})` seeded with `len_-1 = chunk_span(size, nqps)`, and
   `offset_{i+1} = min(offset_i + len_i, size)`. Ten columns of `nqps * nreqs` plain ints
   (`qp_index`, `dev_index`, `req_index`, `length`, `num_sge`, `lkey`, `rkey`, `laddr`,
   `raddr`, `wr_id`) plus `n`, `nqps`, `nreqs`, `opcode`, the per-request `chunk` spans and the
   per-request `posted` totals. Bad `nreqs`, ragged per-request inputs and a negative size
   raise `ValueError`.
7. `multisend_stats(plan)` — `bytes_per_qp` (indexed by **`qp_index`**, not by step),
   `bytes_per_req`, `total_bytes`, `zero_len_wrs`, `sge_wrs`, `max_len`.

Contract notes that are easy to break:

* The chunk span is rounded **up**, so a request usually runs dry before the last queue pair.
  Those tail steps still emit a work request: `length = 0`, `num_sge = 0`, and the `lkey` is
  left **unset (`0`)** — but the `rkey` is still selected, because a zero-byte RDMA write still
  needs one.
* `laddr` / `raddr` advance by the length **actually posted**, so once a request runs dry its
  addresses stop moving.
* Because the span is a ceiling, at most one step per request carries a partial length, and it
  can legitimately be `0`.
* `bytes_per_qp` is keyed by the queue-pair **number**, which is a rotation of the step index
  by `req_id % nqps` — not by the step.
* `max_len` of an all-zero plan is `0`, and `zero_len_wrs + sge_wrs == n` always.
* Every emitted value is a plain Python `int`; `num_sge` is `1`/`0`, not `True`/`False`.
* `imm_be` is a byte swap of the masked 32-bit value, not a bit reversal and not a no-op.

## Constraints

* Pure Python standard library. No new dependencies, no C extensions, no subprocesses, no
  threads.
* Do not weaken, special-case or precompute for the benchmark inputs: the verifier runs a
  separate correctness suite (a hand-traced 2-QP/2-request plan with every column spelled out,
  five queue-pair fabrics from 1 to 8 queue pairs, request counts 1 to 8, all-zero and
  one-byte sends, shared-tag batches, stale generations, malformed slots, and three
  end-to-end FIFO drains) against an independent reference, and any mismatch scores **zero**.
* Do not touch the verifier, the tests directory, or anything outside `ib_multisend_planner.py`.

## How you are scored

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

Exact correctness is required: one mismatched element scores 0 regardless of speed.
The benchmark drains 1536 CTS batches over an
8-queue-pair, 2-device fabric (roughly 55k work requests); there is a large amount of headroom
and it **grows with the batch count**, so an algorithmically better plan builder wins by much
more than a micro-optimized one.

## Where to start

Read the module docstring, then each function's docstring: they pin the exact contract. Then
ask, for one request, what the length recurrence actually produces given that the span is a
ceiling of `size / nqps` — how many steps can be full, how many can be partial, how many must
be zero — and what that makes of the cursor arithmetic. Then ask which quantities in the hot
loop depend on the queue pair at all, and which are constant for the whole batch.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
