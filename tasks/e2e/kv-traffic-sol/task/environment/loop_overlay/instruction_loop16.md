# Paged KV-cache traffic — move the cache at the speed of the hardware (loop16 protocol)

You have the complete source tree of **vLLM 0.10.1.1** at `/app/repo` (installed and importable) on a
single **NVIDIA H20** GPU. An LLM serving engine keeps every request's keys and values in a **paged
KV cache** and spends a large part of its memory bandwidth just **moving that cache around**. Your
job: make that data movement as fast as the memory system allows.

The whole repository is yours. Change any part of it — pool layout and strides, the paging scheme,
the indirection, the Triton kernels, a CUDA extension you compile yourself (`nvcc` is present) — or
write something entirely new. Nothing in `/app/repo` is off limits.

## Required API contract (frozen — the verifier calls exactly this)

Your implementation must live at **`/app/repo/submission/kv_traffic.py`** and expose a class
`KVTrafficEngine` with `__init__(cfg)`, `allocate()`, `begin_step(plan)`, `scatter(layer, k_src,
v_src)`, `gather(layer, k_out, v_out)`, `copy_pages(layer, src_pages, dst_pages)` and `reset()`,
exactly as in the shipped `/app/repo/submission/kv_traffic.py` docstrings (start from that file — it
is already correct). Everything your solution needs at scoring time must persist under
`/app/repo/submission/`.

## Rules the verifier enforces (failing any of them scores 0)

Bit-exact round-trip (no lossy re-encoding), every output element written, nothing aliased instead
of stored, the current plan wins, allocation <= 1.10x the nominal pool size, exact shapes/dtypes/
devices, and correctness is checked FIRST over hidden geometries (single pages, partial tail pages,
unaligned starts, shuffled page order, shared pages, zero-length ranges, `page_size == 1`, several
layers). See the shipped `/app/repo/submission/kv_traffic.py` and the base task statement for the
full contract; it is unchanged in this protocol.

---

## How you are graded (loop16: iterate in-session, at most 16 self-scored submissions)

Let `s` be your measured speedup over the strong baseline and `r` the calibrated reference
speedup over that same baseline. A failed validity check, `r <= 1`, or `s <= r` scores 0.
Otherwise the final reward is `min(1, ln(s/r) / ln(r))`, capped at 1 when `s >= r²`.

This task uses the **loop16 protocol**: in your session you may score the current
`/app/repo/submission/kv_traffic.py` and get feedback **at most 16 times, at least once — you decide
when to stop** (you need not use all 16).

### 1. Each round
After editing, run

```
bash /opt/loop/submit.sh
```

It runs your `KVTrafficEngine` over a **PUBLIC** development suite (the same one
`/app/dev_bench/run_dev_bench.py` uses — DIFFERENT from the scored workloads), enforces the **real
bit-exact KV round-trip** correctness gate, and returns: whether correctness PASSED, a **dev
bandwidth (GB/s)**, best_so_far and the remaining budget.

🔴 **That dev bandwidth is a PUBLIC proxy, not your graded score.** Grading uses a HIDDEN workload
suite, a **strong baseline re-measured in ABBA pairs**, and a calibrated anchor — none of which is in
this container. The dev bandwidth is an ABSOLUTE number on the public shapes (it is NOT divided by
the hidden baseline), so it rises with engine throughput but does not equal the graded speed-up. Use
it for **direction** only; do not overfit to the public shapes.

### 2. Finalize
When satisfied (or at the 16th round, which auto-finalizes) run

```
bash /opt/loop/submit.sh --finalize --reason "<one sentence on why you are stopping>"
```

`--reason` is mandatory and audited. The graded submission is your **best** correctness-passing round
(best-of-k, chosen by dev bandwidth and planted into `/app/repo` for grading; HEAD stays at the baked
baseline commit). A justified stop at k=1 scores the same as using all 16; there is no minimum and no
"keep going until N rounds without gain" rule.

## Hard red line (any one of these makes the whole task score 0)

- Do **not** read, `cat`, `grep`, `find`, copy or edit any verifier / hidden-test / evaluation file
  (anything under `/tests`, `/opt/verifier`, `/opt/loop/private`, `/opt/negative`), and do not try to
  infer their contents — `bash /opt/loop/submit.sh` is the only sanctioned way to score.
- Do **not** run the grader directly, reproduce or reverse-engineer it.
- Do **not** disable, kill or tamper with the timer daemon or the measurement path.
- Do **not** access the internet at runtime, and do not bypass the proxy isolation.

Solve it yourself. The upstream vLLM sources at `/app/repo`, the public dev suite, the public
literature on paged attention, and your own ideas are all fair game — what is forbidden is touching
the measurement.
