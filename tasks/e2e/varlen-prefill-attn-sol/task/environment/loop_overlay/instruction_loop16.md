# Variable-length causal prefill attention — get as close to the hardware as you can (loop16 protocol)

You have the complete source tree of **vLLM** at `/app/repo` (installed and importable) on a single
**NVIDIA H20** GPU. Your job: compute **variable-length causal prefill attention** over a ragged
batch of sequences as fast as the hardware allows, in bfloat16, matching a float32 causal reference.

The whole repository is yours — you may write a Triton kernel, a CUDA extension you compile yourself
(`nvcc` is present), or anything else. You do not have to use vLLM's own code.

## Required API contract (frozen — the verifier calls exactly this)

Your implementation must live at **`/app/repo/submission/varlen_prefill_attn.py`** and expose a class
`VarlenPrefillAttention` with `__init__(cfg)`, `prepare()`, and
`forward(q, k, v, cu_seqlens, max_seqlen, out)` exactly as in the shipped
`/app/repo/submission/varlen_prefill_attn.py` (start from that file — it is already correct).
`q` is `[total_tokens, num_q_heads, head_size]`, `k`/`v` are `[total_tokens, num_kv_heads,
head_size]` (grouped-query attention when `num_q_heads > num_kv_heads`), `cu_seqlens` is the int32
prefix-sum of sequence lengths, and `out` is a verifier-owned `[total_tokens, num_q_heads,
head_size]` buffer every element of which must be written. Everything your solution needs at scoring
time must persist under `/app/repo/submission/`.

## Rules the verifier enforces (failing any of them scores 0)

Exact **causal** attention (position t attends only to <= t), bit-correct enough to match a float32
causal reference within tolerance over hidden rows AND degenerate shapes (length-1 sequences, a
single long sequence, many short sequences), correct shapes/dtypes/devices, no crashes, and a bounded
memory allowance. Correctness is checked FIRST; see the shipped submission file and the base task
statement for the full contract (unchanged in this protocol).

---

## How you are graded (loop16: iterate in-session, at most 16 self-scored submissions)

Let `s` be your measured speedup over the strong baseline and `r` the calibrated reference
speedup over that same baseline. A failed validity check, `r <= 1`, or `s <= r` scores 0.
Otherwise the final reward is `min(1, ln(s/r) / ln(r))`, capped at 1 when `s >= r²`.

This task uses the **loop16 protocol**: in your session you may score the current
`/app/repo/submission/varlen_prefill_attn.py` and get feedback **at most 16 times, at least once —
you decide when to stop** (you need not use all 16).

### 1. Each round
After editing, run

```
bash /opt/loop/submit.sh
```

It runs your `VarlenPrefillAttention` over a **PUBLIC** development suite (the same one
`/app/dev_bench/run_dev_bench.py` uses — DIFFERENT from the scored workloads), enforces the **real
fp32 causal-parity** correctness gate (largest deviation from a float32 causal reference, relative to
the row RMS, must stay below tolerance), and returns: whether correctness PASSED, an achieved
**TFLOP/s**, best_so_far and the remaining budget.

🔴 **That dev TFLOP/s is a PUBLIC proxy, not your graded score.** Grading uses a HIDDEN workload
suite, a **strong baseline re-measured in ABBA pairs**, and a calibrated anchor — none of which is in
this container. The dev TFLOP/s is an ABSOLUTE number on the public shapes (it is NOT divided by the
hidden baseline), so it rises with kernel throughput but does not equal the graded speed-up. Use it
for **direction** only; do not overfit to the public shapes.

### 2. Finalize
When satisfied (or at the 16th round, which auto-finalizes) run

```
bash /opt/loop/submit.sh --finalize --reason "<one sentence on why you are stopping>"
```

`--reason` is mandatory and audited. The graded submission is your **best** correctness-passing round
(best-of-k, chosen by dev TFLOP/s and planted into `/app/repo` for grading; HEAD stays at the baked
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
literature on flash / paged attention, and your own ideas are all fair game — what is forbidden is
touching the measurement.
