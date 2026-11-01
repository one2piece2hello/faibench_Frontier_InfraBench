# Variable-length causal prefill attention — get as close to the hardware as you can

You have the complete source tree of **vLLM 0.10.1.1** at `/app/repo` (installed and importable;
`import vllm`, `import torch`, `import triton` all work) on a single **NVIDIA H20** GPU.

When an LLM inference server accepts a batch of requests it must first run **prefill**: for every
prompt in the batch, every token attends causally over the tokens before it. A continuous-batching
engine packs those prompts back-to-back into one flat tensor and describes their boundaries with a
cumulative-length array, so one call handles a whole batch of prompts whose lengths differ wildly.
This is the single most arithmetic-heavy operation in the whole serving stack: its cost is
dominated by how much matrix-multiply throughput your code can actually extract from the GPU.

**Your job: make that packed causal prefill attention as fast as possible.**

The whole repository is yours. Change any part of it — write a Triton kernel, compile a CUDA
extension with the `nvcc` in the image, use or rewrite anything in `/app/repo`, call any library in
the environment, redesign the tiling, the masking, the numerics, the layout, the dispatch. Nothing
in `/app/repo` is off limits and you do not have to use vLLM's own code at all.

## What is measured

A verifier you cannot see runs your implementation against **hidden prefill workloads** (a range of
batch sizes, prompt-length mixes, head configurations and head sizes) and measures, **from outside
your process**, the wall-clock time of each attention call. It converts that into the fraction of
the GPU's measured peak dense `bfloat16` matrix-multiply throughput that your implementation
achieves for the arithmetic a correct causal attention must perform, and compares it with a
**well-tuned baseline** built from the strongest variable-length causal attention kernel this
environment already provides. The fraction of peak throughput is a diagnostic, not the reward.

Let `s` be your measured speedup over the strong baseline and `r` the calibrated reference
speedup over that same baseline. A failed validity check, `r <= 1`, or `s <= r` scores 0.
Otherwise the final reward is `min(1, ln(s/r) / ln(r))`, capped at 1 when `s >= r²`.

The shipped `/app/repo/submission/varlen_prefill_attn.py` is already correct. It still has to
beat the reference speedup to earn a positive reward.

## Required API contract (frozen — the verifier calls exactly this)

Your implementation must live at **`/app/repo/submission/varlen_prefill_attn.py`** and expose a
class `VarlenPrefillAttention`. Everything your solution needs at scoring time must persist under
`/app/repo/submission/` (edits to installed site-packages alone may be lost when your work is
replayed); if you build a compiled extension, build it into `/app/repo/submission/` and load it
from there.

```python
class VarlenPrefillAttention:
    def __init__(self, cfg: dict) -> None:
        """cfg keys (all present):
             num_q_heads      int  — query heads
             num_kv_heads     int  — key/value heads; num_q_heads is a multiple of it (GQA)
             head_size        int  — 64 or 128
             dtype            str  — "bfloat16" (the q/k/v/out element type)
             device           str  — e.g. "cuda"
             max_num_seqs     int  — upper bound on the number of sequences in a call
             max_seq_len      int  — upper bound on any single sequence length
             max_total_tokens int  — upper bound on the packed token count of a call
             causal           bool — always True
             softmax_scale    float— always 1/sqrt(head_size)
           The same instance is called many times with the same cfg."""

    def prepare(self) -> None:
        """Allocate any persistent workspace. Not timed. Called once after __init__."""

    def forward(self, q, k, v, cu_seqlens, max_seqlen, out):
        """ONE packed variable-length causal attention call. TIMED — this is the whole metric.

           q   : [total_tokens, num_q_heads,  head_size]   contiguous, dtype == cfg['dtype']
           k   : [total_tokens, num_kv_heads, head_size]   contiguous, dtype == cfg['dtype']
           v   : [total_tokens, num_kv_heads, head_size]   contiguous, dtype == cfg['dtype']
           cu_seqlens : int32 [num_seqs + 1] on `device`, non-decreasing, cu[0] == 0,
                        cu[-1] == total_tokens.  Sequence i occupies rows
                        cu[i] .. cu[i+1]-1.  A sequence may be EMPTY (cu[i] == cu[i+1]) and may
                        be as short as one token; lengths are arbitrary integers, not multiples
                        of any tile size.
           max_seqlen : python int, >= every sequence length in this call
           out : [total_tokens, num_q_heads, head_size] PRE-ALLOCATED by the caller, same dtype

           Semantics: for query row r of sequence i (0-based within the sequence), attend over
           that sequence's keys/values at positions 0..r inclusive, with softmax scale
           cfg['softmax_scale']; query head h uses key/value head h // (num_q_heads/num_kv_heads).

           You MUST write the result into `out` and return `out` (returning any other tensor is
           a failure). Every row of `out` must be written on every call.
        """
```

## Rules the verifier enforces (failing any of them scores 0)

1. **Numerical agreement.** The verifier recomputes the attention output itself in float32, from
   the exact q/k/v it gave you, over each query's *whole* causal prefix, and compares. On some
   workloads it checks **every** row, on others a hidden subset. Approximating the prefix — a
   sliding window, dropping or summarising far-away tokens, low-rank or lossy tricks — will not
   pass. The comparison is a scale-relative deviation per query row (the maximum elementwise
   difference divided by the RMS of that row's float32 reference) against a fixed tolerance that
   plain `bfloat16` arithmetic passes with a wide margin.
2. **Degenerate shapes.** Before anything is timed, the verifier runs a suite of small awkward
   batches — length 1, length 0 in the middle of a batch, lengths just below/at/above a tile size,
   a single sequence, one long prompt beside a tiny one, both head sizes, several GQA ratios — and
   checks **every** row of each against the float32 reference. All of them must be correct.
3. **The output buffer.** `out` arrives pre-filled with a sentinel value; if any element of it
   survives your call, or if you return a different tensor, or the shape/dtype is wrong, or a
   value is not finite, the run fails.
4. **Results must come from the inputs.** The verifier calls you with several different workloads
   of the same shape and requires different, individually correct outputs — a cached or replayed
   result fails.
5. **Do not modify the inputs.** `q`, `k` and `v` must be bit-identical after your call returns.
6. **Memory.** The verifier measures your allocation itself. Your persistent workspace plus the
   transient allocation of a timed call must stay within **3 GiB** — you cannot materialise the
   full score matrix of a long prompt.

## Iterating

`python3 /app/dev_bench/run_dev_bench.py` runs your implementation over a **public** development
workload set (different from the scored one), checks it against a float32 reference and prints the
per-call time and achieved TFLOP/s for each case. Use it to iterate; a lower time there generally
means a higher score, but the scored workloads are not the same.

## Hard red line (any one of these makes the whole task score 0)

- Do **not** read, `cat`, `grep`, `find`, copy or edit any verifier, hidden-test or evaluation file
  (anything under `/tests`, `/opt/verifier`, `/opt/negative`), and do not try to infer their
  contents.
- Do **not** disable, kill or tamper with the timer daemon or the measurement path.
- Do **not** access the internet at runtime, and do not attempt to bypass the proxy isolation.

Solve it yourself. The vLLM sources at `/app/repo`, the public literature on FlashAttention /
variable-length packing / GPU tiling, and your own ideas are all fair game — what is forbidden is
touching the measurement.

## 提交方式

把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `git commit`
(判分读工作树;仓库的 HEAD 必须停在初始基线 commit 上)。
