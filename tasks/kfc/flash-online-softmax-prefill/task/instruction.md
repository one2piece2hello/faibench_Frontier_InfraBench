# Performance Optimization Task

You are working on the attention step inside the transformer stack of a
high-throughput LLM inference server. The file `causal_attention.py` implements
`causal_attention` — for a whole input sequence it computes, for every query
position, a probability-weighted average of value rows based on scaled
query-key similarities.

## Behavioral contract

Given queries `q`, keys `k`, values `v`, a scalar `scale`, and a `causal` flag, the
function computes ordinary scaled dot-product attention. For every `(batch b,
query-head h, query-position i)`: the query vector is compared against every key
vector of the key/value head that head `h` shares (see point 2 below) by an inner
product over the feature dimension, scaled by `scale`; those similarities are turned
into a probability distribution over the key axis by a softmax; and the result is the
corresponding probability-weighted average of that head's value rows. When `causal`
is true, query position `i` may only see key positions up to and including `i` — the
later positions are excluded from the distribution outright and receive exactly zero
weight.

1. `q`: shape `(B, H, S, D)`, dtype `torch.bfloat16` or `torch.float16`, CUDA
   tensor. `H` = query heads, `S` = sequence length, `D` = per-head feature width
   (`D` is 64 or 128).
2. `k`, `v`: shape `(B, Hk, S, D)`, same dtype and device as `q`. `Hk` = key/value
   heads. `H` must be an integer multiple of `Hk` — the query heads are cut into
   consecutive groups of `H // Hk` heads and each group shares one key/value head, in
   order (the first group reads key/value head 0, the next reads 1, and so on).
   `Hk == H` means every head is independent.
3. `scale`: python `float` applied to the similarities before the softmax.
4. `causal`: `bool`. When `True`, query position `i` attends only to key
   positions `j <= i`; when `False`, to all positions.

The similarities and the softmax normalization are accumulated in float32 for
numerical stability; the output is cast back to the input dtype. Each query row
is normalized independently and position order is preserved.

Public signature (do NOT change):

```python
def causal_attention(
    q: torch.Tensor,        # (B, H, S, D)  bf16/fp16 CUDA
    k: torch.Tensor,        # (B, Hk, S, D) same dtype/device as q
    v: torch.Tensor,        # (B, Hk, S, D) same dtype/device as q
    scale: float,
    causal: bool = True,
):  # -> out: (B, H, S, D), dtype of q
    ...
```

Error contract: non-floating (bf16/fp16) `q` / `k` / `v`, or a `k` / `v` whose
dtype differs from `q` → `TypeError`; shape violations (`q` / `k` / `v` not 4-D;
`k` / `v` `B`/`S`/`D` axes not matching `q`; `k` and `v` shapes differing; `H`
not an integer multiple of `Hk`) → `ValueError`.

## Why the current implementation is slow

The current implementation forms the full `(S, S)` similarity matrix for every
`(batch, head)`, applies the mask, normalizes it into a full `(S, S)` probability
matrix, and multiplies by the values. Its **peak extra GPU memory grows with
`B · H · S · S` — quadratic in the sequence length** — so at long sequences it
allocates far more memory than the result itself needs and becomes memory-bound.

Make it use **less peak GPU memory** while keeping the numerics within the
verifier's tolerance. Concretely, the peak *extra* memory of a call must stay on
the order of **`O(B · H · S · D)`** (the size of the inputs and output), **not
`O(B · H · S · S)`** — the full score/probability matrix must not be the
dominant allocation. You may use any GPU technique available in the image (for
example custom Triton kernels, reducing the intermediate memory footprint, or
tuning block sizes) as long as the contract above holds.

**Forbidden:** the framework's built-in fused attention primitives —
`torch.nn.functional.scaled_dot_product_attention`, `F.scaled_dot_product_attention`,
`torch.scaled_dot_product_attention`, the private `aten` `_scaled_dot_product_*`
variants, `torch.nn.MultiheadAttention` / `nn.MultiheadAttention`, and any
external `flash_attn` package. The scoring harness stubs these to raise at
runtime, and the verifier scans your submitted file for those tokens (do not
reference them even in comments — the scan is textual and scores the task 0).
Build the attention yourself.

## Correctness comes first

The verifier compares your output against a high-precision reference on multiple
workloads — causal and non-causal, grouped-query (fewer key/value heads than
query heads), bf16 and fp16, an awkward (non-power-of-two) sequence length with
`D=64`, a single-position sequence, error-contract probes, and metamorphic checks
(scaling the values scales the output linearly; adding a constant vector to every
key leaves the output unchanged) — within a fixed rtol/atol. A result outside
tolerance on even one case scores zero.

## Scope

Optimize the product implementation in `causal_attention.py` only. Do **not**
edit tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your peak GPU memory use with the baseline on the same workloads, not
kernel latency.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can act on: **submitting ends the task.**

1. Work on `/app/repo/causal_attention.py` until you believe it is both **correct** and as
   memory-frugal as you can make it. Test it yourself as much as you like —
   for example, use your own inputs to check the outputs and measure peak GPU memory.
2. When — and only when — you are done, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

3. **That call is final.** It records your submission and finalizes automatically.
   Later calls do not score new edits; there is no second round.

Because you cannot iterate, **think the design through before you submit** and
self-test thoroughly: read the current implementation, decide on the change,
convince yourself the contract still holds on the normal, boundary, degenerate and
error-path cases described above, and only then submit. A submission that is fast
but wrong on even one case scores zero, and you cannot repair it afterwards.

The grade is produced by a full, trusted end-of-session verifier (more workloads
than anything you can see), so the submitted state must be genuinely correct — not
just correct on the cases you happened to try.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo`, the sanitized feedback that `submit.sh` returns,
and your own knowledge and reasoning.

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

Stay inside `/app/repo` and scratch directories you create yourself; use only
what `submit.sh` returns for scoring signal.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
