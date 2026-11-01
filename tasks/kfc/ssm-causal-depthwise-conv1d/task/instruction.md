# Performance Optimization Task

You are optimizing a per-row sequence-mixing step inside the block of a
high-throughput sequence model served on GPU. The file `channel_window_op.py`
implements `channel_window_op` — for each row of a `(B, C, L)` tensor it computes a
short trailing-window weighted sum along the length axis, adds a per-row bias, and
applies a smooth gating activation.

## Behavioral contract

For an input `x` of shape `(B, C, L)` (batch, rows, length), a per-row weight matrix
`w` of shape `(C, K)`, and an optional per-row `bias` of shape `(C,)`, the output
`y` has shape `(B, C, L)` and is defined per `(b, c, t)` as follows.

Each output element is a **causal** trailing-window weighted sum taken along the length
axis of its own row: the row's `K` weights are applied to the `K` input positions of that
row ending at the current position `t`, with the **last** weight column paired with the
current position and each earlier column paired with a progressively older position.
Positions that fall before the beginning of the sequence count as zero, so the output
keeps the input length `L`. The row's `bias` (treated as zero when `bias is None`) is
added to that weighted sum, and the sum is finally passed through the **SiLU / swish**
gate — the value multiplied by its own logistic sigmoid.

Key properties:

1. Output position `t` reads only input positions `t-K+1 .. t` — never any position
   beyond `t`. The first `K-1` outputs use the zero-filled history.
2. Rows are independent: output row `c` depends only on `x[:, c, :]`, `w[c, :]` and
   `bias[c]`. Rows are never mixed.
3. The weighted sum is accumulated in float32 and cast back to the dtype of `x`.

`x`, `w` and (if given) `bias` share the same floating dtype — `torch.float32`,
`torch.bfloat16` or `torch.float16` — and live on CUDA. `K = w.shape[1]` is small
(typically 2-4) and `K >= 1`.

Public signature (do NOT change):

```python
def channel_window_op(
    x: torch.Tensor,             # (B, C, L) float32/bf16/fp16 CUDA
    w: torch.Tensor,             # (C, K) same dtype as x  -- per-row weights
    bias: torch.Tensor | None,   # (C,) same dtype as x, or None
):  # -> y: (B, C, L), dtype of x
    ...
```

Error contract: non-floating (fp32/bf16/fp16) `x`/`w`, a `w`/`bias` whose dtype
differs from `x`, or a `bias` that is neither `None` nor a tensor → `TypeError`;
`x` not 3-D, `w` not 2-D with first dim `C`, window length `K < 1`, or `bias` not
1-D of length `C` → `ValueError`.

## Why the current implementation is slow

The current implementation first **materializes a left-padded copy** of the input,
then accumulates the `K` taps **one at a time** — each tap is a separate full-tensor
read/modify/write that round-trips the whole `(B, C, L)` intermediate through global
memory — and finally runs the activation as **one more pass**. It is
launch/bandwidth-bound: it allocates an extra padded buffer, moves the data several
times, and launches many small kernels when the arithmetic itself is tiny (`K` is
small). Make it **faster on the GPU** while keeping the numerics within the
verifier's tolerance — you may use any GPU technique available in the image (for
example custom Triton kernels, cutting the intermediate memory traffic, or tuning
block sizes) as long as the contract holds.

**Forbidden:** the framework's built-in 1-D convolution primitives —
`torch.nn.functional.conv1d`, `F.conv1d`, `torch.conv1d`, and
`torch.nn.Conv1d` / `nn.Conv1d`. The scoring harness stubs these to raise at
runtime, and the verifier scans your submitted file for those tokens (do not
reference them even in comments — the scan is textual and scores the task 0). Build
the operation yourself (element-wise/shift arithmetic, `unfold`, or a hand-written
kernel are all allowed).

## Correctness comes first

The verifier compares your output against a high-precision float32 reference on
multiple workloads — bf16 / fp16 / fp32, a `K=1` pointwise case, a single-element
degenerate case, long sequences, error-contract probes, and metamorphic/structural
checks (no-future-leakage: perturbing the input tail leaves earlier outputs
unchanged; shift-equivariance; per-row independence; impulse-response identity and
delay) — within a dtype-keyed rtol/atol (tight for fp32, looser for bf16/fp16). A
faster result outside tolerance on even one case scores zero.

## Scope

Optimize the product implementation in `channel_window_op.py` only. Do **not** edit
tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can act on: **submitting ends the task.**

1. Work on `/app/repo/channel_window_op.py` until you believe it is both **correct** and as
   fast as you can make it. Test it yourself as much as you like —
   your own scratch scripts, your own timing harnesses, your own reasoning about
   the code path. None of that costs you anything.
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
