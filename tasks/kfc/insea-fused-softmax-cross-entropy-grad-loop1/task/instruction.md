# Performance Optimization Task

You are working on the loss stage of a large language model's training step. The
file `cross_entropy_grad.py` implements `cross_entropy_loss_grad` — it takes a
batch of per-row class scores (`logits`) and one integer target class per row
(`labels`) and returns the mean cross-entropy loss together with the gradient of
that loss with respect to `logits`.

## Behavioral contract

For a batch of `N` rows over `V` classes, each non-ignored row contributes the
**cross entropy between a smoothed target distribution and that row's softmax over
the `V` classes**, in the standard PyTorch label-smoothing convention: with
`s = label_smoothing`, the target distribution places weight `1 - s` on the row's own
label and spreads the remaining `s` **uniformly across all `V` classes, the label
included**. The returned `loss` is the mean of those per-row losses over the
non-ignored rows only (ignored rows enter neither the sum nor the row count). The
returned `grad` is the exact gradient of that returned mean loss with respect to
`logits`, one row per input row, with ignored rows carrying an all-zero gradient row.

1. `logits`: shape `(N, V)`, a floating-point CUDA tensor (`torch.float32` or
   `torch.bfloat16`). `N` is the number of rows (batch times sequence length),
   `V` the number of classes (vocabulary size).
2. `labels`: shape `(N,)`, an integer CUDA tensor. Each entry is a class index in
   `[0, V)`, **or** equals `ignore_index` to mark that row as ignored. Ignored
   rows contribute **zero** loss and **zero** gradient.
3. `ignore_index`: int (default `-100`).
4. `label_smoothing`: float `>= 0` (default `0.0`; `0.0` is ordinary cross
   entropy).

Reductions are computed in float32; the gradient is returned in the input dtype.

Public signature (do NOT change):

```python
def cross_entropy_loss_grad(
    logits: torch.Tensor,          # (N, V) float32/bfloat16 CUDA
    labels: torch.Tensor,          # (N,) integer CUDA; ignore_index marks ignored rows
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple:                        # (loss: scalar tensor, grad: (N, V) same dtype as logits)
```

Error contract: non-floating-point `logits`, or non-integer `labels` →
`TypeError`; `logits` not 2-D, `labels` not 1-D of length `N`, negative
`label_smoothing`, or any non-ignored label outside `[0, V)` → `ValueError`.

## What to improve

The current implementation is **correct but wasteful with intermediate state**: at
large vocabulary sizes the intermediates it keeps resident dominate the memory
footprint. Read it yourself and work out which buffers are alive at the same time.

Your goal is to produce the **same** loss and gradient while holding a far smaller
retained set — the extra working set beyond the input `logits` buffer should stay
small (on the order of `O(N)` plus a bounded working set, not another `O(N * V)`).
You are permitted to write the gradient into the `logits` buffer in place (the caller
passes a buffer you are free to overwrite). Use any GPU technique available in the
image as long as the contract above holds and the result stays within the verifier's
tolerance.

**Forbidden:** the framework's single-call fused cross-entropy loss —
`torch.nn.functional.cross_entropy` / `F.cross_entropy` and
`torch.nn.CrossEntropyLoss`. The scoring harness blocks these at runtime and
scans your submitted file for them (do not reference them even in comments — the
scan is textual and scores the task 0). Compute the loss and gradient yourself.

## Correctness comes first

The verifier compares your `loss` and full `grad` against a high-precision
reference on multiple workloads — ordinary and label-smoothed, rows with the
ignore sentinel (including an all-ignored batch), a single class (`V = 1`), a
single row, a vocabulary size that is not a round multiple, degenerate inputs,
error-contract probes, metamorphic checks (adding a per-row constant to all
logits leaves loss and gradient unchanged; permuting rows permutes the gradient
rows), and a work-evidence check (exactly the non-ignored rows carry a nonzero
gradient) — within a fixed tolerance. A result outside tolerance on even one case
scores zero.

## Scope

Optimize the product implementation in `cross_entropy_grad.py` only. Do **not**
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

1. Work on `/app/repo/cross_entropy_grad.py` until you believe it is both **correct** and as
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
