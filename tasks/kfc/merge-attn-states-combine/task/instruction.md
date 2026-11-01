# Performance Optimization Task

You are working inside a high-throughput LLM inference server. To keep long
sequences on-GPU, attention over the key/value sequence is computed in several
independent chunks: each chunk attends over a disjoint slice of the keys and emits a
**partial** attention output together with the log-sum-exp of that chunk's softmax
scores (its log-domain normalizer). The file `combine_attn_states.py` implements
`combine_attn_states` — it recombines those partial results into the single attention
output that attending over the whole sequence at once would have produced.

## Behavioral contract

Given `N` partial outputs `partial_out[n]` and their log-sum-exp normalizers
`partial_lse[n]`, each row `r` is recombined independently into the result that attending
over all the chunks at once would have produced. A chunk's log-sum-exp says how much of
that row's total softmax mass the chunk carried, so the row's combined output is the
average of the chunks' partial outputs weighted by their shares of the mass — a chunk
that saw more of the probability weight counts for proportionally more. The row's
returned normalizer is the combined log-domain normalizer of all the chunks together
(the log-sum-exp of the per-chunk normalizers). Both are evaluated stably in the log
domain, so widely separated or `-inf` normalizers neither overflow nor contribute
spurious weight.

1. `partial_out`: shape `(N, R, D)` — `N` partial outputs, `R` independent rows (one
   per query token/head), `D` the head/feature dimension. dtype `torch.float32`,
   `torch.bfloat16`, or `torch.float16`; CUDA tensor.
2. `partial_lse`: shape `(N, R)` — the matching log-sum-exp normalizer of each
   partial. Same dtype and device as `partial_out`. A value of `-inf` marks a chunk
   that saw no keys for that row (it contributes nothing).

The weighted average is accumulated in float32 for numerical stability. Each row is
combined independently, row order is preserved, and the result **does not depend on
the order of the `N` chunks**. If every chunk of a row is `-inf` (the row saw no keys
at all), that row's `out` is all-zeros and its `lse` is `-inf`.

Public signature (do NOT change):

```python
def combine_attn_states(
    partial_out: torch.Tensor,  # (N, R, D) fp32/bf16/fp16 CUDA
    partial_lse: torch.Tensor,  # (N, R) same dtype/device as partial_out
):  # -> (out, lse)
    ...
```

The call returns **two** tensors: `out` (shape `(R, D)`, dtype of `partial_out`) and
`lse` (shape `(R,)`, dtype of `partial_lse`).

Error contract: a non-floating (fp32/bf16/fp16) `partial_out` / `partial_lse`, or a
`partial_lse` whose dtype differs from `partial_out` → `TypeError`; shape violations
(`partial_out` not 3-D `(N, R, D)`, `partial_lse` not 2-D `(N, R)`, or their leading
`(N, R)` shapes disagreeing) → `ValueError`.

## Why the current implementation is slow

The current implementation forms the full `(N, R, D)` array of weighted partials as
an intermediate, writes it to memory, and then reduces it — with the max, the
exponentiation, the weighting, the reduction and the division each a **separate
operation** that round-trips data through global memory. The op is **memory-bound**:
it moves far more bytes and launches far more kernels than the computation itself
requires. Make it **faster on the GPU** while keeping the numerics within the
verifier's tolerance — you may use any GPU technique available in the image (for
example custom Triton kernels, cutting the intermediate memory traffic, tuning block
sizes) as long as the contract above holds.

**Forbidden:** delegating the whole combine to a pre-built attention state-combine
primitive from an external attention/inference library (for example a
`merge_attn_states`-style helper such as those shipped by flash-attn or vLLM). Build
the combine yourself from elementwise/reduction operations. The scoring harness scans
your submitted file for such external delegations and scores the task 0 if found.

## Correctness comes first

The verifier compares **both** of your outputs against a high-precision reference on
multiple workloads — fp32 and bf16, single-chunk (`N=1`) and many-chunk (`N=32`)
inputs, a non-power-of-two head dimension, a chunk that dominates, empty chunks and a
fully empty row (`-inf` normalizers), error-contract probes, and metamorphic checks
(permuting the chunk order leaves both outputs unchanged; adding a constant to every
normalizer leaves `out` unchanged and shifts every `lse` by that constant) — within a
fixed rtol/atol. A faster result outside tolerance on even one case scores zero.

## Scope

Optimize the product implementation in `combine_attn_states.py` only. Do **not** edit
tests, benchmark harnesses, workloads, or dependency/build files. The final submitted
diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

Reducing intermediate traffic can help, but the measured quantity is GPU execution time, not
a hardware byte counter.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of ``/app/repo/combine_attn_states.py``, decide on your
   approach, and make **all** the edits you want in `/app/repo`.
2. Self-test as much as you like with scratch scripts **you** write yourself
   (put them outside the scored file, e.g. under `/tmp`). Verify your output
   against the behavioral contract above on your own inputs, and time your own
   before/after to convince yourself the change is genuinely faster.
3. When — and only when — you are confident, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

**One submission only.** Once the submission is recorded, the harness finalizes it.
A later call does not score new edits. Do your own checks before submitting.

Because of that, everything rides on the work you do **before** you submit:
think the design through, read the baseline carefully, self-test the contract
(including the boundary and error cases described above), and only then submit.
Do not submit a half-finished or untested edit hoping to refine it later — there
is no later.

The grade is produced by a full, trusted end-of-session verifier (more workloads
than anything you can see), so your one submission must be genuinely correct as
well as fast.

**🎓 Explain your work for a beginner.** Before you edit, state your approach:
what makes the current code slow and what you intend to change. After you edit
and before you submit, give a short, concrete, step-by-step account of what you
changed, *why* it should be faster in terms of the actual code path, and how you
convinced yourself it is still correct.

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
