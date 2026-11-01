# Performance Optimization Task

You are working on a sequence-mixing layer inside a long-context language model.
The file `gated_state_recurrence.py` implements `gated_state_recurrence` — a causal
layer that mixes a sequence of tokens through a small, continuously updated
**running state** with a learned per-feature decay, instead of through pairwise
attention.

## Behavioral contract

For each `(batch, head)` the layer maintains a state matrix `S` of shape
`(Dk, Dv)`, initialised from `initial_state` (or zeros) and carried across the `L`
positions in **strictly causal, left-to-right** order — the output at a position may
depend only on that position and the ones before it. Mathematically the layer is the
standard **gated linear-attention / gated-decay state recurrence**: at each position
the carried state is first attenuated by that position's decay gate, then the
position's key/value pair is written into it, and only then is the state read out by
that position's query.

Three properties of that per-position step are part of the contract:

1. The decay is **per key feature** and applies to the *carried* state: the factor
   derived from the gate at feature `d` attenuates row `d` of `S` (the `Dk` axis),
   uniformly across the `Dv` axis.
2. The current key/value pair enters as a **rank-1 (outer-product) write** added to
   the already-decayed state — the key indexes the `Dk` axis, the value the `Dv` axis.
3. The query reads the state **after** that same position's write (never the state
   carried in from the previous position), contracting over the `Dk` axis to give one
   `Dv`-vector per position; the query is pre-scaled by `Dk ** -0.5`.

Inputs / outputs:

1. `q`, `k`: shape `(B, H, L, Dk)` — per-position query and key. Same floating
   dtype (`torch.bfloat16` or `torch.float16`), CUDA tensors.
2. `v`: shape `(B, H, L, Dv)` — per-position value, same dtype/device as `q`.
3. `g`: shape `(B, H, L, Dk)` — per-position, per-key-feature gate in **log
   space**; the multiplicative decay applied to the state is `exp(g)` (so `g == 0`
   keeps the state, `g < 0` decays it, a very negative `g` nearly forgets it). Same
   dtype/device as `q`.
4. `initial_state`: optional `(B, H, Dk, Dv)` starting state (`None` = zeros).
5. `output_final_state`: if `True`, also return the final state `S`.

The state is accumulated in **float32** for stability; the returned output is cast
back to the input dtype. The decay-then-write-then-read ordering described above, and
the strictly causal per-position dependency, are part of the contract.

Public signature (do NOT change):

```python
def gated_state_recurrence(
    q: torch.Tensor,             # (B, H, L, Dk) bf16/fp16 CUDA
    k: torch.Tensor,             # (B, H, L, Dk) same dtype/device as q
    v: torch.Tensor,             # (B, H, L, Dv) same dtype/device as q
    g: torch.Tensor,             # (B, H, L, Dk) same dtype/device as q; decay = exp(g)
    initial_state=None,          # optional (B, H, Dk, Dv)
    output_final_state=False,
):  # -> o (B, H, L, Dv) dtype of q, or (o, final_state) when output_final_state
    ...
```

Error contract: non-floating (bf16/fp16) or dtype-mismatched `q`/`k`/`v`/`g` →
`TypeError`; a `k` whose shape differs from `q`, a `g` whose shape differs from
`q`, a `v` not sharing `(B, H, L)`, or an `initial_state` not of shape
`(B, H, Dk, Dv)` → `ValueError`.

## Why the current implementation is slow

The current implementation is a **Python loop over the `L` positions**. Every
position launches its own handful of small GPU operations, and each state update
must wait for the previous one to finish — the sequential dependency chain, not the
arithmetic, dominates the runtime. The per-position work (a per-row scaling, a
`(Dk, Dv)` outer product, and a contraction) is tiny relative to the launch and
latency overhead of doing it thousands of times in order.

Make it **faster on the GPU** while keeping the numerics within the
verifier's tolerance — you may use any GPU technique available in the image (custom
Triton kernels, batched matmuls, a restructured traversal of the sequence, etc.) as
long as the contract above holds.

**Forbidden:** do **not** import or call a third-party or framework-provided
sequence-mixing / attention / state-recurrence library operator to do the work for
you, nor a generic `scaled_dot_product_attention` shortcut (the scoring harness
blocks such library entry points at runtime and the verifier scans your submitted
file for them — referencing one scores the task 0). Build the recurrence yourself
from primitive tensor operations. Keeping the naive per-position Python loop is
allowed but will not earn a speedup.

## Correctness comes first

The verifier compares your output — and, where requested, the final state —
against a high-precision reference on multiple workloads: bf16 and fp16, square and
rectangular `(Dk, Dv)`, a single position, a non-zero initial state, a
no-decay gate (`g == 0`), a strong-decay gate (state nearly reset each step), an
all-zero value, error-contract probes, and metamorphic checks (**prefix
causality** — outputs for the first positions are unchanged when the sequence is
extended; **state threading** — running a prefix and feeding its final state into
the suffix reproduces the full-sequence output, so the block partition cannot
change the value), plus a hidden mixed-decay case — within a fixed rtol/atol. A
faster result outside tolerance on even one case scores zero.

## Scope

Optimize the product implementation in `gated_state_recurrence.py` only. Do **not**
edit tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of ``/app/repo/gated_state_recurrence.py``, decide on your
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
