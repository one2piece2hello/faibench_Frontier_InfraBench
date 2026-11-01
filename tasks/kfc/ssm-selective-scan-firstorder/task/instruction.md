# Performance Optimization Task

You are working on the core sequence operator inside a state-space / linear-recurrence
block of a sequence model. The file `state_space_scan.py` implements
`state_space_scan` — it evolves a hidden state along the time axis by a first-order
linear recurrence and reads out a per-timestep output.

## Behavioral contract

Inputs (all the same floating dtype and CUDA device; `torch.float32` is the primary
dtype, `bfloat16` / `float16` are also accepted):

1. `A`: shape `(B, L, D, N)` — the per-timestep state-transition coefficient (applied
   elementwise to the state). `B` is the batch, `L` the time axis, `D` the channels,
   `N` the state width.
2. `B`: shape `(B, L, N)` — the per-timestep input projection.
3. `C`: shape `(B, L, N)` — the per-timestep output projection.
4. `x`: shape `(B, L, D)` — the input sequence.

Marching `t = 0, 1, ..., L-1` along the time axis from an initial state `h[-1] = 0`,
the operator is the standard **selective-scan / first-order linear state-space
recurrence**, carrying one scalar state per `(batch, channel d, state index n)` track:

- the per-step input drive of a `(D, N)` state slab is the outer product of that
  step's channel input `x[t]` with that step's input projection `B[t]`;
- the state at step `t` is the previous state scaled **elementwise** by that step's
  transition coefficient `A[t]` plus that step's input drive;
- the per-step output for a channel is the state of that channel contracted over the
  state-width axis `N` against that step's output projection `C[t]`, giving `y` of
  shape `(B, L, D)`.

The recurrence is **causal and strictly ordered in time**: `h[t]` (and therefore
`y[t]`) depends only on the inputs at steps `0..t`. Every `(batch, d, n)` track evolves
independently along `t`. The state is accumulated in float32 for numerical stability
(even when the inputs are bf16/fp16); the returned `y` matches the input dtype.

Public signature (do NOT change):

```python
def state_space_scan(
    A: torch.Tensor,   # (B, L, D, N) floating CUDA
    B: torch.Tensor,   # (B, L, N) same dtype/device as A
    C: torch.Tensor,   # (B, L, N) same dtype/device as A
    x: torch.Tensor,   # (B, L, D) same dtype/device as A
):  # -> y: (B, L, D), dtype of x
    ...
```

Error contract: any of `A` / `B` / `C` / `x` that is not a floating
(float16/bfloat16/float32/float64) `torch.Tensor`, or whose dtype differs from the
others → `TypeError`; wrong ranks (`x` not 3-D, `A` not 4-D, `B`/`C` not 3-D) or
inconsistent shared dimensions `(B, L, D, N)` across the inputs → `ValueError`.

## What to improve

The shipped implementation is correct but leaves a great deal of the GPU's capability
unused on long sequences; read and profile it yourself to establish what it is actually
bound on. Make it **faster on the GPU** while keeping the numerics within the verifier's
tolerance — you may use any GPU technique available in the image (`torch` and `triton`
are installed) **as long as the causal state-transition order and the result defined
above are preserved**.

**Forbidden:** delegating the entire recurrence to a library sequence-scan primitive.
Specifically, do not call `torch.associative_scan` / `associative_scan`,
`selective_scan_fn`, any `mamba_ssm` op, or a Triton `tl.associative_scan` builtin as
the whole answer. The scoring harness stubs these to raise at runtime, and the verifier
scans your submitted file for those tokens (do not reference them even in comments — the
scan is textual and scores the task 0). Build the scan yourself out of primitive ops /
your own kernel.

## Correctness comes first

The verifier compares your `y` against a high-precision reference on multiple workloads
— float32 and a bf16 long-sequence regime, a single timestep, a non-power-of-two length,
memoryless (`A = 0`) and pure prefix-sum (`A = 1`) degenerate cases, an amplifying
`A > 1` case (which must reproduce the true growth, not clamp), error-contract probes,
and **state-order-preserving** metamorphic checks (perturbing a future timestep must
leave the earlier outputs unchanged — causality; and running the scan on a truncated
prefix must reproduce the prefix of the full run) — within a fixed rtol/atol. A faster
result outside tolerance on even one case scores zero.

## Scope

Optimize the product implementation in `state_space_scan.py` only. Do **not** edit
tests, benchmark harnesses, workloads, or dependency/build files. The final submitted
diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can learn from: **submitting ends the
task.**

1. Read the current implementation in `/app/repo/state_space_scan.py` and work out, before you touch
   anything, what will actually make it faster.
2. Make your changes to `/app/repo/state_space_scan.py`. Test them yourself as thoroughly as you can
   with scratch scripts you write in your own scratch directory — that self-testing
   is your only safety net.
3. When — and only when — you are confident the implementation is both **correct**
   and **as good as you can make it**, submit it exactly once:

   ```
   bash /opt/loop/submit.sh
   ```

**Submitting is final and irreversible.**

- The moment `submit.sh` returns, this task is **over**. Stop editing and stop
  working on it.
- The harness finalizes automatically after the one submission. Calling it again
  does not score new edits.
- You will **not** receive iterative feedback to improve against. The receipt
  `submit.sh` prints is a record that your one submission was taken, not a signal
  to try again.

Because you only get one shot: think the design through first, re-read the
behavioural contract above, convince yourself every rule and error case is
handled, and satisfy yourself about performance — **then** submit.

Your score comes from a full, trusted end-of-session verifier (more workloads
than any self-test you can write), so correctness is not optional: a fast but
incorrect submission scores **zero**.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo` and your own knowledge and reasoning. `submit.sh`
is a one-shot submission channel, not a scoring oracle you may probe.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — the ONLY sanctioned scoring action is your single `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself. `submit.sh`
may be called exactly once, and only to submit your finished answer.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
