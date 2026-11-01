# Optimize a test-time-training linear-recurrence subsystem

## Objective

`fla/ops/ttt/chunk.py` provides a single public operation, `chunk_ttt_linear`, exposed
through a `torch.autograd.Function`. It implements a test-time-training (TTT) linear layer:
the sequence is processed in fixed-size mini-batches, and each mini-batch performs one
inner-loop gradient step on a linear "fast-weight" state `(h, hb)` under a layer-norm
reconstruction objective, then reads out through the updated state and an output layer
norm. For a mini-batch with queries/keys/values `q_i, k_i, v_i`, learning rate `eta_i`,
LN weight/bias `w, b`:

```
kh        = k_i @ h + hb ;  kh_hat = layernorm(kh)
g         = (w * kh_hat + b - (v_i - k_i)) * w
v_new     = ln_grad(g, kh_hat, rstd)                 # LN backward w.r.t. reconstruction
o_i       = q_i @ h - tril(eta_i * (q_i @ k_iᵀ)) @ v_new + hb - tril(eta_i) @ v_new
h, hb     = h - (eta_i[-1] * k_i)ᵀ @ v_new ,  hb - sum(eta_i[-1] * v_new)
o         = o_i + layernorm(o_i) * w + b
```

The current implementation is **functionally correct but slow**: it walks the sequence one
mini-batch at a time, so its run time grows with the number of dependent mini-batch steps
`T / chunk_size`.

Your job is to make `chunk_ttt_linear` **as fast as possible on the benchmark workload**
while preserving its public behaviour and numerical contract exactly. The reference result
is the recurrence above evaluated in fp32; your output must match it within the harness
tolerance across a range of shapes.

## Editable scope

You may edit **only** this file:

```
fla/ops/ttt/chunk.py
```

Any change to files outside this scope causes the whole task to score zero. The public
entry point and its input/output contract must remain unchanged:

```python
from fla.ops.ttt import chunk_ttt_linear

o, final_state, final_state_bias = chunk_ttt_linear(
    q, k, v, w, b, eta, scale=None, eps=1e-6, chunk_size=16,
    initial_state=None, initial_state_bias=None, output_final_state=False)
```

- `q`, `k`: shape `[B, T, H, K]`, CUDA (bf16 in the benchmark); `v`: `[B, T, H, V]` (`V == K`).
- `w`, `b`: output layer-norm weight/bias, shape `[H, V]`.
- `eta`: inner-loop learning rate, shape `[B, T, H, 1]` (or a scalar float).
- returns `o`: `[B, T, H, V]`; and when `output_final_state=True`, `final_state`
  (`[N, H, K, V]`) and `final_state_bias` (`[N, H, 1, V]`).
- `T` is an arbitrary positive integer (**not necessarily a multiple of `chunk_size`**);
  the operation must be correct for all of them (pad as needed), and for arbitrary `B`,
  `H`, and head width `D` (`K == V`).
- `chunk_ttt_linear` must remain driven through the same `torch.autograd.Function`; the
  forward maps its inputs to `(o, final_state, final_state_bias)` and the gradient path
  must stay consistent with the forward it implements.
- Determinism: given the same inputs, results must be stable across runs.

Helper primitives already present in the surrounding package (outside your scope) are
available for you to call.

## What is measured

- **Correctness** (hard gate): the operation's output (and final state when requested)
  must match the independent fp32 reference within a relative-norm tolerance, over a
  hidden suite of diverse shapes (short and long `T`, several non-multiple-of-`chunk_size`
  `T`, and varied `B`/`H`/`D`).
  Every tested shape must pass; partial correctness earns no reward.
- **Measured speedup**: wall-clock speedup of your implementation over the frozen
  baseline currently in `fla/ops/ttt/chunk.py`, measured on a long-sequence workload with
  warmup and repeated timed iterations.

**Final reward.** Let `s` be the candidate's measured speedup over the baseline and `r` the reference
implementation's speedup over that same baseline. A failed correctness or validity check, `r <= 1`,
or `s <= r` gives 0. Otherwise the reward is `min(1, ln(s/r) / ln(r))`: beating the reference
earns credit, and `s >= r²` earns 1. Development speedup is feedback, not the final reward.

## Rules (A-only red line)

Solve the task **only** by improving the implementation inside the editable scope. The
following cause the whole task to score zero:

- reading, importing, copying, or reconstructing the hidden verifier, reward script, or
  timing harness, or hard-coding to their internals;
- fetching or reusing any external/upstream implementation of this operation, or
  otherwise obtaining the answer rather than deriving it (no `git clone`/`fetch`/`pull`,
  no adding a remote, no `pip install`/`download` of the same package, no `wget`/`curl`
  of upstream files, no web lookup — whether or not the network appears reachable);
- routing the computation to a prebuilt/turnkey recurrence or state-space kernel from
  another module or package (e.g. a `fused_recurrent` entry point, a `fused_chunk` entry
  point such as `fused_chunk_ttt_linear` or its autograd `Function`, `mamba_ssm`,
  `selective_scan`, `causal_conv1d`, a library `associative_scan`, or a custom
  `torch.ops` operator) instead of computing the recurrence in your own code. This also
  covers calling or copying any **complete** implementation of this same operation that
  already exists elsewhere in the tree, even inside this package. You must derive the
  time-parallel form yourself. Composing small per-block helper primitives from the
  surrounding package is still fine — what is banned is delegating the whole operation;
- editing anything outside the declared scope, or weakening/bypassing the correctness check;
- attempting to detect the grading mode or bypass the runtime isolation.

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the performance of *this* task as far as you can. You have a
**submission budget of 1 to 16**: you may submit as few as **1** time and at
most **16** (hard ceiling). There is no minimum beyond a single submission and
no requirement to keep going — finalize as soon as you judge your solution good
enough.

**Iterate ONE round at a time, feedback-driven.** Each round is a single cycle:

1. **Edit** `/app/repo/fla/ops/ttt/chunk.py` — apply ONE concrete change based on your current
   hypothesis (round 1: based on reading the baseline code).
2. **Submit** by running `bash /opt/loop/submit.sh` **once** — it returns
   sanitized correctness + `dev_speedup` + `best_so_far` + `remaining` +
   `finalize_allowed` synchronously.
3. **Read the feedback in full.** If correctness FAIL, the named failing
   invariant is your fix target. If PASS, `dev_speedup` vs `best_so_far`
   tells you whether this round improved.
4. **Analyze in writing** — (a) what the feedback told you, (b) *why* this
   round's change had that effect (mechanistic reasoning), (c) *what specific
   single change* the next round will try.
5. **Edit again based on that analysis**, then go to step 2.

Do **NOT** call `submit.sh` in a batch; do **NOT** plan more than one attempt
ahead — round `k+1`'s plan must depend on round `k`'s measured outcome. Edit
only product code; never touch tests/benchmarks/workloads/build files.

If the feedback ever says `harness_error`, that submission was refunded and
the failure is NOT your code's fault — do not debug the harness; retry or
continue optimizing.

### 2. Termination

Two ways to end:

- (a) reach 16 submissions (auto-finalize triggers as a safety net); or
- (b) at any k with **1 ≤ k < 16**, once you believe you have reached your
  practical/theoretical limit (e.g. the last few submissions converged within
  run noise), run
  `bash /opt/loop/submit.sh --finalize --reason "<one concrete sentence on
  why you are stopping>"` — the reason is recorded and audited.

Early exit is a feature: you keep the best of your submissions, and a
*justified* stop scores exactly the same as burning all 16. You decide when to
stop; there is no minimum submission count.

### 3. 🔴 Red line (hard — ANY one of these ⇒ the whole task scores 0 in audit)

The scoring machinery is protected and off-limits. Solve this task
**independently**: rely ONLY on the code in `/app/repo`, the sanitized feedback
`submit` returns, and your own knowledge and reasoning.

- Do **NOT deliberately** read, `cat`, `open`, `less`, `head`/`tail`, `grep`,
  `find`, or edit ANY verifier / correctness / timing / scoring / hidden-test
  file or directory, wherever it lives. Executing `bash /opt/loop/submit.sh` is
  the ONLY sanctioned interaction with the scoring machinery.
- Do **NOT** run the verifier directly or try to reproduce / reverse-engineer
  it — score **only** by calling `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds,
  thresholds, the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up an upstream implementation or a reference
  solution (no web search, no `git` remote fetch, no network to the source),
  whether internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring /
  source material, however you came to see it.
- Do **NOT** call or reference the banned turnkey ops (`fused_recurrent`, `mamba_ssm`, `selective_scan`, `causal_conv1d`, `associative_scan`, `accelerated_scan`, `scan_combined`, `torch.ops`, `fused_chunk_ttt_linear`, `FusedChunkTTTLinear`, `.fused_chunk`, `import fused_chunk`, `ops.ttt import`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

Stay inside `/app/repo` and scratch directories you created yourself; use only
what `submit` returns for scoring signal.

### 4. 🎓 Explain for a beginner — per round, not in bulk

Before your first edit, explain your **approach** (the hypothesis round 1
tests). Then **after every `submit.sh` feedback — and BEFORE the next edit —**
give a step-by-step analysis: what the result told you, *why* the change had
that effect, and *what single change* you will try next, written so an
undergraduate can follow.

## 提交方式

会话内用 `bash /opt/loop/submit.sh` 自评并拿反馈,完成后 `bash /opt/loop/submit.sh --finalize`。
改动留在工作树里,不需要 `git commit`。
