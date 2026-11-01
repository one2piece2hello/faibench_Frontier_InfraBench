# Optimize a gated delta-rule linear-attention subsystem (Gated DeltaNet)

## Objective

`fla/ops/gated_delta_rule/` provides the chunk forward of a **Gated DeltaNet** (gated
delta-rule linear attention) sequence operator, exposed through a
`torch.autograd.Function` behind the public entry point `chunk_gated_delta_rule`. It
maintains a `K x V` recurrent state per sequence and value head which, at every time
step, is first decayed by that step's **scalar log-space forget gate** `g`, and then
updated by a **beta-weighted delta rule** — the decayed state is probed with the step's
key to obtain the value it currently predicts, the residual between that prediction and
the step's actual value is weighted by the step's `beta`, and the state absorbs that
correction as a rank-one update against the same key. The output for the step is the
updated state read against the step's query, scaled by the usual inverse-square-root of
the key width.

The current implementation is **functionally correct but slow**.

Your job is to make `chunk_gated_delta_rule` **as fast as possible on the benchmark
workload** while preserving its public behaviour and numerical contract exactly. The
reference result is the gated delta-rule recurrence above evaluated in fp32; your output
must match it within the harness tolerance across a range of shapes.

## Editable scope

You may edit **only** these three files:

```
fla/ops/gated_delta_rule/chunk.py
fla/ops/gated_delta_rule/chunk_fwd.py
fla/ops/gated_delta_rule/gate.py
```

Any change to files outside this scope causes the whole task to score zero.

All three sit on the operator's forward call chain: `chunk.py` holds the public entry
point, the `torch.autograd.Function` and the forward driver, and it imports two helpers
at module level — `gdn_gate_chunk_cumsum` from `gate.py` and
`chunk_gated_delta_rule_fwd_intra` from `chunk_fwd.py`. In the tree you are given, both
of those helpers are **non-functional stubs that raise `NotImplementedError`**, so
whatever the forward needs from them you have to supply. You may reorganise the work
across the three files, but `import fla.ops.gated_delta_rule` must keep succeeding (any
name imported at module level has to exist).

The public entry point and its input/output contract must remain unchanged:

```python
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

o, final_state = chunk_gated_delta_rule(q, k, v, g, beta, scale=None,
                                        initial_state=None, output_final_state=False,
                                        use_qk_l2norm_in_kernel=False,
                                        use_beta_sigmoid_in_kernel=False)
```

- `q`, `k`: shape `[B, T, H, K]`, CUDA (bf16 in the benchmark); the benchmark passes keys
  already L2-normalized along the last dim (unit-norm keys).
- `v`: shape `[B, T, HV, V]`, same dtype; the graded regime uses `HV == H`.
- `g`: shape `[B, T, HV]`, the per-step scalar forget gate **in log space** (values
  `<= 0`), same dtype.
- `beta`: shape `[B, T, HV]`, the per-step delta gate in `(0, 1)`, same dtype.
- `scale=None` selects the default scaling by the inverse square root of the key width.
- returns `o`: shape `[B, T, HV, V]`, the readout stream; and `final_state`: the final
  recurrent state of shape `[B, HV, K, V]` when `output_final_state=True`, otherwise
  `None`.
- `initial_state=None` means the state entering the first step is zero; when a state is
  passed (shape `[N, HV, K, V]`) it must be used as that entering state.
- `T` is an arbitrary positive integer (**not necessarily a power of two**); the operation
  must be correct for all sequence lengths, and for arbitrary batch `B` and head count
  `H`, and for the standard head widths `K`, `V` (e.g. `∈ {64, 128}`). Do not
  special-case the benchmark shape.
- the graded regime is dense and single-device: `cu_seqlens=None` (no variable-length
  packing), `cp_context=None` (no context parallelism), `HV == H`, and the in-kernel
  preprocessing flags off (`use_gate_in_kernel=False`, so `g` arrives already in log space;
  `use_qk_l2norm_in_kernel=False`, `use_beta_sigmoid_in_kernel=False`,
  `allow_neg_eigval=False`, `state_v_first=False`). The other paths are not exercised.
- `chunk_gated_delta_rule` must remain driven through the same `torch.autograd.Function`;
  the benchmark exercises the forward path only. If you change what the forward hands to
  its backward, keep the two consistent.
- Determinism: given the same inputs, results must be stable across runs.

Helper primitives already present in the surrounding package (outside your scope) are
available for you to call.

## What is measured

- **Correctness** (hard gate): the operation's output — and the final state, when
  requested — must match an independent fp32 sequential reference of the recurrence
  above within a relative-norm tolerance, over a hidden suite of diverse shapes (short
  and long `T`, a mix of power-of-two and non-power-of-two lengths, several batch sizes
  and head counts, and both head widths).
  Every tested shape must pass; partial correctness earns no reward.
- **Measured speedup**: wall-clock speedup of your implementation over the frozen
  baseline currently in the three scope files, measured on a single **H20** GPU on a
  long-sequence bf16 workload with warmup and repeated timed iterations.

**Final reward.** Let `s` be the baseline time divided by the candidate time on the final workload;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

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
  another module or package instead of computing it in your own code: the submission is
  rejected if any line it **adds** to a scope file mentions `fused_recurrent`,
  `mamba_ssm`, `selective_scan`, `causal_conv1d`, `associative_scan`,
  `accelerated_scan`, `scan_combined` or `torch.ops` (comments count) — this includes
  this package's own prebuilt `fused_recurrent_gated_delta_rule` / `fused_recurrent_gdn`
  path;
- delegating to this package's own eager reference implementation
  (`naive_recurrent_gated_delta_rule`) instead of computing the recurrence yourself;
- editing anything outside the declared scope, or weakening/bypassing the correctness
  check;
- reporting a measurement that does not reflect the full benchmark work (skipping work,
  caching results across timed iterations, or otherwise making the timing
  unrepresentative);
- attempting to detect the grading mode or bypass the runtime isolation.

Writing your own kernels (including Triton) inside the editable scope is allowed — the
ban above targets turnkey operators pulled in from elsewhere, not code you write here.

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the performance of *this* task as far as you can. You have a
**submission budget of 1 to 16**: you may submit as few as **1** time and at
most **16** (hard ceiling). There is no minimum beyond a single submission and
no requirement to keep going — finalize as soon as you judge your solution good
enough.

**Iterate ONE round at a time, feedback-driven.** Each round is a single cycle:

1. **Edit** `/app/repo/fla/ops/gated_delta_rule/chunk.py`, `/app/repo/fla/ops/gated_delta_rule/chunk_fwd.py`, and/or `/app/repo/fla/ops/gated_delta_rule/gate.py` — apply ONE concrete change based on your current
   hypothesis (round 1: based on reading the baseline code).
2. **Submit** by running `bash /opt/loop/submit.sh` **once** — it returns
   development feedback, including `dev_speedup`, `dev_reward`, `best_so_far` and the remaining budget.
3. **Read the feedback.** If the submission fails a development check, use the reported reason
   to guide your next edit. Otherwise compare `dev_speedup` across rounds; `best_so_far`
   identifies the submission currently retained by the development scorer.
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

There is no penalty for stopping after one or more submissions. Of the submissions that pass
all development checks, the harness selects one with the highest `dev_reward` for final grading.
Further rounds may change which candidate is selected; the development score is not the final reward.

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
- Do **NOT** call or reference the banned primitives (`fused_recurrent`, `mamba_ssm`, `selective_scan`, `causal_conv1d`, `associative_scan`, `accelerated_scan`, `scan_combined`, `torch.ops`) anywhere in your submitted file — the verifier scans your file textually for these tokens and scores the task 0 on a hit; do not even name them in comments.

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
