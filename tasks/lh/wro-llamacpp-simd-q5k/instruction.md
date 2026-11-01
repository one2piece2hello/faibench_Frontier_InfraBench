# Performance Optimization Task

## What you are given

A frozen checkpoint of a CPU inference library at `/app/repo` (the `ggml`
tensor library used by `llama.cpp` for on-device / edge inference). The
matrix-multiply inner loop dot-products a 5-bit K-quant (`Q5_K`) weight row against its paired
activation row. The x86 implementation of this hot kernel, `ggml_vec_dot_q5_K_q8_K` in

```
ggml/src/ggml-cpu/arch/x86/quants.c
```

currently delegates to a plain scalar reference: for every 256-element super-block it reconstructs each 5-bit weight (four low bits from `qs`, one high bit from `qh`), unpacks the 6-bit per-sub-block scale and min from the packed `scales`, forms `d·scale·Σ(q5·q8) − dmin·min·Σ(q8)` and accumulates. It is correct but slow.
Your job is to make this dot-product faster **without changing its numerical result**.

## Editable scope (out-of-scope edits fail the task)

You may modify **only**:

```
ggml/src/ggml-cpu/arch/x86/quants.c
```

Any change to a compiled source file outside this scope causes the submission to
score 0. Your edited file is recompiled into `libggml-cpu.a` and linked into the
benchmark on every submission.

## The entry-point contract (must stay behavior-identical)

The exported symbol

```c
void ggml_vec_dot_q5_K_q8_K(int n, float *s, size_t bs,
                            const void *vx, size_t bx,
                            const void *vy, size_t by, int nrc);
```

must keep its signature and its result: given `n` elements laid out as `n/256` consecutive `block_q5_K` (`{ ggml_half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }`)
weight super-blocks and `block_q8_K` (`{ float d; int8_t qs[256]; int16_t bsums[16]; }`) activation super-blocks,
it must write into `*s` the same dot product the scalar reference produces. The
verifier feeds a fixed, deterministic block and requires the exact reference dot
**2304.0**; any implementation that changes this value scores 0.

## What is measured

The verifier recompiles your `quants.c`, links the kernel, and times it on many
random Q5_K×Q8_K blocks against the provided slow scalar baseline. Your kernel must
reproduce the deterministic reference dot above exactly. The CPU exposes the
usual x86 SIMD extensions (SSE4.2 / AVX / AVX2 / FMA / F16C); a vectorized 5-bit K-quant dot — inject the high-bit plane from `qh`, mask the 4-bit low weights into SIMD lanes, unpack the packed 6-bit scales/mins, fold the min correction through the `bsums`, and fuse a multiply-add accumulate recovers
most of the headroom. Headroom grows with block count.

**Final reward.** Let `s` be the median of the paired baseline-time / candidate-time ratios;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

## Rules

- Change only the file(s) in **Editable scope**. Keep the exported symbol and its
  numerical result identical; only the internal implementation may change.
- The starting implementation is correct — do not "fix correctness", make it FAST
  while staying correct.
- Do not special-case the verifier's inputs; the kernel must stay a general
  Q5_K×Q8_K dot product.

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the performance of *this* task as far as you can. You have a
**submission budget of 1 to 16**: you may submit as few as **1** time and at
most **16** (hard ceiling). There is no minimum beyond a single submission and
no requirement to keep going — finalize as soon as you judge your solution good
enough.

**Iterate ONE round at a time, feedback-driven.** Each round is a single cycle:

1. **Edit** `/app/repo/ggml/src/ggml-cpu/arch/x86/quants.c` — apply ONE concrete change based on your current
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
- Do **NOT** route around the task: do not hardcode or special-case the verifier's reference value(s) or its fixed inputs, and do not copy, port, or adapt another quantization kernel's dot-product routine into your file — derive the `Q5_K` 5-bit unpack and per-block dequant yourself so the kernel stays a general `Q5_K×Q8_K` dot product.

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
