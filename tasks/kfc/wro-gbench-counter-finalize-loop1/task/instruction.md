Optimize the per-counter finalization used by a microbenchmark harness.

## Context

A microbenchmark records, for every benchmark case, a set of user counters. Each
counter carries a raw accumulated value plus a bitmask of flags describing how the
raw value must be finalized once the run's `iterations`, `cpu_time` (seconds) and
`num_threads` are known. This mirrors Google Benchmark's `Finish`
(`src/counter.cc`) together with the flag semantics in
`include/benchmark/counter.h`.

The scope file `/app/repo/bench_counter.py` implements
`finalize_counters(values, flags, iterations, cpu_time, num_threads)` over a whole
run's `(B, C)` counter table (`B` benchmarks, `C` counters). It is functionally
correct but slow: it walks every `(benchmark, counter)` cell in a Python loop,
reads each raw value out of the numpy table one element at a time, and applies the
five flag transforms with scalar Python branches. Once `B * C` grows, that
interpreted per-cell finalization dominates.

## Task

Make `finalize_counters` fast while keeping its behaviour **exactly**. Only edit
`/app/repo/bench_counter.py`.

The behavioural contract (checked against an independent reference) — each
counter's raw value `v` is transformed in this FIXED order (per-benchmark
`cpu_time`, `num_threads`, `iterations` broadcast across that benchmark's
counters), returning a `float64` array of shape `(B, C)`:

  * `kIsRate` (`1 << 0`):               `v /= cpu_time`
  * `kAvgThreads` (`1 << 1`):           `v /= num_threads`
  * `kIsIterationInvariant` (`1 << 2`): `v *= iterations`
  * `kAvgIterations` (`1 << 3`):        `v /= iterations`
  * `kInvert` (`1 << 31`):              `v = 1.0 / v`  — applied **last**, always.

Inputs must not be mutated. The expected direction is to replace the per-cell
Python loop with vectorized array operations — boolean masks built from the flag
bits plus `numpy.where` for each transform, with the per-benchmark scalars
broadcast across the counter axis. The order of the transforms (in particular that
the inversion is applied last) must be preserved exactly.

## How you are scored

You get **exactly one** graded submission (`bash /opt/loop/submit.sh`) — submitting ends the task
and a second call does not score new edits. A single mismatched finalized value scores 0 regardless of speed.
Think the change through and self-test before you submit; you cannot iterate.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Anti-cheat red line (A-only)

Reading or tampering with the verifier/scoring assets, reproducing or importing
the hidden harness, fetching the upstream repository/reference implementation, or
bypassing the runtime network isolation — any of these makes the WHOLE task score
zero. Solve it by editing the scope file only.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
