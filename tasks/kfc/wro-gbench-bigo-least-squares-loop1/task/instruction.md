Optimize the empirical Big-O complexity estimator used by a microbenchmark
harness.

## Context

When a benchmark is run across a sweep of workload sizes, a harness can infer the
empirical asymptotic complexity of the measured code by least-squares fitting the
observed runtimes against a family of candidate complexity curves and reporting the
best-fitting one with its leading coefficient. This mirrors Google Benchmark's
`ComputeBigO` / `MinimalLeastSq` (`src/complexity.cc`).

The scope file `/app/repo/bench_bigo.py` implements `compute_bigo(ns, times)` over
a `(B, K)` runtime table (`B` benchmarks, each measured at the `K` sizes in `ns`).
It is functionally correct but slow: for every `(benchmark, curve)` pair it
recomputes the least-squares sums and the residual RMS in scalar Python loops over
the `K` points. Once `B` grows, that interpreted per-`(benchmark, curve)` fit loop
dominates.

## Task

Make `compute_bigo` fast while keeping its behaviour **exactly**. Only edit
`/app/repo/bench_bigo.py`.

The behavioural contract (checked against an independent reference) — candidate
curves `g(n)` in the fixed order `[ "(1)"=1, "lgN"=log2(n), "N"=n, "NlgN"=n*log2(n),
"N^2"=n^2, "N^3"=n^3 ]`; for each benchmark row `t` and each curve:

  * `coef = sum_i( t_i * g_i ) / sum_i( g_i^2 )`     (least squares through origin)
  * `rms  = sqrt( sum_i( (t_i - coef*g_i)^2 ) / K ) / mean(t)`   (mean-normalized RMS)

The chosen complexity is the curve with the SMALLEST `rms`; ties resolve to the
EARLIER curve in the fixed order (i.e. `"(1)"` is the default, displaced only by a
strictly smaller rms). Return a dict with `"complexity"` (list of `B` label
strings), `"coef"` and `"rms"` (`float64` arrays of shape `(B,)`) for the chosen
curve. Inputs must not be mutated. The expected direction is to batch the six
curve fits across all benchmarks with vectorized array ops.

## Scoring

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

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
