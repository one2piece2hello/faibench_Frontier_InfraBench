# Performance Optimization Task

## Scope

You may modify **only** this file:

```
curriculum_cluster.py
```

Everything else is **out of scope**. Any change to a file outside the scope above
causes the submission to score zero. Find where the slowness is *inside the scope*
by reading and profiling the code — that is part of the task.

## Objective

`curriculum_cluster.py` implements **curriculum-learning difficulty-cluster
selection**, the step that stabilises very-large-scale pretraining by feeding
samples in an easy-to-hard order. Samples are pre-bucketed by a difficulty *metric*
into per-difficulty rows (`index_to_sample[r]` is the array of sample-ids whose
metric value is `index_to_metric[r]`, rows sorted easy→hard). At each curriculum
step the trainer selects the flat set of sample-ids that fall inside a difficulty
window, in one of two modes:

- **VALUE**: every row whose metric value is in `(lo, hi]`;
- **PERCENTILE**: the contiguous slice of samples covering the population fraction in
  `[lo, hi)` of `num_bins` equal-count percentile bands, where the two boundary rows
  are sliced **partially** (a running-count walk takes whole interior rows and the
  correct partial prefix/suffix of the two edge rows), stopping once the end count is
  reached.

The current implementation is **functionally correct but slow**: it walks the rows
in a Python loop and grows the result with a fresh `numpy.concatenate` on every
matching row (each append re-copies the entire running buffer, so the row walk is
quadratic in the number of selected rows), and it re-sums the one-epoch population
from scratch.

Your job: **make `select_curriculum_cluster` faster on the benchmark workload while
returning exactly the same selected sample-ids.** You may reorganize the logic
within the scope file however you like (e.g. a cumulative row-size prefix plus a
boundary search to locate the two edge rows, then a single concatenation of the
interior rows and the two partial edge slices), as long as the observable output
below is preserved.

## Behavioral contract (what the grader checks)

The grader calls the public entry point:

```python
select_curriculum_cluster(index_to_sample, index_to_metric, mode, lo, hi, num_bins=None) -> numpy.ndarray
```

- `index_to_sample`: list of 1-D `numpy` int arrays; `index_to_sample[r]` holds the
  sample-ids at difficulty-row `r`;
- `index_to_metric`: 1-D `numpy` array; `index_to_metric[r]` is the metric value of
  row `r` (non-decreasing in `r`);
- `mode`: `"value"` or `"percentile"`;
- `lo`, `hi`: window bounds — VALUE: exclusive-lower / inclusive-upper metric bounds;
  PERCENTILE: inclusive-start / exclusive-end percentile-band indices in
  `[0, num_bins]` (when `hi == num_bins` the band extends to the exact end of the
  population);
- `num_bins`: number of equal-count percentile bands (PERCENTILE mode only).

The returned 1-D `numpy` int array must equal, **element-for-element and in the same
order** (row-then-within-row), the independent reference the grader computes for both
modes. In PERCENTILE mode the two boundary rows must be sliced at the exact
running-count offsets — a whole-row shortcut that ignores the partial edges selects
the wrong sample-ids and scores zero. The public signature must remain unchanged.

The grader times `select_curriculum_cluster` on the benchmark workload.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Notes

- The workload runs on CPU; no GPU is required. Only `numpy` and the Python
  standard library are available.
- Determinism: given the same inputs, your implementation must produce the same
  output on every run.

## Solve independently — prohibited actions (any one ⇒ the whole task scores 0)

- Reading, printing, copying, `cat`/`grep`/`find`-ing, editing, or reproducing ANY
  verifier / scoring / hidden-test / golden file, wherever it lives; or inferring
  hidden inputs/thresholds.
- Downloading or cloning the upstream project or looking up its reference
  implementation in ANY form — `git clone`/`fetch`/`pull`, adding a git remote,
  `pip download`/`pip install` of the same package, `wget`/`curl` of upstream files,
  checking out a different commit, or web lookup — whether the network appears to work
  or not.
- Bypassing or disabling the network isolation (unsetting/overriding
  `http_proxy`/`https_proxy`/`all_proxy`, opening raw sockets, or any other
  circumvention).

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
