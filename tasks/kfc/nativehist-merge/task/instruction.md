# Performance Optimization Task

You are working on the metric-aggregation path inside an observability platform.
Services emit **exponential-bucket histograms** (a compact, log-scale sketch of a
latency/size distribution: bucket `i` covers the value range `(base**i,
base**(i+1)]` for a base fixed by the histogram's *schema*). Because real
observations cluster, almost every bucket is empty, so each histogram is stored
**sparsely** — only its populated buckets. To build a dashboard or answer a query,
the platform folds ("merges") many per-series histograms into one aggregate. The
file `native_histogram_merge.py` implements that merge as a container,
`NativeHistogramMerger`. It is correct but slow.

## Behavioral contract

A histogram is a `dict` with these keys:

- `"schema"`: an `int` resolution tag. All histograms folded into one merge must
  share the same schema.
- `"zero_count"`: a non-negative `int` — the count of observations in the *zero
  bucket* (values at/near zero, which have no finite log bucket).
- `"sum"`: an `int` aggregate that is merged additively.
- `"buckets"`: a list of `(bucket_index, count)` pairs, **sorted ascending by
  `bucket_index`**, with distinct indices and each `count` a positive `int`.
  `bucket_index` is a signed `int` and may range from far negative to far positive.

Missing `"zero_count"` / `"sum"` / `"buckets"` default to `0` / `0` / `[]`.

```python
class NativeHistogramMerger:
    def __init__(self, schema: int = 0): ...

    def add(self, histogram) -> None:
        """Fold one sparse histogram into the running merge."""

    def merged(self):
        """-> the single merged histogram (a dict with the keys above)."""
```

- **`__init__(schema)`**: `schema` must be an `int` (a `bool` is **not** an int
  here and is rejected). `merged()` on a merger with nothing added returns the
  empty histogram `{"schema": schema, "zero_count": 0, "sum": 0, "buckets": []}`.
- **`add(histogram)`**: fold `histogram` into the merge. Its `"schema"` **must**
  equal the merger's schema; a differing schema is a different bucket geometry and
  cannot be summed bucket-for-bucket, so it is rejected.
- **`merged()`** returns a histogram whose:
  - `"schema"` is the merger's schema;
  - `"zero_count"` is the **sum** of every added histogram's `zero_count`;
  - `"sum"` is the **sum** of every added histogram's `sum`;
  - `"buckets"` is the per-index count total over the **union** of all populated
    bucket indices — for each index present in any added histogram, the sum of its
    counts — emitted **sorted ascending by index**, with any index whose total is
    `0` omitted.

The merge is **associative and commutative**: the add order, and any grouping of
the adds, never changes the result. Total observation count is **conserved**:
`merged.zero_count + sum(count for _, count in merged.buckets)` equals the same
total summed over all inputs.

Error contract: `schema` not an `int` (bools rejected) → `TypeError`; folding a
histogram whose `schema` differs from the merger's → `ValueError`.

Public API (do **not** change the names or the return shape): `NativeHistogramMerger`,
`add(histogram)`, `merged() -> dict`.

## Why the current implementation is slow

The current `merged()` allocates a **dense array spanning the entire index range**
`min..max` of every populated bucket, adds each count into it, then re-sparsifies by
walking that whole range. When the populated buckets are few but spread far apart
(the common case — a wide dynamic range with observations clustered in a handful of
buckets), the dense array is mostly zeros and the re-sparsify walk visits a huge
number of empty cells. The cost grows with the index **range**, not with the number
of populated buckets. Make the merge **faster** — do work that scales with the
populated buckets rather than the index span — while producing the exact same
merged histogram. For example, walk the per-histogram sorted bucket lists together
and combine equal indices, instead of materializing the full dense range.

**Forbidden:** delegating the merge to an exponential / native-histogram library, a
distribution-sketch / quantile-estimation package, or a vectorized numeric-array
library. The scoring harness scans your submitted file for those tokens and scores
the task 0 (do not reference them even in comments). Build the bucket alignment and
the count summation yourself from standard-library primitives.

## Correctness comes first

The verifier compares your merged histogram against an independent reference on many
inputs — overlapping and disjoint bucket sets, a single-histogram merge, the empty
merge and the empty-histogram identity (`empty ⊕ x = x`), all histograms populating
one identical bucket, buckets at very distant indices, a mismatched-schema rejection,
a bad-schema-type rejection, merge-order independence, total-count conservation, a
hidden many-histogram wide-index-range merge, and a work-evidence check that the
zero bucket and `sum` are merged (not dropped) and the per-bucket outputs carry the
summed counts. A faster result that is wrong on even one case scores zero.

## Scope

Optimize the product implementation in `native_histogram_merge.py` only. Do **not**
edit tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares the CPU instructions executed by your implementation with the baseline
instruction count, not elapsed time.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** submission. There is no iterative loop, no dev feedback
round, and no second chance.

1. Read the current implementation in `/app/repo/native_histogram_merge.py`, work out your plan, and
   make your changes.
2. Self-test as much as you like **using your own scratch scripts** (write them
   outside the product file, e.g. under `/tmp`), because you will get no feedback
   from the grader before your score is fixed.
3. When — and only when — you are confident the implementation is both correct and
   faster, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

**Submitting ends the task.** Once the submission is recorded, the harness finalizes
automatically. Later calls do not score new edits, so finish your own checks first.

Your reward comes from a full, trusted end-of-session verifier over the whole
correctness suite plus the performance measurement, so keep your submission
genuinely correct: **any** failing correctness case scores the whole task `0`,
regardless of how fast it is.

**🎓 Explain your work.** Before you edit, state your approach: what in the
current code path is doing redundant work, and what you will change. After you
edit and before you submit, give a short, concrete, step-by-step explanation of
*why* your change is faster in terms of the actual code path, and how you
convinced yourself it is still correct. Write it for a beginner reader.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo`, the sanitized feedback that `submit.sh` returns,
and your own knowledge and reasoning.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Your ONE call to `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — your single `bash /opt/loop/submit.sh` call is the only scoring path.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself; you get no scoring signal
before your single submission — rely on your own reasoning and self-tests.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
