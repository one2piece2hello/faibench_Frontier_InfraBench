# Performance Optimization Task

## What you are given
A clean workspace whose single editable file ships a required function `fair_interleave_order`
that is **not implemented** (it raises `NotImplementedError`). It is the **fair scheduling** step
of a multi-tenant serving scheduler (VTC-style fairness: a bursty tenant must not starve its
neighbours): instead of serving the waiting queue in pure arrival order, the scheduler interleaves
tenants round-robin — every tenant's 1st request, then every tenant's 2nd, and so on. **Implement
it to the contract below so it is correct, then make it as fast as possible.** Correctness is a
hard prerequisite: a fast-but-wrong result, or a body still raising `NotImplementedError`,
scores 0.

## Editable scope (out-of-scope edits fail the task)
Edit **only**:
```
submission/kernel.py
```

## Entry point and contract
`fair_interleave_order(tenant_ids, num_tenants) -> numpy.ndarray[int64]`:
- `tenant_ids`: 1-D sequence of `N` integers in `[0, num_tenants)` — the owning tenant of each
  waiting request, given in **arrival order** (index = arrival position).
- `num_tenants`: positive `int` — the number of tenants (ids `0 .. num_tenants-1`).
- For each request compute its **within-tenant round** `r` = the number of EARLIER requests
  (smaller original index) that share its tenant (so a tenant's own requests get rounds
  `0, 1, 2, …` in arrival order).
- The fair schedule orders all requests by `(round asc, tenant_id asc)`. Within a single
  `(round, tenant)` there is at most one request, so this is a total order.
- Return a 1-D `numpy` `int64` array `sched_pos` of length `N`: entry `i` is the **0-based
  position** of request `i` in that fair round-robin schedule.

`custom_kernel(data)` with `data = (tenant_ids, num_tenants)` is already wired to call
`fair_interleave_order` and return its result. Keep both public signatures.

## Required behavioral property (checked)
The verifier drives `custom_kernel` on hidden inputs across a range of queue sizes `N` and tenant
counts, including uniform tenant mixes, a single active tenant (identity schedule), heavily skewed
mixes where one dominant tenant submits a burst (fairness must not let it monopolise the head of
the schedule), and sparse tenant coverage (only some ids present), and compares the returned
position array for **exact equality** against an independent reference implementing the contract.
Outputs must genuinely depend on the input (a constant/cached output is rejected). A submission
that leaves the function `NotImplementedError`, orders by tenant first (grouping a tenant's whole
burst together) instead of by round first, or miscomputes the within-tenant round fails and
scores 0.

## Performance
After correctness passes, the grader times your implementation on the host over a **large**
balanced multi-tenant queue. A naive approach
that groups requests into per-tenant Python lists and then interleaves them round-robin in a nested
Python loop does that work in the interpreter; computing every request's within-tenant round with a
vectorized group-by (a stable sort plus per-group offsets from bin counts), forming a single
composite `(round, tenant)` sort key, and scattering positions with one argsort runs in compiled
code and is far faster. The gap grows with the queue length `N`.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Rules
- Change only `submission/kernel.py`. Implement the complete contract; keep the public signatures.
- Pure Python / NumPy within the scope file; no new third-party dependencies.
- Deterministic given inputs; no reliance on wall-clock, randomness, or external state.
- Do not read, run, reproduce, or infer the scoring/verifier code, the hidden workloads, the
  threshold, the metric, or any reference solution.
- Do not download, clone, fetch, or `pip install` any external repository or reference
  implementation, and do not bypass the environment's network isolation. Any such action scores
  the whole task 0.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
