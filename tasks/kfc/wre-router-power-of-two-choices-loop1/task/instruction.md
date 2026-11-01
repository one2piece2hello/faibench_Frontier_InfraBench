# Performance Optimization Task

## What you are given
A clean workspace whose single editable file ships a required function `route_p2c` that is
**not implemented** (it raises `NotImplementedError`). It is the core of a **load-balancing
request router**: a serving system receives a stream of requests and, for each one, is handed two
candidate backend replicas and must send the request to whichever is currently less loaded (the
**"power of two choices"** rule), then account for the added load. **Implement it to the contract
below so it is correct, then make it as fast as possible.** Correctness is a hard prerequisite: a
fast-but-wrong result, or a body still raising `NotImplementedError`, scores 0.

## Editable scope (out-of-scope edits fail the task)
Edit **only**:
```
submission/kernel.py
```

## Entry point and contract
`route_p2c(num_replicas, cand_a, cand_b, init_load) -> (choices, final_load)`:
- `num_replicas`: an `int` `R` — the number of replicas.
- `cand_a`, `cand_b`: 1-D integer `numpy` arrays of length `N` — for request `i` the two candidate
  replica indices are `cand_a[i]` and `cand_b[i]` (each in `[0, R)`).
- `init_load`: a 1-D integer `numpy` array of length `R` — the starting load of each replica.
- Maintain a mutable `load` array initialised to a **copy** of `init_load`. Process the requests
  **in order**. For request `i`, let `a = cand_a[i]`, `b = cand_b[i]`, and choose
  `chosen = a if load[a] <= load[b] else b` (a **tie goes to `a`**). Record `chosen` as the source
  of request `i`, then increment `load[chosen]` by 1 **before** the next request is processed.
- Return a tuple `(choices, final_load)`:
  - `choices`: 1-D `numpy` `int64` array of length `N` — the chosen replica per request, in order.
  - `final_load`: 1-D `numpy` `int64` array of length `R` — the load array after all `N` requests.

`custom_kernel(data)` with `data = (num_replicas, cand_a, cand_b, init_load)` is already wired to
call `route_p2c` and return its result. Keep both public signatures.

## Required behavioral property (checked)
The verifier drives `custom_kernel` on hidden `(num_replicas, cand_a, cand_b, init_load)` inputs
across a range of replica counts `R` and batch sizes `N`, including boundary cases (a single
replica, all-equal starting loads so ties must resolve to `cand_a`, and streams where the two
candidates are identical), and compares BOTH returned arrays (`choices` and `final_load`) for
**exact equality** against an independent reference implementing the contract. Outputs must
genuinely depend on the input (a constant/cached output is rejected). A submission that leaves the
function `NotImplementedError`, routes to the more-loaded replica, mishandles ties, or forgets to
increment the load after each request fails and scores 0.

## Performance
After correctness passes, the grader times your implementation on the host over a large stream
of requests. The routing is **inherently
sequential** — each choice depends on the loads produced by all earlier requests — so there is no
batched shortcut for the decisions themselves. The cost that dominates is the **per-request
interpreter overhead**: a naive loop that reads and updates the load through per-element `numpy`
scalar indexing (`load[a]`, `load[chosen] += 1`, `cand_a[i]`) does far more work per request than
a loop that operates on lightweight native scalars, because each `numpy` 0-d scalar access boxes
and unboxes an array object. The absolute gap grows with the number of requests `N`.

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

## How you are scored (ONE single graded submission)

- You get **exactly one** graded submission. Submit it with
  `bash /opt/loop/submit.sh`.
- **Submitting ends the task.** The moment `submit.sh` returns you are done: stop
  editing and stop working on this task.
- The harness finalizes automatically after the one submission. Calling it again
  does not score new edits.
- You will **not** get iterative feedback to improve against. Nothing is measured
  for you round by round, so there is no measured signal to chase.
- Therefore: read the code, settle the design, and **self-test thoroughly with
  your own scratch scripts** before you submit. Correctness is a hard gate — a
  fast but incorrect submission scores **zero**.
- Explain your approach and your reasoning in writing before you submit.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
