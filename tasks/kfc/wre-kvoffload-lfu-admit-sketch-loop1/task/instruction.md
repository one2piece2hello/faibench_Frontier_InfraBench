# Performance Optimization Task

## What you are given
A clean workspace whose single editable file ships a required function `select_offload` that is
**not implemented** (it raises `NotImplementedError`). It is the admission step of a **tiered
KV-cache offload policy** (TinyLFU-style): candidate blocks are promoted to the next tier only if a
frequency sketch says they are "hot" enough and they are not already present downstream. Block
frequencies are tracked in a **count-min sketch** — a `[D, W]` counter table with `D` independent
hash rows. **Implement it to the contract below so it is correct, then make it as fast as
possible.** Correctness is a hard prerequisite: a fast-but-wrong result, or a body still raising
`NotImplementedError`, scores 0.

## Editable scope (out-of-scope edits fail the task)
Edit **only**:
```
submission/kernel.py
```

## Entry point and contract
`select_offload(sketch, seeds, keys, present, threshold) -> numpy.ndarray[int64]`:
- `sketch`: a 2-D `numpy` int64 array of shape `[D, W]` — the count-min counters (`D` hash rows,
  `W` columns).
- `seeds`: a 1-D `numpy` int64 array of length `D` — the per-row hash multipliers.
- `keys`: a 1-D `numpy` int64 array of length `N` — the candidate block hash keys, each in
  `[0, 2**31)`.
- `present`: a 1-D `numpy` int array of length `N` — `present[i] == 1` means block `i` is already
  in the destination tier (must **not** be offloaded); `0` means absent.
- `threshold`: an int — the LFU count threshold.
- For block `i`, its column in row `d` is `col = (keys[i] * seeds[d]) % W` (computed in ordinary
  64-bit integer arithmetic; the given magnitudes never overflow). Its estimated frequency is the
  **minimum** counter across its `D` hashed columns: `est[i] = min over d of sketch[d, col]`. (The
  minimum is the count-min estimate — hash collisions only ever inflate a row, so the smallest row
  is the tightest upper bound.)
- Block `i` is **admitted** iff `est[i] > threshold` **and** `present[i] == 0`.
- Return the indices of all admitted blocks in **ascending index order**, as a 1-D `numpy` int64
  array.

`custom_kernel(data)` with `data = (sketch, seeds, keys, present, threshold)` is already wired to
call `select_offload` and return its result. Keep both public signatures.

## Required behavioral property (checked)
The verifier drives `custom_kernel` on hidden inputs across a range of block counts `N`, sketch
shapes `D`/`W` (including small `W` that forces heavy row collisions so the per-row counters differ,
and `D = 1`), thresholds (including one above all counters -> nothing admitted), and presence
densities (including all-present -> nothing admitted), and compares the returned index array for
**exact equality** against an independent reference implementing the contract. Outputs must
genuinely depend on the input (a constant/cached output is rejected). A submission that leaves the
function `NotImplementedError`, reduces the `D` rows with **max instead of min** (over-estimating
the frequency), forgets the presence exclusion, or returns unsorted/duplicate indices fails and
scores 0.

## Performance
After correctness passes, the grader times your implementation on the host over many blocks
against a fixed sketch. A naive approach loops
over each block in Python and takes the per-row minimum one row at a time, doing work that grows as
`O(N * D)` in the interpreter; hashing all blocks at once and reducing with a single vectorized
gather + min over the row axis is `O(N * D)` in compiled array code. The gap grows with the number
of blocks `N`.

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
