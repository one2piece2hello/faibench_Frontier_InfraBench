# Performance Optimization Task

## Overview
You are given a working implementation of a **Mixture-of-Experts routing-map
padding** utility. Given a token-to-expert routing map, it pads each expert's
assigned-token count up to a multiple of a given alignment (needed for aligned
expert GEMMs), by converting the earliest unassigned entries to assigned. The
implementation in the declared scope is **functionally correct but slow**: it
scans the routing map one entry at a time in a Python loop.

Make this subsystem **as fast as possible on the benchmark workloads** while
preserving its exact output.

## Editable scope (you may modify ONLY this file)
```
megatron/core/transformer/moe/moe_utils.py
```
Edits to any file outside this scope are rejected and score zero. Focus on the
`pad_routing_map` function; keep its signature and observable behavior.

## Public entry point + contract
```python
padded = pad_routing_map(routing_map, pad_multiple)
```
- `routing_map`: bool tensor `[num_tokens, num_experts]` (True = token routed to
  expert). `pad_multiple`: int. Returns the padded routing map, same shape.
- **Correctness contract:** for each expert column, the earliest zero entries are
  converted to ones until that column's count of ones is a multiple of
  `pad_multiple`; already-aligned columns are unchanged; assigned entries are never
  removed. The output must exactly equal the reference (bitwise on the boolean map).
- Behavior must hold across token counts, expert counts, and `pad_multiple` values.

## How your work is scored
The grader times the public function end-to-end on the benchmark workloads.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## 🔴 Rules (violation ⇒ the whole task scores 0)
- Do **NOT** read, cat, grep, open, or edit any verifier/scoring/timing/hidden-test
  file, and do not reproduce/reverse-engineer the scoring harness.
- Do **NOT** download, clone, fetch, or otherwise obtain the upstream project or any
  reference implementation of this subsystem (no git clone/fetch/pull, remotes,
  pip install/download of the package, wget/curl, or web lookup), whether or not the
  network appears reachable.
- Do **NOT** bypass or disable the environment's network isolation.

## How you are scored (ONE single graded submission)

- You get **exactly one** graded submission. Submit it with
  `bash /opt/loop/submit.sh`.
- **Submitting ends the task.** The moment `submit.sh` returns you are done: stop
  editing and stop working on this task.
- A second call to `submit.sh` is **refused** and exits non-zero. There is no
  retry, no budget of attempts, no "best of several submissions", and no separate
  finalization step — your one submission *is* the final answer.
- You will **not** get iterative feedback to improve against. Nothing is measured
  for you round by round, so there is no measured signal to chase.
- Whatever state the in-scope file is in when you submit is exactly what gets
  graded. Nothing is restored, re-selected or rolled back for you.
- Therefore: read the code, settle the design, and **self-test thoroughly with
  your own scratch scripts** before you submit. Correctness is a hard gate — a
  fast but incorrect submission scores **zero**.
- Explain your approach and your reasoning in writing before you submit.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
