# Performance Optimization Task

## Scope

You may modify **only** this file:

```
guide_transition.py
```

Everything else is **out of scope**. Any change to a file outside the scope above
causes the submission to score zero. Find where the slowness is *inside the scope*
by reading and profiling the code — that is part of the task.

## Objective

`guide_transition.py` implements the **batched guide-state advance** step of
regex-constrained (guided) decoding. At every decode step the runtime holds one FSM/DFA
guide state per active sequence in the batch; after the sampler picks a token for each
sequence, the guide must advance every sequence's state along the transition its sampled
token induces, and it also needs the advanced state's out-degree (how many tokens are
still valid) to flag sequences that have reached a dead/terminal state.

The current implementation is **functionally correct but slow**: it never builds an
index. For **each** query it linear-scans the entire transition edge list to find the
matching `(state, token)` transition, then linear-scans the entire edge list **again** to
count the out-degree of the landed state. The cost is O(#queries × #edges).

Your job: **make `batch_advance` faster on the benchmark workload while producing the
same result.** You may reorganize the logic within the scope file however you like
(e.g. index the edge list once, then answer each query in O(1)), as long as the
observable output below is preserved.

## Behavioral contract (what the grader checks)

The grader calls the public entry point:

```python
batch_advance(num_states, edges, queries) -> dict
```

- `num_states`: number of DFA states (ids `0..num_states-1`);
- `edges`: the transition edge list — `(state, token_key, next_state)` triples of a
  deterministic automaton (at most one edge per `(state, token_key)`);
- `queries`: the batch — `(current_state, token_key)` pairs, one per active sequence.

The returned `dict` must contain exactly these keys, each equal to the independent
reference the grader computes:

1. `"next_states"`: `list[int]` — for each query, the advanced state, or `-1` if the
   token is not accepted from the current state (the sequence dies);
2. `"out_degrees"`: `list[int]` — for each query, the number of outgoing edges from the
   corresponding `next_states` entry (`0` for a dead query).

Both lists must match the reference exactly, in query order. A shortcut that advances by
the current state alone (ignoring the sampled token) is wrong whenever a state has edges
on multiple tokens and scores zero. The public signature must remain unchanged.

The grader times `batch_advance` on the benchmark workload.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Notes

- The workload runs on CPU; no GPU is required. Only `numpy` and the Python standard
  library are available (the scope itself is pure Python).
- Determinism: given the same inputs, your implementation must produce the same output
  on every run.

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
