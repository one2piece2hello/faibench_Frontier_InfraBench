# Performance Optimization Task

## Overview

You are given a working implementation of an **adaptive-moment first-order
optimizer** (a stochastic-gradient method that maintains, per parameter, a
running average of the gradient and an additive sign-based running estimate of
its second moment, and applies a bias-corrected, denominator-normalized update).
The implementation in the declared scope is **functionally correct but slow**: it
updates the model's parameters one parameter tensor at a time in a Python loop.

Your job is to make this subsystem **as fast as possible on the benchmark
workloads**, while preserving its numerical behavior within tolerance.

## Editable scope (you may modify ONLY this file)

```
torch_optimizer/yogi.py
```

Edits to any file outside this scope are rejected and score zero. You are free to
restructure, add helper functions, and change the internal algorithm within this
file, as long as the public optimizer class keeps its constructor signature,
its `.step()` semantics, and its per-parameter state.

## Public entry point + contract

The subsystem is driven only through the public optimizer class

```python
from torch_optimizer import Yogi

opt = Yogi(params, lr=1e-2, betas=(0.9, 0.999), eps=1e-3,
           initial_accumulator=1e-6, weight_decay=0.0)
# ... set p.grad for each parameter ...
opt.step()
```

- `Yogi(params, ...)` accepts an iterable of parameter tensors (or param-group
  dicts) and the listed hyperparameters. `.step()` updates every parameter in
  place from its `.grad`, maintaining two per-parameter running buffers
  (first-moment `exp_avg` and second-moment `exp_avg_sq`), both initialized to
  `initial_accumulator`.
- **Correctness contract:** after any number of `.step()` calls, the parameters
  must match a reference optimizer that applies the SAME update rule
  (bias-corrected first moment, additive sign-based second-moment update, and a
  `lr/bias_correction1 * exp_avg / (sqrt(exp_avg_sq)/sqrt(bias_correction2) +
  eps)` step) independently per parameter, within a relative-norm tolerance. A
  submission that changes the computed parameters beyond tolerance fails
  correctness and scores zero.
- Behavior must be preserved across the workload axes the benchmark exercises:
  **many small parameters, few large parameters, and multiple `.step()` calls
  (state carried across steps).** Do not special-case the public workload;
  hidden workloads probe other shapes.
- Determinism: given the same inputs, results must be stable across runs.

## How your work is scored

Your solution is timed end-to-end through the public entry point on the benchmark workloads.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## 🔴 Rules (violation ⇒ the whole task scores 0)

Solve this task independently, using only the code in the editable scope, the
surrounding package already present in the environment, and your own knowledge.

- Do **NOT** read, `cat`, `grep`, `find`, open, or edit any verifier, scoring,
  timing, correctness, or hidden-test file or directory, wherever it lives; and
  do not run or attempt to reproduce/reverse-engineer the scoring harness.
- Do **NOT** download, clone, fetch, or otherwise obtain the upstream project or
  any reference/original implementation of this subsystem in ANY form — no
  `git clone`/`git fetch`/`git pull`, no adding a git remote, no
  `pip install`/`pip download` of the same package, no `wget`/`curl` of upstream
  files, and no web lookup — whether or not the network appears reachable.
- Do **NOT** bypass or disable the environment's network isolation (e.g.
  unsetting or repointing `http_proxy`/`https_proxy`/`all_proxy`, opening raw
  sockets, or any other circumvention).

Any one of these actions means the whole task scores 0, regardless of measured
performance.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
