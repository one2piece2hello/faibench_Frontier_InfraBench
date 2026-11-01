# Performance Optimization Task

## Overview

You are given a working implementation of a **rowwise int8 quantization** primitive
— the routine that, during low-precision / int8 mixed-precision training, converts a
high-precision 2-D tensor into int8 by computing a per-row symmetric absmax scale and
quantizing each row. The implementation in the declared scope is **functionally
correct but slow**: it quantizes one row at a time.

Your job is to make this subsystem **as fast as possible on the benchmark
workloads**, while preserving its numerical behavior within tolerance.

## Editable scope (you may modify ONLY this file)

```
torchao/prototype/quantized_training/int8.py
```

Edits to any file outside this scope are rejected and score zero. You may
restructure, add helpers, and change the internal algorithm within this file, as
long as the public function keeps its signature and behavior.

## Public entry point + contract

```python
from torchao.prototype.quantized_training.int8 import quantize_int8_rowwise

int8_tensor, scale = quantize_int8_rowwise(tensor, stochastic_rounding=False, eps=1e-12)
```

- `tensor` is a 2-D high-precision tensor. Returns `(int8_tensor, scale)` where
  `scale` is per-row (one entry per leading-dimension index, same dtype as the
  input) and `int8_tensor` is the quantized result in `torch.int8`.
- **Correctness contract:** for the same input, the per-row `scale` and the
  dequantized result (`int8_tensor * scale`) must match an independent reference
  (per-row `scale = |row|.amax()/127`, then `round(row/scale)` clipped to
  `[-128, 127]` as int8) within a relative-norm tolerance (relative max-abs ≤ 2e-2,
  relative L2 ≤ 1e-2), and the integer codes must match the reference exactly
  (no code off by more than 1). A submission that changes the computed result
  beyond tolerance, or produces a result not derived from the actual input, fails
  correctness and scores zero.
- Behavior must be preserved across the workload axes the benchmark exercises:
  **a range of tensor sizes and the (default) round-to-nearest regime.** Do not
  special-case the public workload.
- Determinism: given the same input (and `stochastic_rounding=False`), results must
  be stable across runs.

Helper primitives already present in the surrounding package (outside your scope)
are available for you to call.

## How your work is scored

Your solution is timed end-to-end through the public function on the benchmark workloads.

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

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
