# Performance Optimization Task

## Overview

You are given a working implementation of a **Mixture-of-Experts routing
preprocessing subsystem** — the step that turns a token dispatcher's per-token
expert selections into the dense per-expert data structures the downstream
grouped computation consumes. Two operations run back to back:

1. converting compact per-token expert *indices* (plus their probabilities) into
   a dense multi-hot **routing map** over the local experts, and
2. **padding** that routing map so every expert's token count is rounded up to a
   fixed alignment multiple (as required by low-precision grouped matmuls).

The implementation in the declared scope is **functionally correct but slow**: it
computes each operation with a sequence of general-purpose eager tensor ops and a
data-dependent gather.

Your job is to make this subsystem **as fast as possible on the benchmark
workloads**, while preserving its exact behavior.

## Editable scope (you may modify ONLY these files)

```
megatron/core/fusions/fused_pad_routing_map.py
megatron/core/fusions/fused_indices_converter.py
```

Edits to any file outside this scope are rejected and score zero. You are free to
restructure, add helper functions/kernels, and change the internal algorithm
within these two files, as long as the public entry points keep their signatures
and behavior.

## Public entry points + contract

The subsystem is driven only through these public functions:

```python
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

# indices: [num_tokens, topk] int64 expert ids; -1 marks a dropped slot.
# probs:   [num_tokens, topk] float32 per-slot probabilities.
routing_map, probs_in_multihot = fused_indices_to_multihot(indices, probs, num_local_experts)
#   routing_map:       [num_tokens, num_local_experts] (bool) multi-hot map
#   probs_in_multihot: [num_tokens, num_local_experts] (probs dtype)

padded_map = fused_pad_routing_map(routing_map, pad_multiple)
#   padded_map: [num_tokens, num_experts] integer map, each expert's token
#               count rounded UP to a multiple of `pad_multiple` by flipping the
#               earliest currently-unrouted token slots of that expert to 1.
```

- **Correctness contract:** for the same inputs, the multi-hot routing map and the
  padded map must be **exactly equal** (as integers) to the reference results, and
  `probs_in_multihot` must match the reference to within a tiny floating-point
  tolerance (the probabilities are only gathered, never recombined). A submission
  that changes the computed result, or that produces a result not derived from the
  actual inputs, fails correctness and scores zero.
- **Semantics to preserve exactly:** an index of `-1` (or any index outside the
  local-expert range) contributes nothing; padding only ever converts unrouted
  (zero) slots to routed (one), never the reverse, and rounds each expert up to
  the next multiple of `pad_multiple`.
- Behavior must be preserved across the workload axes the benchmark exercises:
  **varying token counts, varying local-expert counts, varying top-k widths,
  different alignment multiples, and inputs with and without dropped slots.** Do
  not special-case the public workload; hidden workloads probe other regimes.
- Determinism: given the same inputs, results must be stable across runs.

Helper primitives already present in the surrounding package (outside your scope)
are available for you to call.

## How your work is scored

Your solution is timed end-to-end through the public entry points on the benchmark workloads.

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
