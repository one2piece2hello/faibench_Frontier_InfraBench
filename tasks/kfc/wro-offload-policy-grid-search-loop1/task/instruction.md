# Performance Optimization Task

A single-GPU **heterogeneous-offload** inference runtime in this repository does not fit its
model in GPU memory, so it *tiers* every tensor family across **GPU / CPU DRAM / NVMe**. Before
a run it must pick the offload policy: enumerate a simplex grid of placement fractions, score
each candidate with an analytic cost model, drop the ones that blow the memory budgets, and
keep the fastest survivor. That policy search is implemented in this module.

It is functionally correct but **slow**: the grid is materialised by a Python triple product
that appends one tuple per candidate, and then *every* scoring stage walks the `(N, 6)` array
**row by row**, pulling six cells out of the array with `p[i, k]` per row — the memory model,
the latency cost model, the feasibility test and the arg-min are four separate Python loops
over the same hundreds of thousands of rows. Make it **faster** on the benchmark workload while
**preserving its output exactly** — bit-for-bit, including every float.

## Editable scope

Edit **only** this file (any edit outside this scope scores the whole task zero):

```
offload_policy.py
```

## The subsystem

A policy is six fractions — `wg, wc` for the **weights**, `cg, cc` for the **KV cache**,
`hg, hc` for the **activations** — where `g` is the fraction resident on GPU, `c` the fraction
in CPU DRAM, and the remainder `1 - g - c` lives on disk. Five entry points, all exercised
(`search_best_policy` chains the other four, and each of the four is also graded on its own):

1. `enumerate_grid(steps)` — the `(N, 6)` float64 candidate array. Per family the admissible
   pairs are, **in this order**, `for gi in range(steps+1): for ci in range(steps+1-gi):
   (gi/steps, ci/steps)`, so `P = (steps+1)*(steps+2)//2` pairs; the full grid is the
   lexicographic product over **(weights, kv, activations)**, `N = P**3` rows, column order
   exactly `(wg, wc, cg, cc, hg, hc)`.
2. `policy_peak_memory(specs, pol)` — `(gpu_bytes, cpu_bytes)`, each shaped like `pol[:, 0]`:
   `gpu = n_layers * (wg*W + cg*C + hg*A) + gpu_working_set_bytes` and
   `cpu = n_layers * (wc*W + cc*C + hc*A) + pinned_buf_bytes`.
3. `policy_latency(specs, pol)` — per layer the GPU compute time is
   `t_c = flops_per_layer / (gpu_tflops * 1e12)`; the **non-resident** fractions must be
   fetched, over PCIe from DRAM and over the NVMe link from disk, so
   `t_w = wc*W/pcie + (1-wg-wc)*W/disk` (and likewise `t_k`, `t_a`). Transfers overlap with
   compute: `t_layer = max(t_c, t_w + t_k + t_a)`, total `n_layers * t_layer +
   prefill_overhead_s`.
4. `feasible_mask(specs, pol, budgets)` — `gpu_bytes <= budgets[0]` **and**
   `cpu_bytes <= budgets[1]`.
5. `search_best_policy(specs, budgets, steps)` — the cheapest **feasible** candidate. Returns
   `best_index`, `best_policy`, `best_latency`, `gpu_bytes`, `cpu_bytes`, `n_feasible`,
   `n_candidates`.

Contract notes that are easy to break:

* The cost model is compared for **exact float equality**. Keep the operation order and the
  association of every expression: `a * b / c` is not `a * (b / c)`, and
  `1.0 - wg - wc` is not `1.0 - (wg + wc)`.
* `w_disk` is `1.0 - wg - wc`: the **CPU tier counts**. Dropping it (`1.0 - wg`) charges the
  DRAM-resident bytes to the disk link and is wrong.
* The GPU peak includes `gpu_working_set_bytes` and the CPU peak includes `pinned_buf_bytes`.
* Ties in the arg-min go to the **lowest index** (strict `<` while scanning forward).
* When nothing is feasible: `best_index = -1`, `best_policy = []`, and `best_latency`,
  `gpu_bytes`, `cpu_bytes` all `-1.0`, `n_feasible = 0` — but `n_candidates` is still the real
  grid size.
* `best_policy` is a list of plain Python `float`s; the scalars are Python `float`/`int`.

## Constraints

* Pure Python + NumPy. No new dependencies, no C extensions, no subprocesses, no threads.
* Do not weaken, special-case or precompute for the benchmark inputs: the verifier runs a
  separate correctness suite of 22 hardware/model/budget scenarios (`steps` 1 to 7, tight and
  loose budgets, infeasible budgets, zero activations, one layer, exact-budget boundaries,
  disk == PCIe) against an independent reference, and any mismatch scores **zero**.
* Do not touch the verifier, the tests directory, or anything outside `offload_policy.py`.

## How you are scored

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.
It is **gated on exact correctness** — a single mismatched float scores 0 regardless of speed. The benchmark runs one full policy search on a
`steps=10` grid (287496 candidates); there is a large amount of headroom, and the headroom
**grows with the grid size**, so an algorithmically better solution wins by much more than a
micro-optimized one.

## Where to start

Read the module docstring, then each function's docstring: they pin the exact contract. Then
ask, for each Python loop, what it is really computing over the whole `(N, 6)` array — a
lexicographic product of a triangular index set, an elementwise linear form, an elementwise
cost model, a boolean conjunction, an arg-min over a masked subset.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
