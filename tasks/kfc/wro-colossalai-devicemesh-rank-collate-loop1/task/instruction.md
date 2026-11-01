# Performance Optimization Task

## Overview

You are given a working implementation of a **device-mesh process-group planner**. Devices are
arranged on an N-dimensional logical mesh; for a given device the planner returns, **for each mesh
axis, the list of devices that share that axis line** (i.e. the members of that device's
communication group along each axis). The implementation in the declared scope is **functionally
correct but slow**: to turn a mesh coordinate back into a global device id it scans every device in
the mesh, so resolving all groups (done once per device when a mesh is built) is quadratic in the
number of devices.

Your job is to make this subsystem **as fast as possible on the benchmark workloads**, while
preserving its output exactly.

## Editable scope (you may modify ONLY this file)

```
colossalai/device/device_mesh.py
```

Edits to any file outside this scope are rejected and score zero. You may restructure, add
helpers/attributes, and change the internal algorithm within this file, as long as the public
behavior is preserved.

## Public entry point + contract

```python
mesh = DeviceMesh(physical_mesh_id, mesh_shape=shape, device="cpu")
groups = mesh._collate_global_ranks_in_same_process_group(global_rank)
# groups: {axis_index: [global ranks sharing that axis line with `global_rank`]}
```

- The mesh is built from a 1-D `physical_mesh_id` reshaped (row-major) into `mesh_shape`; each
  global rank maps to its per-axis coordinates (`self._global_to_local_rank_mapping`).
- For each mesh axis `dim`, the returned list is every global rank whose coordinates equal
  `global_rank`'s coordinates on all axes **except** `dim`, where the axis-`dim` coordinate ranges
  over `0 … mesh_shape[dim]-1` **in increasing order**.
- **Correctness contract:** for the same mesh, the returned `{axis: [ranks]}` must match the
  reference **exactly** (same axes, same ranks, same order) for every global rank. A submission that
  changes any group, or produces a result not derived from the actual mesh, fails correctness and
  scores zero.
- Behavior must be preserved across the mesh regimes the benchmark exercises (2-D, 3-D, and
  higher-dimensional meshes of varying sizes). Deterministic across runs.

## How your work is scored

Your solution is timed end-to-end on the benchmark workloads.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## 🔴 Rules (violation ⇒ the whole task scores 0)

Solve this task independently, using only the code in the editable scope, the surrounding package
already present in the environment, and your own knowledge.

- Do **NOT** read, `cat`, `grep`, `find`, open, or edit any verifier, scoring, timing, correctness,
  or hidden-test file or directory, wherever it lives; and do not run or reproduce the scoring harness.
- Do **NOT** download, clone, fetch, or otherwise obtain the upstream project or any reference/
  original implementation of this subsystem in ANY form (no `git clone`/`git fetch`/`git pull`, no
  adding a git remote, no `pip install`/`pip download` of the same package, no `wget`/`curl`, no web
  lookup) — whether or not the network appears reachable.
- Do **NOT** bypass or disable the environment's network isolation.

Any one of these actions means the whole task scores 0, regardless of measured performance.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
