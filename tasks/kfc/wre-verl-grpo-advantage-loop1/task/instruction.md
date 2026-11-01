# Functional Implementation Task

## Objective

Implement a **group-relative advantage estimation** subsystem used in reinforcement-learning
post-training, preserving its numerical behavior within tolerance.

The subsystem takes per-sample scalar rewards that are partitioned into **groups** (multiple
sampled responses share one prompt/group id) and produces, for each sample, an advantage that
is normalized *relative to its own group*. Two estimators are required (GRPO and RLOO) plus the
shared group-statistics primitive they build on.

The starting code raises `NotImplementedError` in every function by design. Implement each one
to the stated contract. A submission that
leaves any function unimplemented, or whose outputs do not match the reference within tolerance,
scores 0.

## Editable scope

You may edit ONLY:

```
submission/advantage_estimators.py
```

Any change outside this file is ignored/invalid. You may add private helper functions inside
this file. You may use `torch` and `numpy`. Do **not** import a third-party package that already
provides these estimators — the implementation must be your own within this file.

## Interface contract (implement exactly these signatures)

All tensors are on CPU, float32 unless noted. `B` = batch size, `L` = response length,
`G` = number of distinct groups.

### `as_torch_index(index, device=None) -> torch.LongTensor`
- `index`: a length-`B` sequence / `np.ndarray` / `torch.Tensor` of group labels. Integer labels
  are used directly (cast to `long`); non-integer/arbitrary labels are mapped to contiguous ids
  preserving first-appearance order.
- Returns a 1-D `torch.long` tensor of shape `(B,)` with values in `[0, G-1]`.

### `group_mean_std(scores, gidx, eps=1e-6, device=None) -> (mean_g, std_g, count_g)`
- `scores`: `(N,)` float tensor of per-sample scalars. `gidx`: `(N,)` integer tensor of group
  ids in `[0, G-1]`.
- Returns three `(G,)` float32 tensors: per-group mean, std, and count, where `G = max(gidx)+1`.
- `std` uses **Bessel correction**: denominator `= max(count-1, 1)`, and `std = sqrt(max(var, eps))`.
- A **singleton** group (count == 1) returns `mean = 0.0` and `std = 1.0`.
- Empty input (`N == 0`) returns three length-0 tensors.

### `compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, epsilon=1e-6, norm_adv_by_std_in_grpo=True, config=None) -> (advantages, returns)`
- `token_level_rewards`: `(B, L)` float. The per-sample scalar **score** is the sum over the
  last (length) dimension.
- `response_mask`: `(B, L)` float. The scalar advantage is broadcast over `L` and multiplied by
  this mask.
- `index`: length-`B` group labels (see `as_torch_index`).
- If `norm_adv_by_std_in_grpo` is `True`: `advantage_i = (score_i - group_mean) / (group_std + epsilon)`.
  If `False`: `advantage_i = score_i - group_mean`.
- Singleton groups use `mean = 0`, `std = 1`.
- Returns `(advantages, returns)`, both `(B, L)` float tensors that are **equal to each other**.
- `config` is an unused placeholder — accept and ignore it.

### `compute_rloo_outcome_advantage(token_level_rewards, response_mask, index, epsilon=1e-6, config=None, **kwargs) -> (advantages, returns)`
- Same input shapes and score/mask conventions as above.
- For a group of size `n > 1`, the per-sample advantage is the leave-one-out baseline:
  `score_i * n/(n-1) - group_mean * n/(n-1)`. A **singleton** group (`n == 1`) uses
  `group_mean = 0` and scale factor `1`, so its advantage is its **own raw score** `score_i`
  (**not** `0`) — matching the naive per-element leave-one-out baseline (empty complement ⇒ baseline 0).
- Returns `(advantages, returns)`, both `(B, L)` and equal.
- `epsilon` is accepted for signature compatibility (unused for RLOO).

## Correctness

Your outputs are compared to a held-out reference on hidden workloads with tolerance
`atol = 2e-4`, `rtol = 1e-3` (float32). Workloads span even and uneven group sizes, singleton
groups, a single-group batch, sparse rewards, and large batches with many groups. Behavior must
be deterministic given the inputs.

## Scoring

The final reward is binary: **1** if every hidden case passes and no hard check fails, **0**
otherwise. Timing is recorded, but raw speedup does not earn partial or extra credit.

## Rules

Solve this yourself, using only the resources already present in this environment. You must not:

- read, `cat`, `grep`, `find`, edit, or reproduce any grading / scoring / hidden-test file, or
  print/infer hidden workloads, seeds, thresholds, or the reference timing;
- download or clone any external repository in any form — `git clone`/`fetch`/`pull`, adding a
  git remote, `pip download`/`pip install` of a package that provides these estimators,
  `wget`/`curl`/web lookup of the original source — whether the internet appears to work or not;
- bypass or disable the network isolation (unsetting/overriding `http_proxy`/`https_proxy`/
  `all_proxy` and their uppercase forms, pointing them elsewhere, or opening raw sockets).

Doing any of these means the whole task scores 0.

## 提交方式

本题为**实现类,只有一次评分**。把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `bash /opt/loop/submit.sh`,也不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上;评分由结束后的 `tests/test.sh` 一次性给出)。
