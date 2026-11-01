# Performance Optimization Task

## What you are given
A clean workspace whose single editable file ships a required function `clip_grads_by_global_norm`
that is **not implemented** (it raises `NotImplementedError`). It is the gradient-clipping step of
a training loop: to keep training stable, every gradient is scaled by one shared factor so the
concatenation of all gradients has bounded L2 norm. **Implement it to the contract below so it is
correct, then make it as fast as possible** on the hidden GPU (H20) workloads. Correctness is a
hard prerequisite: a fast-but-wrong result, or a body still raising `NotImplementedError`, scores 0.

## Editable scope (out-of-scope edits fail the task)
Edit **only**:
```
submission/kernel.py
```

## Entry point and contract
`clip_grads_by_global_norm(grads, max_norm) -> (grads, total_norm)`:
- `grads`: a list of `N` CUDA `float32` tensors of arbitrary shapes (the gradients).
- `total_norm = sqrt(sum over all tensors of sum(g*g))` — the global L2 norm over every element of
  every tensor.
- `clip_coef = max_norm / (total_norm + 1e-6)`. If `clip_coef < 1.0`, multiply **every** gradient
  tensor by `clip_coef` (in place is fine); otherwise leave them unchanged.
- Return `(grads, total_norm)`: the (possibly-scaled) list and a 0-dim `float32` CUDA tensor
  holding the **original** global norm (before clipping).

`custom_kernel(data)` with `data = (grads, config)`, `config = {"max_norm": float}`, is already
wired to call `clip_grads_by_global_norm`. Keep both public signatures.

## Required behavioral property (checked)
The verifier drives `custom_kernel` on hidden gradient lists (clip-triggering and no-clip cases,
odd shapes, single tensor) and compares the returned `total_norm` and every scaled gradient against
an independent reference within `rtol=atol=2e-3`. Outputs must genuinely depend on the input. A
submission that leaves the function `NotImplementedError`, clips each tensor by its own local norm
instead of the shared global norm, or reports the post-clip norm fails and scores 0.

## Performance
After correctness passes, the grader measures GPU time on a large list of gradient tensors.
A naive approach loops over the tensors in
Python — one reduction per tensor to accumulate the norm, then one multiply per tensor to scale —
issuing ~2N small kernels; the norm of all tensors and the scaling of all tensors can each be done
in a single fused vectorized call. The gap grows with the number of tensors `N`.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Rules
- Change only `submission/kernel.py`. Implement the complete contract; keep the public signatures.
- Use `torch` (installed); no new third-party dependencies.
- Deterministic given inputs; no reliance on wall-clock, randomness, or external state.
- Do not read, run, reproduce, or infer the scoring/verifier code, the hidden workloads, the
  threshold, the metric, or any reference solution.
- Do not download, clone, fetch, or `pip install` any external repository or reference
  implementation, and do not bypass the environment's network isolation. Any such action scores
  the whole task 0.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
