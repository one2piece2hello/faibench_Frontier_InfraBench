# Task: speed up multi-scale retention (RetNet) chunked linear attention in `fla-org/flash-linear-attention`

This is a scoped performance task in a real subsystem of **fla-org/flash-linear-attention**
(a flagship open-source LLM-infra repository, >2000 GitHub stars). The subsystem implements
the forward path of **multi-scale retention (RetNet)** — a linear-attention operator with a
fixed, data-independent per-head exponential decay.

In the working tree, the in-scope implementation is a **correct but slow** eager path: it
evaluates the retention state recurrence strictly one time step at a time (an `O(T)` sequential
Python loop over the sequence, with no time-parallel / chunked reformulation). Your job is to
make it **fast** while keeping it numerically correct.

## Objective
Make the forward pass of the retention subsystem as fast as possible on the benchmark workload,
without changing its numerical result (within tolerance) or its public contract. A restored,
production-grade chunked/blocked kernel is dramatically faster than the shipped per-timestep
scan. To earn a positive reward, your implementation must be correct and faster than the
calibrated reference, not just faster than the shipped slow baseline.

## Editable scope (out-of-scope edits are rejected)
```
fla/ops/retention/chunk.py
```
Only this file may be modified. It exposes the public entry point:

```python
def chunk_retention(
    q, k, v,                       # q,k: [B, T, H, K];  v: [B, T, H, V]  (torch.Tensor)
    scale=None,                    # float; default K ** -0.5
    initial_state=None,            # [N, H, K, V] or None
    output_final_state=False,      # bool
    cu_seqlens=None,               # [N+1] LongTensor for varlen, or None
    **kwargs,
) -> tuple[o, final_state]         # o: [B, T, H, V];  final_state: [N,H,K,V] or None
```

Operator definition (this is what the grader checks against, computed independently):
```
gamma[h] = 1 - 2 ** (-5 - h)                      # fixed per-head decay, h = 0..H-1
o_t      = scale * sum_{s <= t} gamma[h]^(t-s) (q_t . k_s) v_s
```
i.e. the causal decay-weighted linear-attention output; equivalently the state recurrence
`S_t = gamma[h] * S_{t-1} + k_t^T v_t`, `o_t = scale * (q_t @ S_t)`. In the scored workload
`initial_state=None`, `output_final_state=False`, `cu_seqlens=None`.

## What is provided
- `/app/repo` — the repository tree at the frozen baseline, importable at that path
  (`import fla`). The full package (utility kernels, layers, other operators) is available for
  you to build on; only `fla/ops/retention/chunk.py` is editable.
- A public smoke you can run: `bash tests/test.sh` (drives the scope and prints a JSON verdict).
  The hidden grading harness uses the same contract with held-out inputs.

## How you are graded
- **Correctness gate**: your `chunk_retention` output must match the independent fp32 retention
  reference on randomized inputs within tolerance (`rel_max <= 3e-2`, `rel_l2 <= 1.5e-2`).
- **Speed**: `s = baseline_ms / candidate_ms` is your raw speedup; `r` is the reference
  implementation's speedup over the same baseline. A failed validity check, `r <= 1`, or
  `s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`,
  capped at 1 when `s >= r²`.

## Grading integrity (submissions violating this score 0 — academic-integrity clause)
- The grader, its reference outputs, and all held-out test inputs live outside your workspace.
  Do not read, `cat`, `grep`, import, reconstruct, or modify the grader or any scoring asset,
  and do not reproduce the scoring harness.
- Do **not** fetch the upstream repository, its git history, PRs, or any external reference
  implementation of this subsystem (no `git clone/fetch/pull`, adding remotes, `pip download/install`,
  `wget`/`curl`, or web lookup).
- Only the file in the declared editable scope may be modified; a submission that changes any
  out-of-scope file is marked invalid.
- Do not attempt to reach the network or bypass the sandbox isolation.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
