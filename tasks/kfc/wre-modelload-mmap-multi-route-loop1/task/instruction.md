# Performance Optimization Task

## What you are given
A clean workspace whose single editable file ships a required function `resolve_tensor_files` that
is **not implemented** (it raises `NotImplementedError`). It is the name-routing step of a
**multi-file memory-mapped weight loader**: several shard files are deserialized in order, each
declaring the tensor names it contains, and when the same tensor name is declared by more than one
file the **last** file to declare it wins. After scanning, tensors are fetched by name and must map
to the file that finally owns them.
**Implement it to the contract below so it is correct, then make it as fast as possible.**
Correctness is a hard prerequisite: a fast-but-wrong result, or a body still raising
`NotImplementedError`, scores 0.

## Editable scope (out-of-scope edits fail the task)
Edit **only**:
```
submission/kernel.py
```

## Entry point and contract
`resolve_tensor_files(decl_name, decl_file, n_names, query) -> numpy.ndarray[int64]`:
- `decl_name`: a 1-D `numpy` integer array of `D` name ids, each in `[0, n_names)`. `decl_name[k]`
  is the tensor name declared by the `k`-th declaration, given in **scan order** (increasing `k`
  means later in the load).
- `decl_file`: a 1-D `numpy` integer array of `D` file ids; `decl_file[k]` is the file that made
  declaration `k`.
- `n_names`: the total number of distinct name ids (name ids run over `[0, n_names)`).
- `query`: a 1-D `numpy` integer array of `Q` name ids to resolve, each in `[0, n_names)` and
  guaranteed to have been declared at least once.
- For each `query[q]`, return the file id of the **last** declaration (the largest `k`) whose
  `decl_name[k] == query[q]`.
- Return a 1-D `numpy` `int64` array of length `Q` (one file id per query), in order.

`custom_kernel(data)` with `data = (decl_name, decl_file, n_names, query)` is already wired to call
`resolve_tensor_files` and return its result. Keep both public signatures.

## Required behavioral property (checked)
The verifier drives `custom_kernel` on hidden `(decl_name, decl_file, n_names, query)` inputs across
a range of declaration counts `D`, name counts, and file counts (including a single file, a single
name, and names declared by several files), and compares the returned file-id array for **exact
equality** against an independent reference implementing the contract. Outputs must genuinely depend
on the input (a constant/cached output is rejected). A submission that leaves the function
`NotImplementedError`, returns the **first** declaring file instead of the last, or otherwise
deviates from last-write-wins fails and scores 0.

## Performance
After correctness passes, the grader times your implementation on the host over many queries
against many declarations. A naive approach
that, for each query, scans the declarations from last to first until it finds a match does work
that grows as `O(Q * D)`; building the name-to-file routing table **once** (a single pass that lets
later declarations overwrite earlier ones) and then resolving all queries against it is `O(D + Q)`
(and runs in compiled code). The gap grows with the number of declarations `D`.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Rules
- Change only `submission/kernel.py`. Implement the complete contract; keep the public signatures.
- Pure Python / NumPy within the scope file; no new third-party dependencies.
- Deterministic given inputs; no reliance on wall-clock, randomness, or external state.
- Do not read, run, reproduce, or infer the scoring/verifier code, the hidden workloads, the
  threshold, the metric, or any reference solution.
- Do not download, clone, fetch, or `pip install` any external repository or reference
  implementation, and do not bypass the environment's network isolation. Any such action scores
  the whole task 0.

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
