# Performance Optimization Task

You are working on the data-movement front-end of a chunked GPU workload in a
high-throughput inference server. A large tensor lives in **host (CPU) memory**, already
split into `N` chunks (row-blocks). The file `streamed_apply.py` implements
`streamed_chunk_apply` — it moves each chunk from host to device, applies a per-chunk GPU
`compute` op, and concatenates the per-chunk results into one device tensor.

## Behavioral contract

```python
def streamed_chunk_apply(chunks, compute):  # -> torch.Tensor (CUDA)
    ...
```

- `chunks`: a `list`/`tuple` of CPU `torch.Tensor` (host memory). Each chunk is at least
  1-D; its leading dimension is the row/block axis and **may differ across chunks**, while
  every chunk shares the same trailing shape `chunk.shape[1:]` and the same floating dtype.
- `compute`: a callable mapping a CUDA tensor to a CUDA tensor. It is applied **once** to
  the on-device copy of each chunk and preserves the leading (row) dimension:
  `(rows_i, *trailing) -> (rows_i, *out_trailing)`. Treat it as an **opaque** GPU operation —
  invoke it once per chunk; do not inspect, fuse, or rewrite it.

For each chunk `c`, in input order: copy `c` to the GPU, then `out_c = compute(c_on_gpu)`.
The result is `concatenate([out_c for every chunk], dim=0)` — a single CUDA tensor whose
row blocks appear in the **original chunk order**. An empty `chunks` list returns an empty
CUDA tensor.

Public signature (do NOT change): `streamed_chunk_apply(chunks, compute)`.

Error contract:
- `TypeError` if `compute` is not callable, `chunks` is not a list/tuple, a chunk is not a
  `torch.Tensor`, or the chunks do not all share one dtype.
- `ValueError` if a chunk is not in host (CPU) memory, a chunk is 0-D, or the chunks do not
  all share the same trailing shape `chunk.shape[1:]`.

## Why the current implementation is slow

The current implementation moves one chunk to the GPU and only then launches its compute,
one chunk fully after another on a single stream. While a chunk's bytes are streaming in,
the GPU compute units sit **idle**; while a chunk is being computed, the host-to-device
transfer engine sits **idle**. The two phases never run at the same time, so total latency
is the **sum** of every transfer plus every compute — the transfer latency is fully exposed.

Make it **faster on the GPU** by **hiding the transfer latency behind compute**: while the
current chunk is being computed, the next chunk's host-to-device copy can already be in
flight, so total latency approaches `max(total_copy, total_compute)` instead of their sum.
You must build this overlap yourself with explicit CUDA stream and event management, and you
must manage the host/device staging buffers so that a chunk is never read before its copy
has completed and never overwritten before its consumer is done. The output must be
**numerically identical (within tolerance) to the sequential copy-then-compute reference**.

**Allowed / intended tools:** `torch.cuda.Stream`, `torch.cuda.Event`, pinned host memory,
non-blocking (`non_blocking=True`) copies, and explicit synchronization.

**Forbidden:** delegating the pipelining to a framework auto-overlap / graph-capture
convenience — CUDA-graph capture-and-replay of the loop (`torch.cuda.CUDAGraph`,
`torch.cuda.graph`, `torch.cuda.make_graphed_callables`), dataloader-style background
prefetchers, or any helper that auto-pipelines host-to-device copies with compute. The
scoring harness blocks these at runtime and the verifier scans your submitted file for those
tokens (do not reference them even in comments — the scan is textual and scores the task 0).
Build the overlap explicitly with streams and events.

## Correctness comes first

The verifier compares your result against a high-precision **sequential copy-then-compute**
reference on multiple workloads — several multi-chunk shapes, bf16 and fp16, chunks with
varying row counts, a single chunk (no overlap possible), many tiny chunks, a zero-row
chunk, an empty list, error-contract probes, and metamorphic checks (permuting the chunk
list permutes the output blocks; repeated runs are identical) — within a fixed rtol/atol. A
faster result outside tolerance on even one case, or a result whose blocks are out of chunk
order, scores zero. In particular, reading a chunk before its host-to-device copy has
completed (or overwriting a staging buffer still in use) corrupts the result and fails.

## Scope

Optimize the product implementation in `streamed_apply.py` only. Do **not** edit tests,
benchmark harnesses, workloads, or dependency/build files. The final submitted diff must
contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares your GPU execution time with the baseline on the same workloads.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of ``/app/repo/streamed_apply.py``, decide on your
   approach, and make **all** the edits you want in `/app/repo`.
2. Self-test as much as you like with scratch scripts **you** write yourself
   (put them outside the scored file, e.g. under `/tmp`). Verify your output
   against the behavioral contract above on your own inputs, and time your own
   before/after to convince yourself the change is genuinely faster.
3. When — and only when — you are confident, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

**One submission only.** Once the submission is recorded, the harness finalizes it.
A later call does not score new edits. Do your own checks before submitting.

Because of that, everything rides on the work you do **before** you submit:
think the design through, read the baseline carefully, self-test the contract
(including the boundary and error cases described above), and only then submit.
Do not submit a half-finished or untested edit hoping to refine it later — there
is no later.

The grade is produced by a full, trusted end-of-session verifier (more workloads
than anything you can see), so your one submission must be genuinely correct as
well as fast.

**🎓 Explain your work for a beginner.** Before you edit, state your approach:
what makes the current code slow and what you intend to change. After you edit
and before you submit, give a short, concrete, step-by-step account of what you
changed, *why* it should be faster in terms of the actual code path, and how you
convinced yourself it is still correct.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo`, the sanitized feedback that `submit.sh` returns,
and your own knowledge and reasoning.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Running `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — score ONLY by calling `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself; use only
what `submit.sh` returns for scoring signal.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
