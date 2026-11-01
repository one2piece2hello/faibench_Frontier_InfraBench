# Performance Optimization Task

You are working on the gradient-exchange path of a data-parallel training system
that runs over a **bandwidth-constrained** network. Every step, each worker has to
transmit its gradient buffer to the others. The file `grad_compress.py` implements
`compress` / `decompress` — it currently sends the whole buffer at full precision,
which saturates the link. Your job is to move a **much smaller payload per step**
while keeping training unbiased over time.

## Behavioral contract

Two public functions (do **not** change their names or signatures):

```python
def compress(buf, residual):
    # buf:      gradient tensor, any shape, dtype torch.float32, CUDA tensor
    # residual: persistent accumulator, torch.float32, SAME numel as buf
    ...  # -> (payload, new_residual)

def decompress(payload):
    ...  # -> torch.Tensor, shape == buf.shape, dtype float32
```

The value actually compressed on a step is the buffer **compensated** by the
accumulator — their sum — which is the standard **error-feedback** scheme. Its compact
representation is a **scaled 1-bit (sign) code**, defined as follows:

1. The flattened compensated buffer is split into contiguous **blocks** of a fixed size
   `B` (`B = 2048`; the final block may be shorter), and each block gets one scalar
   **scale**: the **root-mean-square** of that block's elements (its L2 norm divided by
   the square root of its length).
2. Every element contributes only its **sign**, mapped to `{-1, +1}`, with `sign(0)`
   taken as `+1`.
3. The reconstruction of a block is its scale applied to those signs — one magnitude
   for the whole block, the sign per element. `decompress` returns that reconstruction
   reshaped to `buf.shape`.
4. The signs are stored **packed to one bit each** (8 signs per byte) — this is what
   makes the payload small. `payload` must carry the per-block `scale`, the packed sign
   bits, the element count, and the original shape (and nothing that grows with the
   buffer at full precision).
5. The accumulator is updated to hold exactly the part of the compensated buffer the
   reconstruction did **not** represent (reshaped to `buf.shape`). This carries the
   un-transmitted residue into the next step and keeps the sequence unbiased.

`compress` returns `(payload, new_residual)`; `decompress(payload)` returns the
reconstruction described in step 3, shaped like `buf`.

Error contract:
- `TypeError` if `buf` or `residual` is not a `torch.float32` tensor.
- `ValueError` if `residual` does not have the same number of elements as `buf`.

## Why the current implementation moves too many bytes

The current code puts the **entire compensated buffer** into the payload at
`float32` (4 bytes per element) and never compacts it — so it moves the maximum
possible number of bytes every step, and the persistent accumulator is trivially
zero. Send a compact payload instead: one scale per block plus one bit per element
is a small fraction of `4 * numel` bytes. Build the scale, the sign mapping, the
bit-level packing/unpacking, and the accumulator update yourself from primitive
tensor operations.

**Forbidden:** the framework's built-in quantization primitives —
`torch.quantize_per_tensor`, `torch.quantize_per_channel`,
`torch.fake_quantize_per_tensor_affine`, `torch.fake_quantize_per_channel_affine`.
The scoring harness stubs these to raise at runtime, and the verifier scans your
submitted file for those tokens (do not reference them even in comments — the scan
is textual and scores the task 0). Build the compaction yourself.

## Correctness comes first

Byte savings only count if the compressor is **correct and unbiased**. The verifier
checks, on multiple workloads:

- the accumulator identity `new_residual == (buf + residual) - decompress(payload)`
  (exact, up to fp32 rounding);
- **multi-step conservation** — over `K` steps against a fixed target `t`, the
  running total of the decompressed outputs plus the final accumulator equals
  `K * t`;
- **multi-step unbiasedness** — the running *mean* of the decompressed outputs
  converges to `t` (a compressor that drops the accumulator update, or reconstructs
  a biased/near-empty estimate, drifts away and fails);
- shape/dtype and determinism of `decompress`;
- boundaries (element count not a multiple of the block size; all-positive /
  all-negative blocks), degenerate inputs (all-zero, a single dominating spike, a
  one-element buffer), the error contract, and a metamorphic scale check
  (scaling `buf` by `c > 0` scales the reconstruction by `c`).

A payload that is smaller but fails any correctness gate scores zero.

## Scope

Optimize the product implementation in `grad_compress.py` only. Do **not** edit
tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares the bytes in your encoded sign payload with the baseline payload size,
not the time needed to encode it.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of `/app/repo/grad_compress.py`, decide on your
   approach, and make **all** the edits you want in `/app/repo`.
2. Self-test as much as you like with scratch scripts **you** write yourself
   (put them outside the scored file, e.g. under `/tmp`). Verify your output
   against the behavioral contract above on your own inputs, and measure your own
   before/after to convince yourself the change genuinely moves fewer bytes.
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
well as lean.

**🎓 Explain your work for a beginner.** Before you edit, state your approach:
what limits the current code and what you intend to change. After you edit
and before you submit, give a short, concrete, step-by-step account of what you
changed, *why* it should be better in terms of the actual code path, and how you
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
