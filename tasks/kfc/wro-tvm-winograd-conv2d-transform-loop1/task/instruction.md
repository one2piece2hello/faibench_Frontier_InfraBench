# Performance Optimization Task

A 3x3 convolution on a phone or an edge box is the single most expensive node in most
vision graphs, and there is no GPU behind it — the host CPU *is* the convolution. Every CPU
inference runtime therefore ships a Winograd path: instead of `9 * CI * CO` multiplies per
output pixel it cuts the activation into overlapping `alpha x alpha` tiles
(`alpha = m + r - 1`), maps each tile and each filter into a transform domain with the
constant Cook-Toom matrices `B` and `G`, multiplies *element-wise* in that domain — which
collapses into one `CI`-deep GEMM per transform position — and maps the result back down to
an `m x m` output tile with `A`. For `F(4, 3)` that is 36 multiplies per tile per channel
pair instead of 144.

This module is that front end. It is functionally correct but **slow**: each of the six
stages is a literal transliteration of the scalar reduction its `te.compute` declares — one
output element and one tap at a time. Make it **faster** on the benchmark workload while
**preserving its numerical contract**: every returned array must match the independent
reference to `1e-12` relative, every array must stay `float64` with the documented shape,
and every documented `ValueError` must still be raised.

## Editable scope

Edit **only** this file (any edit outside this scope scores the whole task zero):

```
winograd_conv.py
```

## The subsystem

Everything is NHWC and `float64`. `data` is `[N, H, W, CI]`, `weight` is `[KH, KW, CI, CO]`
and the convolution is a **correlation** (the filter is not flipped). Only `stride = 1` and
`dilation = 1` are in scope, and the kernel must be square. `P = N * nH * nW` is the
flattened tile index, walked as `p = n * nH * nW + ph * nW + pw`.

Three cheap planners are graded for correctness and are **not** where the time goes:

* `winograd_output_size(in_size, pad_begin, pad_end, kernel_size)` —
  `in_size + pad_begin + pad_end - kernel_size + 1`.
* `winograd_tile_geometry(batch, out_h, out_w, tile_size)` — `(nH, nW, P)` with
  `nH = ceil(out_h / m)`, `nW = ceil(out_w / m)`, `P = batch * nH * nW`.
* `winograd_transform_matrices(tile_size, kernel_size)` — the Cook-Toom triple
  `(A, B, G)` with shapes `[alpha, m]`, `[alpha, alpha]` and `[alpha, r]`. The reduction
  index is the **first** axis of `A` and `B` and the **second** axis of `G`. Supported range
  is `2 <= tile_size <= 8` and `3 <= kernel_size <= 7`; anything else raises `ValueError`.

Six coupled stages carry the cost. All of them are graded — the pipeline entry point
`winograd_conv2d` is graded end to end, and each stage is *also* graded on its own, so a
stage cannot be "optimised" by folding it into a caller.

1. `pad_and_tile(data, padding, tile_size, kernel_size)` — `[N, H, W, CI]` ->
   `input_tile[alpha, alpha, P, CI]`:

   ```
   input_tile[eps, nu, p, ci] = data_pad[p // (nH*nW),
                                         ((p // nW) % nH) * m + eps,
                                         (p % nW) * m + nu, ci]
   ```

   where `data_pad` row `0` sits at input row `-pad_top` and column `0` at input column
   `-pad_left`. Every read outside the real activation — the explicit padding *and* the
   extra tail rows and columns the last tile of each axis needs — contributes `0`; that
   zero fill is what lets the transform run a full `alpha x alpha` tile unconditionally.
   `padding` is `(pad_top, pad_left, pad_bottom, pad_right)`; a pair is read as
   `(top, left)` mirrored and a scalar as all four. Returns a fresh array — it must not
   alias its input.
2. `transform_input(input_tile, B)` — `data_pack[alpha, alpha, P, CI]`:

   ```
   data_pack[eps, nu, p, ci] = sum_{r_a, r_b} input_tile[r_a, r_b, p, ci]
                               * B[r_a, eps] * B[r_b, nu]
   ```

   i.e. `B^T . tile . B` on the two transform axes, `p` and `ci` untouched. Together with
   the batched GEMM this stage dominates the benchmark cost.
3. `transform_kernel(weight, G)` — `[KH, KW, CI, CO]` -> `kernel_pack[alpha, alpha, CO, CI]`:

   ```
   kernel_pack[eps, nu, co, ci] = sum_{r_kh, r_kw} weight[r_kh, r_kw, ci, co]
                                  * G[eps, r_kh] * G[nu, r_kw]
   ```

   `eps` pairs with the kernel *row* and `nu` with the kernel *column*, and the output
   carries `co` **before** `ci` — the transpose of the `weight` channel order.
4. `batched_gemm(data_pack, kernel_pack)` — `bgemm[alpha, alpha, P, CO]`:

   ```
   bgemm[eps, nu, p, co] = sum_ci data_pack[eps, nu, p, ci] * kernel_pack[eps, nu, co, ci]
   ```

   One independent `[P, CI] x [CI, CO]` product per transform position; nothing couples
   `(eps, nu)` to anything else.
5. `inverse_transform(bgemm, A)` — `inverse[m, m, P, CO]`:

   ```
   inverse[vh, vw, p, co] = sum_{r_a, r_b} bgemm[r_a, r_b, p, co] * A[r_a, vh] * A[r_b, vw]
   ```

   i.e. `A^T . tile . A`, shrinking each `alpha x alpha` transform tile to the `m x m`
   output tile it encodes.
6. `untile(inverse, batch, out_h, out_w, tile_size)` — `[N, out_h, out_w, CO]`:

   ```
   out[n, h, w, co] = inverse[h % m, w % m,
                              n * nH * nW + (h // m) * nW + (w // m), co]
   ```

   When `out_h` or `out_w` is not a multiple of `m` the trailing rows and columns of the
   last tile are dropped. Returns a fresh array — it must not alias its input.

`winograd_conv2d(data, weight, padding=0, tile_size=4)` chains them:
`winograd_transform_matrices`, `pad_and_tile`, `transform_kernel`, `transform_input`,
`batched_gemm`, `inverse_transform`, `untile`. It returns
`{"out", "A", "B", "G", "input_tile", "data_pack", "kernel_pack", "bgemm", "inverse",
"alpha", "tile_size", "out_h", "out_w", "num_tiles"}`.

Contract notes that are easy to break:

* The reduction index of `B` is its **first** axis on *both* sides of the forward
  activation transform. `B` is square, so contracting the other axis has the right shape
  and computes a different linear map.
* `G` is `[alpha, r]`, so its reduction index is the **second** axis; `A` is `[alpha, m]`,
  so its reduction index is the **first**.
* `transform_kernel` emits `[..., co, ci]` while `weight` is `[..., ci, co]`. Forgetting
  that transpose still type-checks whenever `CI == CO`.
* `inverse_transform`'s `vh` is the tile **row** and pairs with the *first* `A` factor.
  Swapping `vh` and `vw` has the right shape and transposes every output tile.
* `untile` walks the tile grid **row-major with stride `nW`** and takes the position inside
  a tile from the *remainder*, not the quotient. A column-major tile walk has the right
  shape and the wrong contents.
* Both `pad_and_tile` and `untile` must return a **fresh** array; returning a view of the
  input fails. Beware the degenerate shapes where a reshape-plus-slice happens to stay
  contiguous.
* The zero fill in `pad_and_tile` covers *two* different regions: the explicit padding quad
  and the tail that `(nH - 1) * m + alpha` rows demand beyond `H + pad_top + pad_bottom`.
  Both are part of the contract and both are visible in the graded intermediate.
* `tile_size = 2`, a `1x1` spatial extent, a single channel, `batch > 1`, an asymmetric
  padding quad, a `5x5` or `7x7` kernel and an output extent that is not a multiple of
  `tile_size` all have to work. Non-finite input raises `ValueError`.

## Constraints

* Python standard library plus **numpy** (already installed). No new dependencies, no C
  extensions, no subprocesses, no threads, no torch/scipy/tvm.
* Do not weaken, special-case or precompute for the benchmark inputs: the verifier runs a
  separate correctness suite (a hand-traced `winograd_output_size` table plus an exhaustive
  geometry cross-check, a dense `winograd_tile_geometry` sweep, the Cook-Toom matrices
  checked against literal golden values, against an independently derived reference and
  against the defining minimal-filtering identity for all 35 supported `(m, r)` pairs, nine
  tile-gather configurations, six forward activation transforms, eight filter packings,
  seven batched GEMMs, seven inverse transforms, seven output scatters, eight end-to-end
  pipeline configurations each cross-checked against both the chained stages and a plain
  direct convolution, and a smallest-legal-block degeneracy invariant) against an
  independent reference, and any mismatch scores **zero**.
* Do not touch the verifier, the tests directory, or anything outside `winograd_conv.py`.

## How you are scored

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

Correctness is required: one mismatched element scores 0 regardless of speed.
The benchmark runs two Winograd blocks: an
`F(4, 3)` block over a `1x20x20x16` input with a `3x3x16x24` filter and `padding=1`, then an
`F(2, 3)` block over a `1x16x16x24` input with a `3x3x24x16` filter and `padding=1`. That is
roughly 900k forward-transform multiply-accumulates, 740k transform-domain GEMM
multiply-accumulates, 410k inverse-transform taps, 180k filter-transform taps and 40k layout
stores. There is a large amount of headroom and it **grows with the channel count, the
spatial extent and `alpha`**, so a structurally better block wins by much more than a
micro-optimized one.

## Where to start

Read the module docstring, then each stage's docstring: they pin the exact contract. Then
look at what the innermost Python loop of each stage actually computes and ask whether the
same number can be produced for the whole array at once. The tile gather is a zero-padded
copy plus `alpha * alpha` strided slices. Both forward transforms and the inverse transform
are a pair of contractions against a tiny constant matrix — `alpha^4 * P * CI` Python-level
multiply-accumulates for a quantity two `tensordot`s produce. The batched GEMM is
`alpha * alpha` independent matrix products that one stacked `@` evaluates in a single call.
The output scatter is a reshape, a transpose and a crop. The six stages are independent of
each other — each one you lift off the Python interpreter takes another slice out of the
wall clock.

## How you are scored (ONE single graded submission)

- You get **exactly one** graded submission. Submit it with
  `bash /opt/loop/submit.sh`.
- **Submitting ends the task.** The moment `submit.sh` returns you are done: stop
  editing and stop working on this task.
- The harness finalizes automatically after the one submission. Calling it again
  does not score new edits.
- You will **not** get iterative feedback to improve against. Nothing is measured
  for you round by round, so there is no measured signal to chase.
- Therefore: read the code, settle the design, and **self-test thoroughly with
  your own scratch scripts** before you submit. Correctness is a hard gate — a
  fast but incorrect submission scores **zero**.
- Explain your approach and your reasoning in writing before you submit.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
