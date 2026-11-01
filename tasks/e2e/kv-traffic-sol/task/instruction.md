# Paged KV-cache traffic — move the cache at the speed of the hardware

You have the complete source tree of **vLLM 0.10.1.1** at `/app/repo` (installed and importable;
`import vllm`, `import torch`, `import triton` all work) on a single **NVIDIA H20** GPU.

An LLM serving engine keeps every request's keys and values in a **paged KV cache**: a pool of
fixed-size pages, plus a per-request *block table* that says which physical page holds which slice
of that request's token positions. The engine then spends a large part of its memory bandwidth just
**moving that cache around** — writing freshly computed keys/values into the pages of a ragged batch
of requests, pulling a request's paged history back out into a packed contiguous buffer (for
prefix reuse, KV transfer between engines, offload/restore, cache inspection), and duplicating whole
pages when two requests fork or share a prefix. None of that arithmetic is interesting; all of it is
**data movement through HBM**, and how fast it runs is decided by the layout you store the cache in,
how you resolve the block-table indirection, how coalesced your accesses are, how wide your loads
are, and how many passes over the bytes you make.

**Your job: make that data movement as fast as the memory system allows.**

The whole repository is yours. Change any part of it — the pool layout and strides, the paging
scheme, the indirection, the Triton kernels, a CUDA extension you compile yourself (`nvcc` is
present), the dispatch logic — or write something entirely new. Nothing in `/app/repo` is off
limits and you do not have to use vLLM's own code at all.

## What is measured

A verifier you cannot see drives your engine over **hidden workloads** (a range of batch sizes,
head geometries, page sizes, layer counts and per-request position ranges) and measures, **from
outside your process**, the wall-clock time of complete traffic steps. It computes the **minimum
number of bytes** each step must read and write itself, from the workload definition, so moving
*more* bytes than necessary can only cost you — the score is not something your code reports.

The verifier compares your timings to a well-tuned baseline built from the repository's paged-KV
movement paths. It also reports the fraction of peak HBM bandwidth achieved per case as a
diagnostic; that fraction is not the reward.

For each paired run, take the geometric mean of `baseline time / candidate time` across the
held-out workload cases. Let `s` be the median of those paired speedups and `r` the calibrated
reference speedup against the same baseline. A failed validity check, `r <= 1`, or `s <= r` scores 0.
Otherwise the final reward is `min(1, ln(s/r) / ln(r))`, capped at 1 when `s >= r²`.

The shipped `/app/repo/submission/kv_traffic.py` is already correct. It still has to beat the
reference speedup to earn a positive reward.

## Required API contract (frozen — the verifier calls exactly this)

Your implementation must live at **`/app/repo/submission/kv_traffic.py`** and expose a class
`KVTrafficEngine`. Everything your solution needs at scoring time must persist under
`/app/repo/submission/` (edits to installed site-packages alone may be lost when your work is
replayed); if you build a compiled extension, build it into `/app/repo/submission/` and load it
from there.

```python
class KVTrafficEngine:
    def __init__(self, cfg: dict) -> None:
        """cfg keys (all present; ints unless noted):
             num_layers, num_kv_heads, head_size, page_size, num_pages,
             max_batch, max_pages_per_request,
             dtype  : str, "bfloat16" or "float16" — the KV element type
             device : str, e.g. "cuda"
           Logical token position p of a request lives in page block_table[p // page_size]
           at offset p % page_size."""

    def allocate(self) -> None:
        """Allocate your paged pool: per layer, num_pages pages of page_size token slots, each
           slot holding one key vector and one value vector of num_kv_heads*head_size elements
           of cfg['dtype']. The internal layout is YOURS to choose. See the memory rule below."""

    def begin_step(self, plan: dict) -> None:
        """Called once immediately before a run of per-layer scatter() or gather() calls that
           all use this same plan (this is how a serving engine works: one plan per forward
           pass, every layer consumes it). INSIDE the measured region.
           plan keys:
             "block_table"      cuda int32 [batch, max_pages_per_request]; entry j is the
                                PHYSICAL page holding logical positions
                                [j*page_size, (j+1)*page_size). Unused entries are -1.
                                Two requests MAY name the same physical page (a shared prefix);
                                when they do, the bytes there are identical.
             "ctx_lens"         cuda int32 [batch]; first logical position of this step's range
                                for each request. NOT necessarily a multiple of page_size.
             "new_lens"         cuda int32 [batch]; how many consecutive logical positions this
                                step covers for each request. MAY be 0.
             "block_table_cpu", "ctx_lens_cpu", "new_lens_cpu"
                                host int32 mirrors of those three tensors (same values).
             "total_tokens"     python int, == sum(new_lens).
             "batch"            python int.
           A plan is valid only until the next begin_step(); you may cache anything derived from
           it for that step's layers, but a later step's tensors and values will differ."""

    def scatter(self, layer: int, k_src, v_src) -> None:
        """Store k_src/v_src into `layer`'s pool at the positions the current plan describes.
           k_src, v_src: [total_tokens, num_kv_heads, head_size], dtype == cfg['dtype'],
           packed REQUEST-MAJOR: rows [cumsum(new_lens)[b], cumsum(new_lens)[b+1]) are
           request b's tokens in increasing logical position.
           You may NOT keep a reference to k_src/v_src instead of storing the bytes: the
           verifier overwrites those buffers after the call returns. TIMED."""

    def gather(self, layer: int, k_out, v_out) -> None:
        """Fill k_out/v_out — [total_tokens, num_kv_heads, head_size], dtype == cfg['dtype'],
           allocated and owned by the VERIFIER — with the keys/values stored at the positions
           the current plan describes, in the same request-major packing. EVERY element must be
           written: the verifier pre-fills these buffers with a poison value and requires the
           result to be bit-identical to what was stored. TIMED."""

    def copy_pages(self, layer: int, src_pages, dst_pages) -> None:
        """src_pages, dst_pages: cuda int32 [n] physical page ids, the two sets DISJOINT.
           For every i, page dst_pages[i] of `layer` must become an exact copy of page
           src_pages[i] — all page_size slots, keys and values. TIMED."""

    def reset(self) -> None:
        """Drop all state; the pool may be reused afterwards. Not timed."""
```

## Rules the verifier enforces (failing any of them scores 0)

1. **Bit-exact round-trip.** What `gather` returns must equal what was `scatter`-ed, **bit for
   bit**, and a page copied by `copy_pages` must read back bit-identically. This is data movement:
   there is no tolerance. Any lossy re-encoding (lower-precision storage, quantisation, dropping
   or summarising positions) fails. Lossless transformations are fine.
2. **Every output element is written.** Verifier-owned output buffers arrive poisoned; leaving any
   element unwritten is a failure, so you cannot skip part of the work.
3. **Nothing is aliased instead of stored.** The verifier overwrites the source buffers after a
   `scatter` and then reads the data back through `gather`.
4. **The current plan wins.** Two `begin_step` calls may arrive back to back; the second one
   defines what the following calls mean.
5. **Memory.** The verifier measures your allocation itself. What `allocate()` reserves must not
   exceed **1.10x** the nominal pool size
   (`2 * num_layers * num_pages * page_size * num_kv_heads * head_size * itemsize` bytes), and
   your peak allocation must stay within a small working allowance of it — so a second,
   differently-laid-out copy of the cache, or a staging buffer the size of the workload, will not
   fit.
6. **Shapes, dtypes and devices** exactly as specified above; no crashes.
7. Correctness is checked **first**, over hidden configurations that include awkward geometries
   (single pages, partial tail pages, unaligned starts, shuffled page order, shared pages,
   zero-length ranges, one-token writes, `page_size == 1`, several layers). It is a hard
   prerequisite: failing it scores 0 before any timing happens.

## Iterating

`python3 /app/dev_bench/run_dev_bench.py` runs your engine over a **public** development workload
set (different from the scored one), checks the round-trip, and prints each case's step time and
achieved bandwidth. Use it to iterate; faster there generally means a higher score, but the scored
workloads are not the same ones.

## Hard red line (any one of these makes the whole task score 0)

- Do **not** read, `cat`, `grep`, `find`, copy or edit any verifier, hidden-test or evaluation file
  (anything under `/tests`, `/opt/verifier`, `/opt/negative`), and do not try to infer their
  contents.
- Do **not** disable, kill or tamper with the timer daemon or the measurement path.
- Do **not** access the internet at runtime, and do not attempt to bypass the proxy isolation.

Solve it yourself. The upstream vLLM sources at `/app/repo`, the public literature on paged
attention and KV-cache management, and your own ideas are all fair game — what is forbidden is
touching the measurement.

## Submission / 提交方式

In the default loop16 image, run `bash /opt/loop/submit.sh` once per candidate (1–16 submissions).
To stop before round 16, run `bash /opt/loop/submit.sh --finalize --reason "..."`; round 16 finalizes automatically.
Of the candidates that pass the dev checks, the one with the highest public dev bandwidth (GB/s) is selected for the
single held-out evaluation. The dev metric ranks candidates; it is not the final reward.
If you run the base image without the submit helper, the grader uses the final working tree instead.
请保持 `/app/repo` 的 HEAD 在初始提交，不要 `git commit`。默认镜像须用上述脚本提交；仅基础镜像直接评测最终工作树。
