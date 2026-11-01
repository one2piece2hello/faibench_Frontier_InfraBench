# Committing the Accepted Prefix of a Verified Draft Tree — Implementation Task

## Objective
When a target model finishes verifying a batch of speculative draft trees, the *accepted* part of
every request has to be committed before the next round can start. The draft pass wrote its keys and
values into scratch slots of the KV pool; the accepted ones now have to end up in the slots the
request's own paged KV table points at. That is not a memcpy: every request accepted a different
number of tokens, so the plan of what moves where is ragged, and the scratch positions that survived
verification are scattered through a fixed-width table with a `-1` sentinel wherever a candidate was
rejected. On top of the movement the per-request bookkeeping has to be carried forward — the new
sequence length, the filter the next draft round is scheduled from, and the bonus token that seeds
it.

Everything above is one subsystem and every piece of it is coupled to the ragged accepted lengths, so
getting one of them off by one silently corrupts the others. Two public entry points (the plan alone,
and the plan plus the movement and the bookkeeping) are already written; both are thin wrappers over
a single core whose body is **not implemented** (it raises `NotImplementedError`). Implement it to the
contract below, then make it fast.

## Editable scope (only this file may change)
```
accept_compact.py
```
Any change to a file outside this scope fails the task.

## Interface contract (implement EXACTLY this function)

`_accept_core(req_pool_indices, req_to_token, seq_lens, num_correct_drafts, accept_index,
out_cache_loc, accept_tokens, unfinished_index, kv_cache)` — the two public wrappers
`plan_accept_move(...)` and `commit_verified_step(...)` are already written and must keep working
unchanged. The **full, authoritative contract is the `_accept_core` docstring in
`accept_compact.py`**; read it first, including the notation `accept_lens[j] = num_correct_drafts[j]
+ 1` (the `+ 1` is the bonus token). In outline:

* **Validation**, all `ValueError`, all of it before any output is produced: the fifteen conditions
  the docstring lists — ranks, `int64` / `float32` dtypes, `1 <= bs <= MAX_BS` and
  `1 <= width <= MAX_WIDTH`, the `(bs,)` companion shapes, page-table row ids in range, no negative
  lengths, the accepted run fitting in both the accept table and the page-table row, `accept_index`
  entries either `-1` or a valid `out_cache_loc` position, the optional `accept_tokens` /
  `unfinished_index` / `kv_cache` shapes, and — when a pool is supplied — both every scratch slot
  *and every planned destination* being a real row of it. No input array is ever mutated.
* **The destination plan.** `offsets` is the exclusive prefix sum of `accept_lens`, `n_move` their
  total, and each request writes `req_to_token[req_pool_indices[j], seq_lens[j] : seq_lens[j] +
  accept_lens[j]]` into `tgt_cache_loc[offsets[j] : offsets[j] + accept_lens[j]]` of a zero-filled
  `(bs * width,)` buffer. Only the first `n_move` entries are ever written.
* **The source compaction.** `accept_index` is read flat in row-major order; a position survives when
  it is not `-1`, and it lands at the number of survivors strictly before it — **one global running
  count over the whole flattened table, not a per-request one**. Sentinels may sit anywhere in a row
  and a request's survivor count need not match its `accept_lens`; the rule is total either way.
* **The bookkeeping.** `seq_lens_next = seq_lens + accept_lens` as a new array;
  `num_accept_tokens_filter` a zero-filled `(bs,)` array with `accept_lens` scattered into the
  `unfinished_index` positions; `bonus_tokens[j] = accept_tokens[j, accept_lens[j] - 1]`, i.e. the
  *last* entry of the accepted run. Each of the three optional inputs being `None` turns exactly its
  own output into `None` and changes nothing else.
* **The move.** With a pool supplied, the returned array is a fresh copy of the incoming pool in
  which, for `p` ascending over `range(n_move)`, `out[tgt_cache_loc[p]] =
  incoming[accept_out_cache_loc[p]]` — sources always read from the incoming pool, and if two plan
  entries share a destination the later one wins.
* **Returns** a `dict` with the eight keys `tgt_cache_loc` `int64`, `accept_out_cache_loc` `int64`,
  `n_move` python `int`, `n_accept` python `int`, `seq_lens_next` `int64`,
  `num_accept_tokens_filter` `int64` or `None`, `bonus_tokens` `int64` or `None`, `kv_cache`
  `float32` or `None`. Shapes and dtypes are part of the contract.

## Correctness & how you are scored
Correctness is a hard gate. A curated set of 39 verified steps drives every coupled contract point at
once — all-zero accepted counts, fully accepted requests, mixed counts, sequence lengths of `0` and
at the page-table limit, two requests sharing a page-table row, a page table whose rows repeat a slot
so duplicate destinations make the later-plan-entry-wins rule observable, sentinels in the middle of a
row, requests with fewer survivors than their accepted length and requests with more, an all-sentinel
table, an empty / unsorted / duplicated unfinished list, `width == 1` and `bs == 1`, the plan-only
flavour, each of the three `None` paths, non-contiguous and negative-stride inputs, and sixteen
generated batches. Every returned array is compared against an independent reference — shape, dtype
and value — the two scalars are checked to be real python `int`s, no input may be mutated, and 42
error contracts must fire. `NotImplementedError` anywhere scores 0.

Once correct, you are scored on **wall-clock speed** on one large hidden commit (a couple of thousand
requests, a draft width in the low tens, a KV pool of tens of thousands of rows with a few dozen
channels), so the ragged plan, the compaction and the row movement all matter. A direct transcription
of the contract into per-request Python loops is correct but roughly an order of magnitude off the
pace. `numpy` is available.

Let `s` be the candidate's speedup (baseline time / candidate time) and `r` the reference
implementation's speedup over the same baseline. A failed correctness or other hard check,
`r <= 1`, or `s <= r` gives 0. Otherwise the final reward is `min(1, ln(s/r) / ln(r))`:
matching the reference earns 0 and `s >= r²` earns 1.

## Solve independently — prohibited actions (any one => the whole task scores 0)
- Reading, printing, copying, `cat`/`grep`/`find`-ing, editing, or reproducing ANY verifier /
  scoring / hidden-test / golden file, wherever it lives; or inferring hidden inputs/thresholds.
- Downloading or cloning the upstream project or looking up its reference implementation in ANY
  form — `git clone`/`fetch`/`pull`, adding a git remote, `pip download`/`pip install` of the
  same package, `wget`/`curl` of upstream files, or web lookup — whether the network appears to
  work or not.
- Bypassing or disabling the network isolation (unsetting/overriding `http_proxy`/`https_proxy`/
  `all_proxy`, opening raw sockets, or any other circumvention).

## 提交方式

本题只有**一次**评分提交(kfc 全子集单次)。改完后运行一次 `bash /opt/loop/submit.sh`:它给当前 `/app/repo` 评分并**立即定稿**(无需再单独调 `--finalize`,也没有第二次机会 —— 再次调用只会重新定稿同一份已记录的最佳快照,不给新改动评分)。请先用你自己的脚本充分自测再提交。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
