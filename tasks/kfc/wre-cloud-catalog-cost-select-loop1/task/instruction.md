# Choosing Cloud Instance Types for a Batch of Resource Requests — Implementation Task

## Objective
A cloud catalog is a flat table of *offerings*: one row per (cloud, region, instance type), carrying that
type's vCPU count, its memory, the accelerator it comes with, and the two prices it is billed at — on
demand and, where the cloud offers one, spot. A resource request does not name an instance type; it states
what it *needs* — optionally a particular cloud, at least so many vCPUs, at least so much memory, exactly
so many accelerators of one kind — how it is willing to be billed, and how much it is willing to pay.
Turning a batch of such requests into a plan means, for every request, narrowing the catalog to the
offerings that satisfy it, working out what each of those would actually cost *under that request's own
billing mode*, picking the cheapest, and — because the cheapest region can be out of capacity when the
request is really launched — laying out the ordered ladder of regions it should fall back through.

That is one subsystem, and every stage feeds the next: the billing mode decides what a row costs, what a
row costs decides whether the price cap admits it, the admitted rows decide the pick and its tie-break,
the same admitted rows decide which regions are candidates and what each candidate region's own cost is,
and that is what the fallback ladder orders and truncates. Get the mode-1 spot preference, the exact
accelerator (kind, count) match, the tie-break order or the "count the regions before you truncate the
ladder" rule subtly wrong and you get a plan that still looks self-consistent while quietly overpaying,
launching the wrong shape, or falling back to a region that never had capacity for the request at all.
Two public entry points (the plan alone, and the plan plus a batch of catalog name lookups) are already
written; both are thin wrappers over a single core whose body is **not implemented** (it raises
`NotImplementedError`). Implement it to the contract below, then make it fast.

## Editable scope (only this file may change)
```
catalog_select.py
```
Any change to a file outside this scope fails the task.

## Interface contract (implement EXACTLY this function)

`_select_core(row_cloud, row_region, row_type, row_vcpu, row_mem, row_acc, row_accn, row_price, row_spot,
req_cloud, req_vcpu, req_mem, req_acc, req_accn, req_mode, req_cap, max_ladder, query_cloud, query_type)`
— the two public wrappers `select_offerings(...)` and `select_offerings_with_lookup(...)` are already
written and must keep working unchanged. The **full, authoritative contract is the `_select_core`
docstring in `catalog_select.py`**; read it first, including the notation (`n_rows`, `n_req`, `n_query`
and `mode(q)`). In outline:

* **Validation**, all `ValueError`, all of it before any output is produced: the conditions the docstring
  lists — sixteen 1-D `int64` arrays with the documented cross-lengths, a `max_ladder` that is a real
  python `int` (a `bool` does not count) inside its range, cloud / region / type / accelerator ids, vCPU
  and memory sizes, prices and caps each inside their own ranges, a row (and a request) carrying an
  accelerator *kind* exactly when it carries a positive *count*, a billing mode that is one of three, and
  the two lookup inputs either both `None` or both 1-D `int64` arrays of a common length whose entries
  name a real cloud and a real instance-type id. No input array is ever mutated.
* **What a row costs a request.** Mode `0` bills on demand only, mode `2` at spot only — under each the
  row is billable only if it has that price. Mode `1` takes the row's **spot** price whenever it has one
  and its on-demand price otherwise, and is unbillable only when it has neither. A row billed at its spot
  price is *taken at spot*.
* **Feasibility.** The cloud matches or the request did not care; the row's vCPU and memory are **at
  least** what the request stated (floors, not exact sizes); the accelerator matches *exactly* — a
  request that wants none needs a row with none, and otherwise both the kind AND the count must be equal,
  so four of a kind is not served by eight of it; the row is billable under the request's mode; and what
  it costs is within the request's cap.
* **The pick.** The cheapest feasible row, ties broken by smaller `row_vcpu`, then smaller `row_mem`, then
  smaller row index; `-1` everywhere when nothing is feasible. `req_price` is what it costs and `req_spot`
  says whether that was the spot price.
* **The fallback ladder.** A request's candidate regions are the distinct regions of its feasible rows,
  and a candidate region's own cost is the least a feasible row of that region costs *that* request. The
  ladder lists them by ascending cost then ascending region, truncated to `max_ladder`; all the ladders
  are concatenated in ascending request order with `ladder_ptr` marking the group boundaries.
* **Statistics.** Per request, how many rows were feasible and how many regions were candidates — the
  latter counted BEFORE the ladder is truncated. Per row, how many requests it was feasible for and how
  many picked it.
* **Name lookup.** `None` in the plan-only flavour. Otherwise the **smallest** row index whose cloud and
  instance-type name are exactly the ones asked for, and `-1` when the catalog has no such row.
* **Returns** a `dict` with the eleven keys `req_pick`, `req_price`, `req_spot`, `req_n_feasible`,
  `req_n_regions`, `ladder_ptr`, `ladder_region`, `ladder_cost`, `row_n_feasible`, `row_n_picked`,
  `query_row` — every one a contiguous 1-D `int64` array of the documented length, with `query_row`
  `None` in the plan-only flavour. Shapes and dtypes are part of the contract.

## Correctness & how you are scored
Correctness is a hard gate. A curated set of 61 batches drives every coupled contract point at once — all
three billing modes including a mode-1 request that must take a spot price even when it is *dearer* than
the on-demand one, a row billable under neither price, a row offered at spot only, each floor met exactly
and missed by one, the pinned cloud and the cloud-agnostic request side by side, the accelerator match in
all of its shapes (none wanted, kind and count both right, the right kind at the wrong count, the right
count of the wrong kind, a kind the catalog never lists), the price cap hit exactly and missed by one and
a zero cap against a free row, the pick tie-break walked down each of its three further levels, the
ladder's cost-then-region order with the truncation at zero / one / two / beyond the number of candidate
regions while `req_n_regions` still reports the untruncated total, the per-row feasibility and pick
counters, requests with nothing feasible at all, an empty catalog, an empty request batch, a lookup that
must resolve to the smallest of several matching rows, a lookup that misses, an empty lookup batch, the
lookup-free flavour, the longest allowed prices and the largest allowed shapes, and non-contiguous and
negative-stride inputs. Every returned array is compared against an independent reference — shape, dtype,
contiguity and value — no input may be mutated, and 157 error contracts must fire.
`NotImplementedError` anywhere scores 0.

The grader measures CPU time on one large hidden batch (of the order of tens of
thousands of catalog rows and tens of thousands of requests, plus tens of thousands of catalog name
lookups), so the feasibility sweep, the pick, the ladder and the lookup all matter. A direct
transcription of the contract into a per-request pass over the whole catalog is correct but several times
off the pace. `numpy` is available.

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

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
