# Performance Optimization Task

You are working on the work-reuse layer of a batch execution service. The service
receives a stream of structured **request signatures**. Before running an expensive
step for a signature, it computes a short string **identity** for that signature and
looks the identity up in a table of already-processed identities; a hit means the
prior result can be reused and the expensive step is skipped. The file
`identity_key.py` implements this identity as a single function, `identity_key`. It
is correct but produces far more distinct identities than necessary.

## Behavioral contract

```python
def identity_key(signature: dict) -> str:
    ...
```

A **signature** is a mapping describing one request:

- `"op"`: the operation name — a string.
- `"operands"`: a list of operands. Each operand is a mapping with `"shape"` (a list
  of integers) and `"dtype"` (a string).
- `"flags"` (optional): a mapping of configuration flag name → value.
- `"meta"` (optional): a mapping of incidental annotations.

`identity_key(signature)` returns a **string**. It must be **deterministic and
reproducible**: the same signature (and any signature equivalent to it, see below)
always yields the same string, on every run and every machine — so derive it from the
signature's content with a fixed rule, not from anything process- or run-dependent.

### When are two signatures equivalent?

Two signatures are **equivalent** (they describe the same request and must receive
the **same** identity) when they are equal after applying **all** of the following
normalizations. Any difference that survives all of them makes the signatures
**distinct**, and distinct signatures must receive **different** identities.

1. **Incidental annotations are ignored.** The `"meta"` mapping never affects the
   identity — two signatures that differ only in `"meta"` (or in whether `"meta"` is
   present at all) are equivalent.
2. **Field order is irrelevant.** The order in which the keys of any mapping
   (`signature` itself, an operand, `"flags"`, `"meta"`) are written does not matter.
3. **Equivalent dtype spellings are equal.** Each of these spelling groups denotes
   one element type; any spelling in a group equals any other:
   - `"f32"`, `"float32"`, `"single"`
   - `"f16"`, `"float16"`, `"half"`
   - `"i32"`, `"int32"`, `"int"`
   - `"bf16"`, `"bfloat16"`
4. **Default-valued flags are equivalent to absent flags.** A flag set to its default
   value is the same as omitting it. The defaults are:
   `fastmath = false`, `layout = "row_major"`, `precision = "default"`.
   A flag set to any **other** value is meaningful and must be kept.
5. **Order-insensitive operations.** For the operations `add`, `mul`, `max`, `min`
   the **order of the operands carries no meaning** (the same operands in any order
   describe the same request). For **every other operation** the operand order is
   meaningful and two signatures that differ only in operand order are **distinct**.

Everything not listed above is **meaningful**: the operation name, each operand's
shape, each operand's normalized dtype, the operand order for order-sensitive
operations, the number of operands, and every non-default flag value. Signatures that
differ in any of these are **distinct** and must not share an identity.

## Why the current implementation is wasteful

The current code derives the identity from the signature almost verbatim: it only
neutralizes the order of a mapping's keys. So two signatures that are equivalent but
merely spelled differently — an alternate dtype spelling, a default flag written out
in full, a stray annotation, order-insensitive operands in a different order — are
treated as **different** identities. The reuse table then holds one entry per spelling
instead of one entry per genuinely distinct request, and the expensive step is repeated
for every spelling of the same request.

Make the identity **collapse the equivalent spellings** so that equivalent signatures
map to one identity and the reuse table holds as few entries as there are genuinely
distinct requests — while **never** giving two genuinely different signatures the same
identity.

## Correctness comes first

A single **collision between two genuinely different signatures** — for example
treating two different element types, two different shapes, a different operand count,
a swapped pair of order-sensitive operands, or two different non-default flag values as
the same — is a hard failure and scores **zero**, no matter how few identities you
produce. Merging spellings that are only *incidentally* different is the goal; merging
requests that are *actually* different is never allowed. The verifier checks both a
suite of correctness cases and, over the whole workload, that no identity is shared by
two genuinely different requests.

## Forbidden

Do **not** import a tensor / graph / array framework, and do **not** import or read
the test harness or its workload. Compute the identity yourself from the signature
content using the standard library only. The scoring harness scans your submitted file
for those references and scores the task `0` (do not reference them even in comments).

## Scope

Optimize the product implementation in `identity_key.py` only. Do **not** edit tests,
harnesses, workloads, or dependency/build files. The final submitted diff must contain
only product-code changes. Public/dev feedback, when present, is representative only;
final verification runs additional trusted and hidden workloads.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader counts distinct cache keys. Equivalent spellings may share a key;
using the same key for semantically different requests fails correctness.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** graded submission. There is no iteration loop, no budget
of retries, and no feedback round you can act on: **submitting ends the task.**

1. Use scratch checks of your own to confirm that equivalent signatures share a key and
   genuinely different signatures remain distinct. Count emitted keys on your own inputs
   to assess the change.
2. When — and only when — you are done, submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

3. **That call is final.** It records your submission and finalizes automatically.
   Later calls do not score new edits; there is no second round.

Because you cannot iterate, **think the design through before you submit** and
self-test thoroughly: read the current implementation, decide on the change,
convince yourself the contract still holds on the normal, boundary, degenerate and
error-path cases described above, and only then submit. A wrong key on even one correctness
case scores zero, regardless of how many redundant keys were removed.

The grade is produced by a full, trusted end-of-session verifier (more workloads
than anything you can see), so the submitted state must be genuinely correct — not
just correct on the cases you happened to try.

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
