# Performance Optimization Task

You are working on the index-arithmetic front-end of a kernel code generator. When
a high-level tensor program is lowered to a loop nest, each memory access turns into
an integer *index expression* over the loop variables — address math like
`(i % 4) + (i // 4) * 4`. These expressions are built up mechanically and are often
far larger than they need to be. Before the expression is handed to the backend it
is passed through a simplifier that rewrites it into a smaller, canonical form that
computes the *same* integer. The file `index_expr_simplify.py` implements this
simplifier as a function `simplify_expr`. It is correct but leaves large expressions.

## Behavioral contract

An index expression is a tree built from these node types (plain nested tuples):

```
("const", n)                      an integer literal n
("var", name)                     a variable (name is a str)
("add",      left, right)         left + right
("mul",      left, right)         left * right
("floordiv", left, right)         left // right     (Python floor division)
("mod",      left, right)         left %  right     (Python modulo)
("min",      left, right)         min(left, right)
("max",      left, right)         max(left, right)
```

Every variable is a **bounded non-negative integer**. The second argument,
`bounds`, is a dict mapping each variable name that occurs in the expression to an
inclusive integer range `(lo, hi)` with `0 <= lo <= hi`.

```python
def simplify_expr(expr, bounds):
    """-> a new expression tree in the same grammar."""
```

The returned tree must satisfy **both**:

- **Value equality.** For every assignment of the variables to integers within
  their declared ranges, the returned tree evaluates (with Python `//` and `%`
  semantics) to the same integer as `expr` — or raises the same arithmetic error
  (e.g. division by zero) on the same assignments.
- **No growth.** The returned tree contains no more nodes than `expr`. A
  simplification never makes the expression bigger.

Error contract: raise `ValueError` if a variable that occurs in `expr` has no entry
in `bounds` (it is unbounded and cannot be reasoned about), and raise `ValueError`
if a `floordiv` or `mod` has a divisor that is the constant `0`.

Public API (do **not** change the name or the call shape): `simplify_expr(expr,
bounds) -> expr`.

## Why the current implementation leaves large expressions

The current code makes a single bottom-up pass that only folds all-constant
subtrees and the most obvious per-node identities (adding `0`, multiplying by `1`
or `0`). It never re-examines a node after rewriting a child, and it never relates
terms that sit on opposite sides of a `+`. So an expression whose size collapses
only after reasoning about how a remainder and its matching quotient fit together,
or that needs a second pass once a subtree has shrunk, stays large. Make the output
**smaller** — fewer tree nodes for the same values — while keeping the contract
above exact.

**Forbidden:** delegating the simplification to a computer-algebra system or any
external symbolic-math package. The scoring harness scans your submitted file for
those imports and scores the task `0` (do not reference them even in comments).
Build the rewrite yourself over the tuple grammar. Standard library only.

## Correctness comes first

The verifier compares your output against an independent reference on many
expressions. It checks value equality by **enumerating the full bounded domain** of
each expression's variables (every in-range assignment), covering: the common
recombination idioms and trivial identities, constant-only trees, a bare variable,
a variable pinned to a single value, an already-canonical expression (returned
unchanged), an unbounded variable and a divide-by-zero constant (both rejected),
idempotence (simplifying twice changes nothing), non-power-of-two moduli, and a
work-evidence check that the output still genuinely depends on its variables. A
smaller result that is wrong on even one in-range assignment scores zero.

## Scope

Optimize the product implementation in `index_expr_simplify.py` only. Do **not**
edit tests, benchmark harnesses, workloads, or dependency/build files. The final
submitted diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader compares the number of expression-tree nodes after simplification with the
baseline node count.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly ONE submission**. There is no iteration loop, no dev feedback
round, and no second chance.

1. Read the current implementation of `/app/repo/index_expr_simplify.py`, decide on your
   approach, and make **all** the edits you want in `/app/repo`.
2. Self-test as much as you like with scratch scripts **you** write yourself
   (put them outside the scored file, e.g. under `/tmp`). Verify your output
   against the behavioral contract above on your own inputs, and measure your own
   before/after to convince yourself the change genuinely produces smaller output.
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
well as small.

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
