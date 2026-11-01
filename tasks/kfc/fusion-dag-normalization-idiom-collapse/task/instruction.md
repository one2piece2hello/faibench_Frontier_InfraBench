# Performance Optimization Task

You are working on a **graph fusion pass** inside an ahead-of-time compiler for
tensor programs. A model is lowered to a computation **DAG** whose nodes are
primitive ops (elementwise arithmetic, reductions, dtype/pass-through markers).
Before the graph is handed to code generation, a fusion pass rewrites it into an
**equivalent** graph with **fewer nodes** — collapsing a recognized numerical
idiom into a single fused op and dropping redundant pass-through nodes. Fewer
nodes means fewer kernels launched downstream. The file `dag_fusion.py` implements
this pass as `fuse(graph)`. It is correct but leaves many nodes behind.

## Behavioral contract

### Graph representation

A *graph* is a plain dict (no third-party types):

```python
{
  "inputs":    ["x"],                       # external input tensor names (runtime)
  "outputs":   ["y"],                        # external output tensor names
  "constants": {"two":[2.0], "eps":[1e-5],   # compile-time constant tensors
                "w":[...], "b":[...]},
  "nodes": [                                 # each node produces exactly one tensor
    {"op":"ReduceMean","name":"n0","inputs":["x"],     "outputs":["mean"],"attrs":{}},
    {"op":"Sub",       "name":"n1","inputs":["x","mean"],"outputs":["c"], "attrs":{}},
    # ...
  ],
}
```

A **node** has an `op` (op-type string), a unique `name`, `inputs`/`outputs`
(tensor-name lists), and an `attrs` dict. A tensor is *defined* as a graph input,
a key of `constants`, or a node output. Edges are implicit via shared tensor
names. The graph is single-assignment (each tensor is produced once).

### Tensor / op semantics

Tensors are 1-D float vectors; a **length-1 vector is a scalar that broadcasts**.

| op | meaning |
|----|---------|
| `Sub/Add/Mul/Div(a,b)` | elementwise, with length-1 broadcast |
| `Pow(a,e)` | `a[i] ** e`, where `e` is a length-1 **constant** exponent input |
| `ReduceMean(a)` | `[mean(a)]` (a length-1 vector) |
| `Sqrt(a)` | elementwise square root |
| `Identity(a)` / `Cast(a)` | pass-through / dtype-only — a **numeric no-op** here |
| `FusedNorm(x,w,b)` attr `epsilon` | the collapsed idiom (below) |

### The idiom to collapse

The pass recognizes the op chain that spells out a **layer normalization with an
affine transform** over the vector: the mean over the vector is subtracted, the
centered values are squared and averaged to give the variance (variance = **mean of
squares** of the centered values), a small positive epsilon is added, the square root
of that is divided into the centered values, and the normalized result is finally
scaled and shifted by two 1-D constants. In the graph this appears as the primitive
ops of the table above (mean reduction, subtract, square via `Pow`, mean reduction,
add, `Sqrt`, divide, multiply, add).

Such a chain is replaced by a single node
`{"op":"FusedNorm","inputs":[x, w, b],"outputs":[y],"attrs":{"epsilon": eps}}`
whose value equals the chain's output **exactly** (same variance convention, epsilon
carried as the attribute).

### `fuse(graph) -> graph`

Return a **new** graph with the same `inputs`, `outputs` and external behaviour but
fewer `nodes`. Public API — do **not** rename `fuse` or change the graph shape.

## Correctness comes first

The verifier scores your output graph against an **independent evaluator**: for
many random inputs it evaluates both the input graph and your output graph and
requires **every external output to match**. Correctness is external-output
equivalence — never a match against any particular target graph. A collapse that
changes any output, or that deletes a tensor another node still reads, scores
**zero**.

Because equivalence is mandatory, you must **not** collapse a subgraph when doing
so would change the result. Leave it expanded (still correct) when:

- the square's exponent constant is not exactly `2`;
- the epsilon operand is not a compile-time **constant scalar** (a fused node
  carries epsilon as an attribute, so a runtime epsilon is not representable);
- the scale/shift operands are not **1-D constants**;
- an **interior** tensor of the matched chain (any tensor the chain produces other than
  its final output) is consumed **outside** the matched region — collapsing would drop
  a value still needed.

A graph containing no idiom must be returned **unchanged** (same node count).

## The optimization

The current pass recognizes the idiom only when it is spelled in one exact, rigid
shape, and it deletes no redundant nodes. A graph that contains the *same* computation
written differently — for instance with a numeric no-op node sitting inside the chain,
or with a repeated subexpression materialized by more than one node — is therefore
left fully expanded, and so are no-op nodes that could simply be dropped.

Make the pass collapse **more** — produce an output graph with **fewer nodes** for
the same external behaviour. Always honour the four safety conditions above.

**Forbidden:** delegating the traversal / matching / rewrite to a graph-optimizer,
a pattern-rewrite engine, or a framework graph library. The scoring harness scans
your submitted file for those and scores the task `0` (do not reference them even
in comments). Build the traversal, the multi-condition match, the safety check,
and the node replacement yourself.

## Scope

Optimize the product implementation in `dag_fusion.py` only. Do **not** edit tests,
the benchmark harness, workloads, or dependency/build files. The final submitted
diff must contain only product-code changes.

## How this task is scored — ONE single graded submission (READ CAREFULLY)

The grader counts operations in the output graph and compares that count with the baseline
output.
Let `s` be the resulting improvement ratio (baseline / candidate) and `r` the reference
implementation's ratio against the same baseline. A failed validity check, `r <= 1`, or
`s <= r` gives zero reward. Otherwise the reward is `min(1, ln(s/r) / ln(r))`.

You get **exactly one** submission. There is no iterative loop, no dev feedback
round, and no second chance.

1. Read the current implementation in `/app/repo/dag_fusion.py`, work out your plan, and
   make your changes.
2. Self-test as much as you like **using your own scratch scripts** (write them
   outside the product file, e.g. under `/tmp`), because you will get no feedback
   from the grader before your score is fixed.
3. Once your pass preserves the result and reduces the output graph's operation count,
   submit **once**:

   ```
   bash /opt/loop/submit.sh
   ```

**Submitting ends the task.** Once the submission is recorded, the harness finalizes
automatically. Later calls do not score new edits, so finish your own checks first.

Your reward comes from a full, trusted end-of-session verifier over the whole
correctness suite plus the performance measurement, so keep your submission
genuinely correct: **any** failing correctness case scores the whole task `0`,
regardless of how compact the output graph is.

**🎓 Explain your work.** Before you edit, state your approach: what in the
current code path is doing redundant work, and what you will change. After you
edit and before you submit, give a short, concrete, step-by-step explanation of
which operations your pass removes or fuses, and how you checked that the resulting graph
remains equivalent. Write it for a beginner reader.

## 🔴 Red line (hard — ANY one of these makes the whole task score 0)

The scoring machinery is off-limits. Solve this task **independently**, using
ONLY the code in `/app/repo`, the sanitized feedback that `submit.sh` returns,
and your own knowledge and reasoning.

- Do **NOT** read, `cat`, `open`, `less`, `head`/`tail`, `grep`, `find`, or edit
  ANY verifier / correctness / timing / scoring / hidden-test file or directory,
  wherever it may live. Your ONE call to `bash /opt/loop/submit.sh` is the ONLY sanctioned
  interaction with the scoring machinery.
- Do **NOT** run any verifier directly or try to reproduce or reverse-engineer it
  — your single `bash /opt/loop/submit.sh` call is the only scoring path.
- Do **NOT** search for, print, or infer the hidden workloads, seeds, thresholds,
  the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up the source PR, the upstream repository, or a
  reference solution (no web search, no `git` remote fetch, no network to the
  source), whether the internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring / source
  material, however you might have come to see it.

Stay inside `/app/repo` and scratch directories you create yourself; you get no scoring signal
before your single submission — rely on your own reasoning and self-tests.

## 提交方式

本题只允许提交一次。先用自己的脚本自测，再运行 `bash /opt/loop/submit.sh`。提交记录成功后会自动定稿，无需另调 `--finalize`。最终奖励由会话结束后的评测器给出；再次调用提交命令不会给新的改动评分。
改动留在工作树里,不需要 `git commit`(判分读工作树,仓库 HEAD 必须停在初始基线 commit 上)。
