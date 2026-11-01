# Fix torchtitan's gpt-oss MoE expert compute (routing counts + grouped SwiGLU experts)

## Context
torchtitan is a PyTorch-native LLM training platform. Its gpt-oss Mixture-of-Experts layer
routes each token to a small number of experts and then runs a grouped expert MLP. Two coupled
files on the per-token forward path implement this:

- `torchtitan/models/common/moe.py` — the base `MoE.forward`. It builds a one-hot **routing
  map** marking which experts each token is routed to, reduces it to the **per-expert token
  counts** `num_local_tokens_per_expert_E`, and maintains a running **`tokens_per_expert_E`**
  load counter (auxiliary-loss-free load balancing, https://arxiv.org/abs/2408.15664).
- `torchtitan/models/gpt_oss/moe.py` — the gpt-oss `swiglu` activation and
  `GptOssGroupedExperts._experts_forward`. `_experts_forward` turns the per-expert token counts
  into the grouped-matmul **segment offsets** and the **repeat-interleaved per-token biases**,
  then runs the gpt-oss interleaved-gate `swiglu`.

The two files are **coupled**: `_experts_forward` consumes the exact per-expert token counts
that `MoE.forward` produces to segment the grouped matmul and place the per-token biases. A
correct expert output requires `MoE.forward` to count the routed experts correctly **and**
`_experts_forward`/`swiglu` to split / clamp / bias / compute correctly — neither is correct in
isolation.

This exercises **pure MoE tensor logic**: small CPU tensors, **no GPU, no distributed backend,
no model weights, no device mesh**. The grading harness runs `MoE.forward` (with a fixed router
and a capture-only experts stub) and `GptOssGroupedExperts._experts_forward` directly, and
neutralizes the hardware grouped-matmul primitive `torch._grouped_mm` to a faithful CPU
segment-matmul (it is **not** part of your editable scope — do not try to "fix" it).

## Scope
- Edit **only** `torchtitan/models/common/moe.py` and `torchtitan/models/gpt_oss/moe.py`.
- Keep every public name and signature exactly as shipped. Do not change other functions, the
  module imports, or any other file. **Out-of-scope edits hard-fail (reward 0).**

## Required behavioural contract (must hold after your fix)

### `moe.py` — `MoE.forward`
1. **Routing map / per-expert counts.** The one-hot routing map marks **all top-k** experts
   each token is routed to (shape `(B, L, E)`, dtype bool), and the per-expert token counts
   `num_local_tokens_per_expert_E` are its sum over the batch and sequence dims (shape `(E,)`).
   For `top_k > 1`, every one of a token's selected experts must be counted.
2. **`tokens_per_expert_E` accumulation.** The running per-expert load counter
   **accumulates** the per-expert counts across forward calls (i.e. `+=`),
   so multi-step load statistics are preserved.

### `gpt_oss/moe.py` — `swiglu` and `GptOssGroupedExperts._experts_forward`
3. **`swiglu` gate/linear split.** The gate and linear halves are the **interleaved even/odd
   channels** of the last dimension (`x[..., ::2]` gate, `x[..., 1::2]` linear).
4. **`swiglu` linear bias of 1.** The linear branch carries an **extra +1 bias**: the output is
   `out_glu * (1 + x_linear)` (equivalently `addcmul(out_glu, out_glu, x_linear)`).
5. **`swiglu` glu clamp.** The **glu** branch is clamped one-sided (`max = limit` only); the
   **linear** branch is clamped two-sided (`[-limit, +limit]`).
6. **mlp2 output bias.** `_experts_forward` adds the mlp2 (down-projection) output bias back to
   the grouped-matmul result (after the tp-aware `ScaleBiasForward`), so the returned expert
   output includes the down-projection bias.

## How you are graded
A hidden suite of small deterministic CPU cases checks each contract point plus
end-to-end cases that run `MoE.forward` and feed its per-expert counts into
`_experts_forward` (coupling both files). Final reward is binary: 1 if the submission builds,
imports, passes every case, and satisfies the edit-scope and integrity checks; otherwise 0.

## Submission (single submission — this is a correctness task)

This task is graded on **correctness only**: your solution either satisfies every
hidden test case (full credit) or it does not (zero). There is **no speedup to
optimize and no iteration loop** — you get **exactly ONE submission**.

1. **Edit** `/app/repo/torchtitan/models/common/moe.py` and `/app/repo/torchtitan/models/gpt_oss/moe.py` until you believe it fully satisfies the behavioral
   contract above.
2. **Submit** by leaving your edits in the working tree — no `git commit` is needed
   (grading diffs the working tree against the baked baseline commit; the repo's HEAD
   must stay on that initial commit). Grading runs `tests/test.sh` against your
   `/app/repo` **once**; a correctness task does not reward iteration.

### 🔴 Red line (hard — ANY one of these ⇒ the whole task scores 0 in audit)

The scoring machinery is protected and off-limits. Solve this task
**independently**: rely ONLY on the code in `/app/repo`, your own knowledge, and
your own reasoning.

- Do **NOT deliberately** read, `cat`, `open`, `less`, `head`/`tail`, `grep`,
  `find`, or edit ANY verifier / correctness / scoring / hidden-test file or
  directory, wherever it lives. Leaving your edits in the working tree is the ONLY
  sanctioned way to submit.
- Do **NOT** run the verifier directly or try to reproduce / reverse-engineer
  it — leave your changes in `/app/repo` and let grading score them.
- Do **NOT** search for, print, or infer the hidden test cases, seeds,
  thresholds, or the metric definition from any source.
- Do **NOT** fetch or look up an upstream implementation or a reference
  solution (no web search, no `git` remote fetch, no network to the source),
  whether internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring /
  source material, however you came to see it.
- Do **NOT** route around the task by importing or calling a prebuilt equivalent of the function you are asked to implement; build the computation yourself in the scope file.

Stay inside `/app/repo` and scratch directories you created yourself; use only
what `submit` returns for scoring signal.

### 🎓 Explain for a beginner

Before you edit, explain your **approach** to the contract. After you have
finished editing and before you submit, give a short step-by-step walkthrough of
*why* your implementation satisfies each clause of the behavioral contract,
written so an undergraduate can follow.

## 提交方式

把改动**留在工作树里**即可 —— 直接编辑允许的文件,不需要 `git commit`
(判分读工作树;仓库的 HEAD 必须停在初始基线 commit 上)。
