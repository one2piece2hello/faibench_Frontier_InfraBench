# Performance Optimization Task

## Objective
A multi-LoRA batched apply subsystem (the shrink → expand path) in this repository is
functionally correct but slow. Make it **faster** on the benchmark workloads while
**preserving its numerical behavior** (outputs within a **2% relative tolerance**;
fp16 inputs, fp32 accumulation). All work must stay inside the declared editable scope.

## Editable scope
Edit **only** these three files (edits outside this scope score the submission zero):
```
vllm/lora/ops/triton_ops/kernel_utils.py
vllm/lora/ops/triton_ops/lora_shrink_op.py
vllm/lora/ops/triton_ops/lora_expand_op.py
```

## Entry points and contract
Two registered ops drive the subsystem:
- `lora_shrink(inputs, lora_a_weights, output_tensor, token_lora_mapping, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag_cpu, scaling)` — for each token, projects it through its assigned LoRA's A matrix (scaled). `output_tensor` `[num_slices, num_tokens, rank]` is written in place.
- `lora_expand(inputs, lora_b_weights, output_tensor, ...same routing..., offset_start=0, add_inputs=False)` — projects the shrunk buffer through the LoRA's B matrix into the output slice region (adding when `add_inputs`). `output_tensor` `[num_tokens, sum-of-slice-out-sizes]` is written in place.
- Each token carries a LoRA id via `token_lora_mapping` (`-1` = no LoRA); the routing tensors group tokens by LoRA id. `lora_a_weights[s]`: `[num_loras, rank, hidden]`; `lora_b_weights[s]`: `[num_loras, out, rank]`.

**Correctness requirement.** For every benchmark input, the outputs must match the
reference LoRA result within a **2% relative tolerance**. Any change altering the
result beyond that tolerance scores zero.

## Workload shape
The benchmark drives batched multi-LoRA application: thousands of tokens each routed
to one of several LoRA adapters, small adapter rank, shrink then expand. A hidden set
covers additional shapes.

**Final reward.** Let `s` be the median of the paired baseline-time / candidate-time ratios;
`r` is the calibrated speedup of the reference implementation over that same baseline.
Failing any required check, including correctness, earns 0. Even a valid run earns 0 if `r <= 1`
or `s <= r`. Only when you beat a valid reference does the log curve apply: `min(1, ln(s/r) / ln(r))`.
The reward reaches 1 at `s >= r²`.

## Rules
- Preserve the ops' signatures and behavior (within tolerance).
- Do **not** read, run, reproduce, or infer the scoring/verifier code, hidden test
  workloads, thresholds, or any reference solution.
- Do **not** download, clone, fetch, or otherwise obtain the upstream project, its
  history, or any external reference implementation, and do **not** bypass or disable
  the network isolation. Solve from the code present. Any such action scores the whole task zero.
- Write the optimization **in these three files**, using the libraries they already
  import (PyTorch / Triton). Do **not** copy or recover an implementation from any other
  copy of this package inside the container (for example an installed
  `site-packages`/`dist-packages` copy, a version-control object store, or another module
  of the repository), and do **not** load code from outside the declared scope at run time
  (`importlib`, `spec_from_file_location`, `exec`, or shelling out via
  `subprocess`/`os.system`/`os.popen`). A submission whose in-scope code does any of this
  scores zero.

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the performance of *this* task as far as you can. You have a
**submission budget of 1 to 16**: you may submit as few as **1** time and at
most **16** (hard ceiling). There is no minimum beyond a single submission and
no requirement to keep going — finalize as soon as you judge your solution good
enough.

**Iterate ONE round at a time, feedback-driven.** Each round is a single cycle:

1. **Edit** one of the in-scope files (`/app/repo/vllm/lora/ops/triton_ops/kernel_utils.py`, `/app/repo/vllm/lora/ops/triton_ops/lora_shrink_op.py`, `/app/repo/vllm/lora/ops/triton_ops/lora_expand_op.py`) — apply ONE concrete change based on your current
   hypothesis (round 1: based on reading the baseline code).
2. **Submit** by running `bash /opt/loop/submit.sh` **once** — it returns
   development feedback, including `dev_speedup`, `dev_reward`, `best_so_far` and the remaining budget.
3. **Read the feedback.** If the submission fails a development check, use the reported reason
   to guide your next edit. Otherwise compare `dev_speedup` across rounds; `best_so_far`
   identifies the submission currently retained by the development scorer.
4. **Analyze in writing** — (a) what the feedback told you, (b) *why* this
   round's change had that effect (mechanistic reasoning), (c) *what specific
   single change* the next round will try.
5. **Edit again based on that analysis**, then go to step 2.

Do **NOT** call `submit.sh` in a batch; do **NOT** plan more than one attempt
ahead — round `k+1`'s plan must depend on round `k`'s measured outcome. Edit
only product code; never touch tests/benchmarks/workloads/build files.

If the feedback ever says `harness_error`, that submission was refunded and
the failure is NOT your code's fault — do not debug the harness; retry or
continue optimizing.

### 2. Termination

Two ways to end:

- (a) reach 16 submissions (auto-finalize triggers as a safety net); or
- (b) at any k with **1 ≤ k < 16**, once you believe you have reached your
  practical/theoretical limit (e.g. the last few submissions converged within
  run noise), run
  `bash /opt/loop/submit.sh --finalize --reason "<one concrete sentence on
  why you are stopping>"` — the reason is recorded and audited.

There is no penalty for stopping after one or more submissions. Of the submissions that pass
all development checks, the harness selects one with the highest `dev_reward` for final grading.
Further rounds may change which candidate is selected; the development score is not the final reward.

### 3. 🔴 Red line (hard — ANY one of these ⇒ the whole task scores 0 in audit)

The scoring machinery is protected and off-limits. Solve this task
**independently**: rely ONLY on the code in `/app/repo`, the sanitized feedback
`submit` returns, and your own knowledge and reasoning.

- Do **NOT deliberately** read, `cat`, `open`, `less`, `head`/`tail`, `grep`,
  `find`, or edit ANY verifier / correctness / timing / scoring / hidden-test
  file or directory, wherever it lives. Executing `bash /opt/loop/submit.sh` is
  the ONLY sanctioned interaction with the scoring machinery.
- Do **NOT** run the verifier directly or try to reproduce / reverse-engineer
  it — score **only** by calling `bash /opt/loop/submit.sh`.
- Do **NOT** search for, print, or infer the hidden workloads, seeds,
  thresholds, the metric name, or the reference speedup from any source.
- Do **NOT** fetch or look up an upstream implementation or a reference
  solution (no web search, no `git` remote fetch, no network to the source),
  whether internet is on or off.
- Do **NOT** use, quote, or condition your code on any verifier / scoring /
  source material, however you came to see it.
- Do **NOT** call or reference the banned route-around tokens (`site-packages`, `dist-packages`, `importlib`, `spec_from_file_location`, `import_module`, `subprocess`, `os.system`, `os.popen`, `exec(`, `.git/`, `gated_gelu`, `ops.torch_ops`, `ops.xla_ops`) anywhere in your submitted code — the verifier scans your three scope files textually for these tokens and scores the task 0 on a hit. Build the shrink/expand path yourself, inside the scope files, from the code already present; do not dynamically load or copy an implementation from anywhere else in the container.

Stay inside `/app/repo` and scratch directories you created yourself; use only
what `submit` returns for scoring signal.

### 4. 🎓 Explain for a beginner — per round, not in bulk

Before your first edit, explain your **approach** (the hypothesis round 1
tests). Then **after every `submit.sh` feedback — and BEFORE the next edit —**
give a step-by-step analysis: what the result told you, *why* the change had
that effect, and *what single change* you will try next, written so an
undergraduate can follow.

## 提交方式

会话内用 `bash /opt/loop/submit.sh` 自评并拿反馈,完成后 `bash /opt/loop/submit.sh --finalize`。
改动留在工作树里,不需要 `git commit`。
