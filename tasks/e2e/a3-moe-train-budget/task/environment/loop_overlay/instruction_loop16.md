# Train the best Mixture-of-Experts language model you can within a fixed budget — loop16 protocol

You are given the full [nanoGPT](https://github.com/karpathy/nanoGPT) training system at `/app/repo`
(fully editable) and a tokenized WikiText-103 corpus on ONE **NVIDIA H20** GPU. Train the best
language model you can within a fixed **wall-clock budget**, subject to a **hard total-parameter
FLOOR** (the grader re-counts your model's total parameters and rejects a sub-floor model), then the
grader scores its **bits-per-byte (`val_bpb`) on a held-out split you never see** — lower is better.

At the required capacity a DENSE model is too compute-heavy to train well in the budget; a sparse
top-k Mixture-of-Experts trains far more steps at the same total capacity — IF you realize the
sparsity as real single-card throughput (sorted/grouped-GEMM token dispatch) with a balanced router
(load-balancing aux loss) and a tuned top-k / experts / capacity frontier. The shipped starter is a
plain dense model at the floor.

For the final score, let `s = baseline_bpb / candidate_val_bpb` on the held-out
split and let `r` be the calibrated reference recipe's ratio at the same budget.
A failed validity check, an invalid reference (`r <= 1`), or `s <= r` scores 0.
Otherwise the reward is `min(1, ln(s/r) / ln(r))`: it reaches 1 at `s >= r²`.

## Entry contract (start from the shipped files)

- `/app/submission/run_training.sh` — the harness runs it under a harness-owned wall-clock timer with
  `SEED`, `PARAM_FLOOR`, `WALLCLOCK_SEC`, `OUT_CKPT`, `DATA_PATH`, `TOKENIZER_PATH`,
  `NANOGPT_REPO=/app/repo` set. It MUST write a checkpoint to `$OUT_CKPT` within the budget (save
  periodically — the timer can stop you at any moment).
- `/app/submission/train_gpt.py` — must keep `load_model_for_verification(checkpoint_path, device)`
  importable, returning a model whose `model(x) -> logits (B,T,V)` and `model(x,y) -> scalar CE loss`
  (keep any load-balancing aux loss INSIDE your training loop, not in this forward). All of
  `/app/repo` (nanoGPT) and `/app/submission` is editable. Leave changes in the working tree; do not
  `git commit`.

---

## How you are graded (loop16: iterate in-session, at most 16 self-scored submissions)

This task uses the **loop16 protocol**: in your session you may score your current recipe and get
feedback **at most 16 times, at least once — you decide when to stop** (you need not use all 16).

### 1. Each round
After editing, run

```
bash /opt/loop/submit.sh
```

It trains your recipe on a **SMALL PUBLIC dev shard** of the corpus for a **SHORT dev wall-clock**
(far below the graded budget), under the same uid-drop the grade uses, then reloads the checkpoint
and runs a first-principles self-check (total-param floor, forward CE agreement, logits vary,
plausible band) and reports a **dev `val_bpb`** on the PUBLIC val split, best_so_far and the remaining
budget.

🔴 **That dev `val_bpb` is a PUBLIC proxy, not your graded score.** The overlay contains NO held-out
corpus and NO calibrated anchor. Unlike some siblings, this task's budget is WALL-CLOCK, so the dev
`val_bpb` depends on both your recipe AND this node's speed and is only a COARSE **direction** signal
(a more efficient recipe should reach a lower `val_bpb` in the short dev budget). The graded run trains
at the FULL budget on the HIDDEN split with the calibrated anchor. Do not overfit to the dev shard.

### 2. Finalize
When satisfied (or at the 16th round, which auto-finalizes) run

```
bash /opt/loop/submit.sh --finalize --reason "<one sentence on why you are stopping>"
```

`--reason` is mandatory and audited. The graded submission is your **best** correctness-passing round
(best-of-k, chosen by dev `val_bpb` and planted for grading; `/app/repo` HEAD stays at the baked
baseline commit). A justified stop at k=1 scores the same as using all 16; there is no minimum and no
"keep going until N rounds without gain" rule.

## Red line (any one makes the whole task score 0)

- Do **not** read, `cat`, `grep`, `find`, copy or edit any verifier / hidden-test / evaluation file
  (under `/tests`, `/opt/loop/private`), and do not try to infer their contents — `bash
  /opt/loop/submit.sh` is the only sanctioned way to score.
- Do **not** run the grader directly, reproduce or reverse-engineer it.
- Do **not** search for / print / infer the hidden test split, the fixed seeds, the wall-clock budget,
  any threshold, the baseline `val_bpb` or the reference score.
- Do **not** forge the loss, smuggle a pre-built checkpoint, train on anything other than the shard at
  `DATA_PATH`, or tamper with the parameter / budget / uid guards.
- Do **not** fetch an upstream reference solution or access the internet at runtime. The public method
  (modded-nanogpt, Muon, MoE routing literature) is allowed; the MEASUREMENT is protected.
