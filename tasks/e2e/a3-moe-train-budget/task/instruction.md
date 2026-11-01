# Train the best Mixture-of-Experts language model you can within a fixed budget

You are given the full [nanoGPT](https://github.com/karpathy/nanoGPT) training
system at `/app/repo` (fully editable) and a tokenized WikiText-103 corpus. Your
job is to **train the best language model you can on ONE H20 GPU within a fixed
wall-clock budget, subject to a hard total-parameter floor**, then let the grader
score it:

- a **hard total-parameter FLOOR** — your model must have at least a required
  number of total parameters. The grader **re-counts your model's parameters** from
  the checkpoint and **rejects the run (score 0) if it is below the floor**. (A tiny
  model cannot win by being fast.)
- a **hard wall-clock budget** — the grader starts your training, runs it under a
  timer it owns, and stops the whole process group at the budget. Your training must
  periodically save its checkpoint so the latest save is scored.

The grader then loads the checkpoint your training produced and measures its
**bits-per-byte (`val_bpb`) on a held-out WikiText-103 split you never see**.
**Lower `val_bpb` is better.**

## How you are scored

The shipped plain dense recipe at the parameter floor is the baseline. Beating that
starter is not enough: you must beat the reference recipe at the same wall-clock budget.

For the final score, let `s = baseline_bpb / candidate_val_bpb` on the held-out
split and let `r` be the calibrated reference recipe's ratio at the same budget.
A failed validity check, an invalid reference (`r <= 1`), or `s <= r` scores 0.
Otherwise the reward is `min(1, ln(s/r) / ln(r))`: it reaches 1 at `s >= r²`.

Examples of invalid runs include falling below the parameter floor, failing to save a checkpoint
within the budget, submitting a degenerate or forged-loss model, or touching the grader's files.

## Why this is hard (the MoE lever)

The parameter floor puts you at a capacity where a **dense** model spends its full
compute on every token — so in a fixed wall-clock budget a dense model of that size
takes far too few optimizer steps and converges poorly. That is exactly what the
provided starter does, and it is why beating it is possible at all. The way to
train a floor-sized model well in the budget is a **sparse Mixture-of-Experts
(MoE)**: many expert feed-forward networks, but only a few (top-k) active per token,
so the compute per token — and therefore the number of steps you fit in the budget —
is governed by the **active** parameters, not the total.

But sparsity only helps if you **realize it as real single-card throughput**. The
levers that matter:

- **efficient token dispatch** — route each token to its top-k experts and compute
  each expert only on its routed tokens (sorted/grouped/batched GEMM, a per-expert
  capacity buffer). A naive loop that runs *every* expert on *every* token has dense
  FLOPs and gives no speedup — it will barely beat the dense starter. This is where
  most of the throughput is won or lost.
- **load balancing** — without a load-balancing objective the router collapses onto
  a few experts, wasting the floor capacity and overflowing expert buffers; a
  balancing auxiliary loss keeps experts utilized.
- **the routing frontier** — number of experts, top-k, and capacity factor trade
  quality against throughput and drop rate.

The design space is wide: you may change the model, router, dispatch, optimizer, LR
schedule, attention, data loading, precision — anything under `/app/repo` (nanoGPT)
and `/app/submission`. Public references (nanoGPT, the MoE literature — Switch
Transformer, GShard, MegaBlocks/dropless MoE, Tutel) are fair game.

## Data

- Train on `/data/datasets/wikitext_bpe8k/train.bin` (flat `uint16`, nanoGPT
  layout). `/data/datasets/wikitext_bpe8k/val.bin` is your **local progress
  monitor** (the WikiText-103 *validation* split); the grader scores on the
  held-out *test* split. Use local `val_bpb` to track progress, not to predict the final score.
- Tokenizer: SentencePiece **BPE-8K** at `/data/tokenizers/wikitext_bpe8k.model`
  (`vocab_size = 8192`). `meta.pkl` in the data dir carries `vocab_size`.
- The grader pins `DATA_PATH` and `TOKENIZER_PATH` itself; they are not taken from
  your environment.

## What to submit (the entry contract)

Everything the grader needs must live under `/app/submission/`:

1. **`/app/submission/run_training.sh`** — your training entry. The grader runs it
   with `SEED`, `PARAM_FLOOR`, `WALLCLOCK_SEC`, `OUT_CKPT`, `DATA_PATH`,
   `TOKENIZER_PATH`, `NANOGPT_REPO` in the environment. It **must write a checkpoint
   to `$OUT_CKPT`** within the budget (save periodically — the timer can stop you at
   any moment, and it stops your whole process group, so a detached background
   trainer buys you nothing).
2. **`/app/submission/train_gpt.py`** — must define
   `load_model_for_verification(checkpoint_path, device)` returning either an
   `nn.Module` or `(args, model)`. The loaded model's `forward` must satisfy:
   - `model(input_ids)` → floating-point logits `(batch, seq_len, vocab_size)` that
     depend on token position;
   - `model(input_ids, target_ids)` → a **scalar** loss equal (within a small
     tolerance) to `F.cross_entropy(logits.reshape(-1, V), target_ids.reshape(-1))`
     on those same logits. This is the **pure** cross-entropy of the logits — do
     **not** fold an MoE auxiliary (load-balancing) loss into this returned loss;
     add the auxiliary loss inside your training loop only.

   A working starting `train_gpt.py` — the dense-at-floor recipe described above —
   is provided. It is a starting point, **not** a contract; you may replace all of
   it.

The grader also enforces that your model is a **real LM**: it scores real held-out
text noticeably better than arbitrary token IDs, has at least the floor number of
parameters, its `forward(x,y)` loss agrees with the cross-entropy of its
`forward(x)` logits, and its `val_bpb` falls inside a plausible band. A degenerate
or forged-loss model scores 0.

## Rules

- Edit the allowed files under `/app/submission/` and `/app/repo`. Leave changes uncommitted;
  keep the repository HEAD at its initial commit. See the submission protocol below.
- The training seed is fixed by the grader (you cannot seed-shop).
- You do **not** have internet access; all packages you need are pre-installed.
- **Do not** read, copy, grep, or edit any grader / held-out / timer file
  (`/tests/*`); do **not** disable or tamper with the wall-clock timer; do **not**
  try to read the held-out corpus or infer the reference score. The grader
  fingerprints its own files around your run, so any of these scores the whole task
  **0**. (Consulting the public nanoGPT repo and the public MoE literature is
  allowed — what is protected is the grader's measurement, not the method.)

---

## How you are graded

In the default loop16 image, run `bash /opt/loop/submit.sh` once per candidate (1–16 submissions).
To stop before round 16, run `bash /opt/loop/submit.sh --finalize --reason "..."`; round 16 finalizes automatically.
Of the candidates that pass the dev checks, the one with the lowest public dev `val_bpb` is selected for the
single held-out evaluation. The dev metric ranks candidates; it is not the final reward.
If you run the base image without the submit helper, the grader uses the final working tree instead.

The final grader runs `/app/submission/run_training.sh` under its own wall-clock budget,
then loads the resulting checkpoint and measures held-out `val_bpb`.

At a fixed capacity
your real lever is **throughput** — how many optimizer steps you fit in the fixed
wall-clock budget — so watch tokens/sec as the primary signal that a
dispatch/precision/batching change made training faster, and use local `val_bpb` only for
coarse "is it still learning?" direction. The graded budget is wall-clock, so the number
of steps it buys drifts with GPU node contention; do not read much into a tiny `val_bpb`
change. `val.bin` is a PUBLIC proxy, NOT the graded score (the grade uses a held-out
split) — do not overfit it.

### 🔴 Red line (hard — ANY one ⇒ the whole task scores 0)

1. Do **NOT deliberately** read/`cat`/`grep`/`find`/edit ANY verifier / timing / scoring
   / hidden-test / held-out file or dir, wherever it lives.
2. Do **NOT** run the grader directly or reproduce/reverse-engineer it.
3. Do **NOT** search for, print, or infer the held-out corpus, the grader's seed, the
   parameter-floor value, the wall-clock budget value, the metric thresholds, or the
   reference score.
4. Do **NOT** fetch an upstream implementation or reference solution (no web search, no
   git remote fetch, no network). Public nanoGPT + MoE knowledge you already have is fine.
5. Do **NOT** use, quote, or condition your code on any verifier / scoring / held-out
   material, however you saw it.
