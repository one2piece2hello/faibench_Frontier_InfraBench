# Train the best language model you can from a FIXED NUMBER OF TRAINING TOKENS

You are given the full [nanoGPT](https://github.com/karpathy/nanoGPT) training system at
`/app/repo` (fully editable) and a tokenized WikiText-103 corpus. Your job is to **train the
best language model you can on ONE H20 GPU while consuming no more than a fixed budget of
training tokens**, then let the grader score it.

The grader then loads the checkpoint your training produced and measures its
**bits-per-byte (`val_bpb`) on a held-out WikiText-103 split you never see**. **Lower
`val_bpb` is better.**

## How you are scored

The shipped AdamW recipe is the baseline. Beating that starter is not enough:
you must beat the reference recipe at the same token budget.

For the final score, let `s = baseline_bpb / median_candidate_val_bpb` on the held-out
split and let `r` be the calibrated reference recipe's ratio at the same budget.
A failed validity check, an invalid reference (`r <= 1`), or `s <= r` scores 0.
Otherwise the reward is `min(1, ln(s/r) / ln(r))`: it reaches 1 at `s >= r²`.

Examples of invalid runs include exceeding the parameter cap, failing to save a checkpoint,
producing byte-identical weights across seeds, forging the loss, or touching the grader's files.

## The budget is TOKENS, not time — read this carefully

This is what makes this task different from a throughput task:

- The grader **counts the training tokens your run consumes** and enforces a hard cap. The
  count is taken by the grader from outside your process; it is not a number your code
  reports.
- **Wall-clock is NOT the budget and speed earns you NOTHING here.** Making your training
  step faster — mixed precision, `torch.compile`, a bigger batch, a faster data loader,
  fusing kernels — does not let you consume more tokens, so it cannot improve your score.
  (A generous wall-clock ceiling exists purely as a feasibility guard; see *Limits* below.)
- Therefore the only way to a lower loss is to **extract more learning from each token** —
  i.e. real **sample efficiency**.

## Why this is hard

The starting recipe is already a **well-tuned AdamW baseline** (tuned peak LR, warmup +
trapezoidal decay, gradient clipping, a batch size chosen at this token budget). The starter already includes the easy configuration wins.
To beat the reference, consider changes to the training algorithm, for example:

- **the optimizer / update rule** — e.g. spectral / orthogonalized updates for the hidden
  weight matrices (the `Muon` family), second-order-flavoured preconditioning, per-layer
  update scaling, or a different treatment of embeddings vs hidden layers vs the head;
- **the architecture** — normalization placement and type, rotary vs learned positions,
  query/key normalization, the MLP nonlinearity, initialization scale (including zero-init of
  residual projections), weight tying, skip/residual structure, logit scaling or softcapping;
- **the data schedule** — the order, packing, sequence length, and curriculum of the tokens
  you spend, and how the batch size or LR schedule co-varies with it;
- **the loss / regularization** — auxiliary losses, weight-decay discipline, label smoothing.

Public references are fair game: the nanoGPT repo, the
[modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) and its record
history, the [Muon writeup](https://kellerjordan.github.io/posts/muon/), and the general
literature on optimizers, architectures and training schedules. **What is protected is the
grader's measurement, not the method**. You may build on known techniques; your reward depends on
whether the resulting recipe beats the calibrated reference.

## Data

- Train on the token shard the grader hands you at `DATA_PATH` (flat `uint16`, nanoGPT
  layout). In your own session `/data/datasets/wikitext_bpe8k/train.bin` is available so you
  can develop and iterate; at scoring `DATA_PATH` points at the budgeted shard instead.
- `/data/datasets/wikitext_bpe8k/val.bin` is your **local progress monitor** (the WikiText-103
  *validation* split). Use it to compare recipes; final evaluation uses a separate, held-out *test* split.
- Tokenizer: SentencePiece **BPE-8K** at `/data/tokenizers/wikitext_bpe8k.model`
  (`vocab_size = 8192`).
- The corpus bytes are checksum-pinned by the grader, and the scored shard is cut from them.
  Training on anything else — other data, generated data, or a modified copy — is a hard
  failure.

## What to submit (the entry contract)

Everything the grader needs must live under `/app/submission/`, and the grader calls **two
functions** in `/app/submission/train_gpt.py`:

1. **`train_for_budget(seed, out_dir, budget_seconds)`** — your training entry. The grader
   calls it once per scored seed and requires it to write a checkpoint to
   `<out_dir>/model_ckpt.pt`. Save periodically: the grader stops you if you exceed the
   wall-clock guard, and the latest saved checkpoint is what gets scored.
   - The grader sets `DATA_PATH` in the environment to **your token budget** — a shard
     containing *exactly* `TOKEN_BUDGET` tokens. **During scoring this shard is the only
     readable token source** (the full corpus is locked for the duration of the run), so the
     budget is enforced by construction rather than by trusting a counter. `SEED`,
     `TOKEN_BUDGET`, `MAX_WALLCLOCK_SEC`, `MAX_PARAMS`, `TOKENIZER_PATH` and `NANOGPT_REPO`
     are also in the environment.
   - You may read the shard in **any order, packing, sequence length or curriculum**, and you
     may pass over it more than once — the budget is the *number of tokens made available*,
     and how you spend them is a lever. (Repeating tokens invites memorisation, which the
     held-out score punishes; that trade-off is yours to make.)
   - It is run from an isolated working directory that contains a copy of your **source**
     files only. Model-weight blobs are not carried over, and training happens fresh every
     time — so the checkpoint must be produced by this call.
2. **`load_model_for_verification(checkpoint_path, device)`** — returns either an `nn.Module`
   or `(args, model)`. The loaded model's `forward` must satisfy:
   - `model(input_ids)` → floating-point logits `(batch, seq_len, vocab_size)` that depend on
     token position;
   - `model(input_ids, target_ids)` → a **scalar** loss equal (within a small tolerance) to
     `F.cross_entropy(logits.reshape(-1, V), target_ids.reshape(-1))` on those same logits.

A working starting `train_gpt.py` (the AdamW baseline on nanoGPT's model, implementing
both functions) is provided — it is a starting point, **not** a contract. You may replace all
of it.

## Limits (all enforced by the grader — disclosed so nothing surprises you)

- **Token budget** — the hard cap named in `TOKEN_BUDGET`, counted by the grader. Exceeding it
  scores 0.
- **Parameter cap** — the grader re-counts the parameters of your loaded model and enforces the
  cap named in `MAX_PARAMS`. This is why "just make the model enormous" is not the answer; you
  must get more out of each token at a comparable model scale. There is also a minimum
  parameter count, so a degenerate tiny model cannot pass.
- **Wall-clock ceiling** — `MAX_WALLCLOCK_SEC` is a *feasibility guard*, not the budget. It is
  set generously (well above what the reference recipe needs) so that a slower-but-smarter
  recipe is never penalised; a run that exceeds it is stopped and its latest checkpoint scored.
- **Fixed seed** — the training seed is fixed by the grader, so you cannot seed-shop. Your
  recipe is scored as the **median over several grader-chosen seeds**, so it must be robust,
  not lucky.
- **Real-LM checks** — the grader verifies your loaded model is a real language model: it
  scores real held-out text noticeably better than arbitrary token IDs, its `forward(x,y)` loss
  agrees with the cross-entropy of its `forward(x)` logits, its logits vary across positions,
  and its `val_bpb` falls inside a plausible band. A degenerate or forged-loss model scores 0.
- **Fresh training** — the grader trains from scratch by running your entry; it does not accept
  pre-trained weights. Handing back a checkpoint you built earlier instead of training scores 0.
- No internet access. All packages you need (including `torch`) are pre-installed.

## Rules

- **Do not** read, copy, grep, or edit any grader / held-out / timer file (`/tests/*`); do
  **not** disable or tamper with the grader's token counter or wall-clock guard; do **not**
  try to read the held-out corpus or infer the reference score. The grader fingerprints its own
  files around every training call, so any of these scores the whole task **0**. (Consulting the public nanoGPT / modded-nanogpt / Muon references and the public
  training literature is explicitly allowed — see above.)
- Solve this independently. Everything needed at scoring must persist under `/app/submission/`.

## Submission / 提交方式

In the default loop16 image, run `bash /opt/loop/submit.sh` once per candidate (1–16 submissions).
To stop before round 16, run `bash /opt/loop/submit.sh --finalize --reason "..."`; round 16 finalizes automatically.
Of the candidates that pass the dev checks, the one with the lowest public dev `val_bpb` is selected for the
single held-out evaluation. The dev metric ranks candidates; it is not the final reward.
If you run the base image without the submit helper, the grader uses the final working tree instead.
请保持 `/app/repo` 的 HEAD 在初始提交，不要 `git commit`。默认镜像须用上述脚本提交；仅基础镜像直接评测最终工作树。
