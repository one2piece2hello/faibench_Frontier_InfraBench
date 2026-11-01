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
of it, and you may edit the nanoGPT source in `/app/repo` your recipe imports.

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

---

## 🔁 Iterative submission protocol (this task uses a per-task loop)

### 1. Goal + protocol

Optimize the held-out `val_bpb` of *this* task as far as you can inside the token budget. You
have a **submission budget of 1 to 16**: at least **1** submission, at most **16** (hard
ceiling). Iterate **one round at a time, feedback-driven**. Each round is a single cycle:

1. **Edit** your recipe — `/app/submission/train_gpt.py` and/or the nanoGPT source in
   `/app/repo` it imports — applying ONE concrete change based on your current hypothesis
   (round 1: based on reading the shipped starter).
2. **Submit** by running `bash /opt/loop/submit.sh` **once**. It trains your recipe on a
   **small public token budget** and returns, synchronously, a sanitized block:
   `correctness` (PASS/FAIL) · `dev_val_bpb` · `best_so_far` · `remaining` ·
   `finalize_allowed` · and on failure the **named `failing_invariant`**.
3. **Read the feedback in full**, then **analyze in writing** (see §4).
4. **Edit again** based on that analysis, then go to step 2.

Do **NOT** call `submit.sh` in a batch, and do **NOT** plan more than one round ahead — round
`k+1`'s change must depend on round `k`'s measured outcome.

🔴 **`dev_val_bpb` is a PUBLIC PROXY, not your graded score.** It is measured by training your
recipe on a *small public* token budget (sliced from the same public `train.bin` you already
have) and evaluating on the *public* `val.bin` monitor — nothing about the held-out test split
or the reference recipe lives in this container. Because the proxy budget is **tokens, not
wall-clock**, the number is stable and does not drift with node speed. Use it for **direction**
(did this change lower `val_bpb` or not?), not as your grade: your real score is computed after
the session, over the **hidden** test split at the **full** token budget, with the frozen
anchor. Do not overfit to the public proxy.

If the feedback ever says `harness_error`, that round was **refunded** (your budget is
unchanged) and the failure is NOT your code's fault — do not debug the harness; retry or
continue optimizing.

### 2. Termination

Two ways to end:

- (a) reaching 16 submissions triggers an automatic finalize (a safety net); or
- (b) at **any k ≥ 1**, once you judge you have reached your practical/theoretical limit (e.g.
  the last few rounds converged within run noise), run
  `bash /opt/loop/submit.sh --finalize --reason "<one concrete sentence on why you are stopping>"`.

Give a `--reason` when finalizing early. Among rounds that pass the dev checks, the one with
the lowest dev `val_bpb` is selected for the held-out evaluation. There is no reward for using
more rounds; you may stop after the first submission.

### 3. 🔴 Red line (hard — hitting ANY one ⇒ the whole task is audited to 0)

1. Do **NOT deliberately** `cat` / `open` / `less` / `head` / `tail` / `grep` / `find` / edit
   **any** verifier / correctness / timing / scoring / hidden-test file or directory, wherever
   it lives — `bash /opt/loop/submit.sh` is the **only** sanctioned way to interact with the
   scoring machinery.
2. Do **NOT** run the grader directly, and do **NOT** reproduce or reverse-engineer it — score
   only by calling `bash /opt/loop/submit.sh`.
3. Do **NOT** search for, print, or infer the hidden test split, the fixed seeds, the token
   budget, any threshold, the metric internals, the baseline `val_bpb`, or the reference score.
4. Do **NOT** fetch or look up an upstream reference solution (no web fetch, no `git` remote
   fetch, no network), regardless of any internet switch.
5. Do **NOT** use, quote, or condition your code on any verifier / scoring / held-out material,
   however you came to see it — and do not forge the loss, smuggle a pre-built checkpoint, train
   on anything other than the shard at `DATA_PATH`, or tamper with the parameter/token/uid
   guards.
6. **One round at a time:** do not batch submissions and do not plan more than one round ahead.
7. Between rounds you **must** write the analysis in §4.
8. `--reason` on `--finalize` is **mandatory** and is audited.

Stay inside `/app/submission` + `/app/repo` and scratch directories you create yourself; use
only what `submit.sh` returns for scoring signal.

### 4. 🎓 Explain for a beginner — per round, not in bulk

Before your first edit, state your **hypothesis** (what round 1 tests and why). Then, **after
every `submit.sh` feedback and BEFORE the next edit**, give a step-by-step analysis an
undergraduate could follow: (a) what the feedback told you, (b) *why* this round's change had
that effect on `val_bpb` (mechanistic reasoning about sample efficiency — the optimizer /
architecture / data schedule), and (c) *what single change* you will try next.
