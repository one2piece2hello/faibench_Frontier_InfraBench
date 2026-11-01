# Adapt a frozen language model to a new domain — with a hard budget on the bytes you may ship

You have the full [🤗 PEFT](https://github.com/huggingface/peft) library source at
`/app/repo` (fully editable), a **frozen pretrained base model** at `/app/base_model`
(Qwen2.5-0.5B-Instruct, bf16), one H20 GPU, and a training corpus of real
distributed-training systems Python source at `/data/datasets/a8_syscode/`.

Your job: **make that frozen base model as good as possible at this corpus's domain,
while shipping only a tiny adaptation artifact.**

The grader loads the base model **itself**, from its own read-only copy, and then asks
your code to turn it into your adapted model. So the *only* thing that travels from your
work to the score is the handful of bytes you are allowed to submit — everything else is
reconstructed from the untouched base weights.

## The budget (the rule that makes this hard)

At grading time exactly **two files** are carried over:

| file | what it is | cap |
|---|---|---|
| `/app/submission/adapter.bin` | your adaptation payload, any format you like | **262 144 bytes** |
| `/app/submission/adapter_entry.py` | the code that rebuilds your adapted model | **65 536 bytes** |
| — | **their combined size** | **327 680 bytes** |

The grader re-measures both files with `stat()` and rejects anything over budget — a
self-reported size is never trusted, and **code counts**, so a payload smuggled into the
`.py` as literals buys you nothing. Nothing else in the container is readable while your
`build_adapted_model` runs: `/app/submission`, `/app/repo` and the training corpus are all
moved aside for the eval window, and there is no network. Whatever your adapted model
knows must arrive inside those bytes (or already be in the frozen base weights).

## Scoring

The grader:

1. loads the frozen base model and measures its cross-entropy on a **held-out split of
   the same domain that you never see**;
2. loads a second, fresh copy of the frozen base model, calls your
   `build_adapted_model(...)`, and measures **your** model's cross-entropy on that same
   held-out split (median of three shifted passes, fp32 cross-entropy);
3. scores you on **how much of the achievable adaptation gain you captured**, relative to
   a strong baseline adapter trained under the *same* byte budget:

   ```
   gain_ratio = (base_CE - your_CE) / (base_CE - strong_baseline_CE)
   ```

   `0.0` means the adapter matches the frozen base; `1.0` means it matches the strong baseline.

   Your **reward is bounded to `[0.0, 1.0]`** and grows *logarithmically* in that ratio:

   ```
   reward = min(1.0, ln(gain_ratio / REF) / ln(REF))   if gain_ratio > REF   else 0.0
   ```

   `REF` is the reference recipe's calibrated gain ratio at the same byte budget. If the
   calibration is invalid (`REF <= 1`) or any validity check fails, the reward is zero.
   Matching the strong baseline (`gain_ratio = 1`) or the reference (`gain_ratio = REF`)
   also earns zero. Half credit requires `gain_ratio = REF^1.5`; full credit starts at `REF^2`.

Reaching a small positive gain ratio is easy: the provided starter recipe trains and submits a
valid adapter as-is — but a ratio at or under `REF` still scores **0**. Beating the reference is
hard — the reference recipe already uses a tuned learning rate and schedule, spends its entire byte
budget, and spreads its capacity over the highest-leverage projections, so simply raising the rank,
adding target modules, or training longer will not get you there. You have to make each byte carry
more adaptation.

## The entry contract

`/app/submission/adapter_entry.py` must define:

```python
def build_adapted_model(base_model, artifact_path, device) -> torch.nn.Module:
    ...
```

* `base_model` — a freshly loaded, **unmodified** copy of the frozen base model
  (`transformers` `Qwen2ForCausalLM`, bf16, already on `device`). Modify it, wrap it,
  replace submodules — whatever you need.
* `artifact_path` — the path to your `adapter.bin`.
* The returned module must be a real language model: `model(input_ids)` returns
  floating-point logits `(batch, seq_len, vocab_size)` that depend on token position
  (a `transformers`-style output with `.logits`, or a bare tensor). It is evaluated with
  `torch.inference_mode()` under bf16 autocast on one H20.
* `build_adapted_model` must return within **90 seconds** and gets no training data.

The grader also checks that the returned model is a genuine LM (it scores real held-out
text far better than arbitrary token ids, it has the base model's parameters in it, its
logits are position-dependent, and its held-out CE is inside a plausible band), and that
the base weights in `/app/base_model` are byte-identical to the frozen originals. A model
that fails any of these scores **0**.

## What you may change

Everything except the grader. `/app/repo` (the whole PEFT library), `/app/submission`, any
package in the environment, your training script, the adapter parameterisation, its
numeric encoding, the initialisation, the optimizer, the data order, how many tensors you
touch and which — the design space is deliberately wide, and the published
parameter-efficient-fine-tuning literature is fair game (this is a public research area;
you are graded on the result, not on novelty).

## What you are given

* `/app/base_model/` — the frozen base model + tokenizer (read-only in effect: it is
  digest-checked, so do not modify it).
* `/data/datasets/a8_syscode/train.txt` — your training corpus.
* `/data/datasets/a8_syscode/val.txt` — a **local** progress monitor from the same
  distribution. Use it to compare adapters; the held-out evaluation uses neither public file.
* `/app/submission/train_adapter.py` — a working but deliberately unambitious starter
  (rank-1 bf16 LoRA on two projections). Run it, beat it.
* `/app/submission/adapter_entry.py` — a reference loader for the starter's artifact
  format. Rewrite it freely.
* `/opt/peft_ro/src` — a read-only copy of the PEFT sources that stays importable during
  grading (your edits under `/app/repo` do **not** travel to the eval).

Suggested loop: train → check `adapter.bin`'s size and your local val CE → reconsider how
the budget is spent → retrain.

## Rules (violating any of these scores the whole task 0)

* Do **not** read, `cat`, `grep`, `find`, copy or edit any grader / hidden-eval /
  frozen-surface file, and do not try to infer the held-out split, the thresholds, or the
  reference recipe's numbers.
* Do **not** modify `/app/base_model` (it is digest-checked), and do not try to smuggle
  state into the eval through any path other than the two submitted files.
* Do **not** access the network at runtime, or work around the sandbox's proxy isolation.
* Solve the task yourself; consulting public documentation and literature is expected and
  allowed.

## Submission / 提交方式

In the default loop16 image, run `bash /opt/loop/submit.sh` once per candidate (1–16 submissions).
To stop before round 16, run `bash /opt/loop/submit.sh --finalize --reason "..."`; round 16 finalizes automatically.
The selection signal for passing rounds is public dev CE reduction (higher is better). Only the
selected adapter is evaluated on the held-out corpus; dev gain is not the final reward.
If you run the base image without the submit helper, the grader uses the final working tree instead.
请将适配器文件放在 `/app/submission/`，不要 `git commit`。默认镜像须用上述脚本提交；仅基础镜像直接评测最终工作树。
