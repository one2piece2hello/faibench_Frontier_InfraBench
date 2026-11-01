# Adapt a frozen language model to a new domain — with a hard budget on the bytes you may ship (loop16)

You adapt a FROZEN pretrained base model (Qwen2.5-0.5B-Instruct, at `/app/base_model`, digest-checked
and read-only) to a new target domain while shipping at most **327,680 bytes** of adaptation —
`adapter.bin` (<= 262,144 B) + `adapter_entry.py` (<= 65,536 B), both re-measured by the harness. The
harness loads the base weights itself, hands them to your entry hook with your byte-capped artifact,
and scores held-out cross-entropy. It is EVAL-ONLY: the harness never re-runs your fine-tune.

The held-out quality metric is `s = (base_CE - candidate_CE) / (base_CE - strong_reference_CE)`.
Let `r` be the calibrated gain ratio of the reference ceiling recipe. A failed validity check,
`r <= 1`, or `s <= r` scores 0, including a tie with the ceiling recipe. Above it,
reward is `min(1, ln(s/r) / ln(r))`. All of `/app/repo` (peft) + `/app/submission` is editable
except the frozen eval surface.

## Entry contract (start from the shipped files)

- `/app/submission/adapter_entry.py` :: `build_adapted_model(base_model, artifact_path, device) ->
  nn.Module` — the grader calls exactly this, with a fresh UNMODIFIED base and your `adapter.bin`.
- `/app/submission/adapter.bin` — your byte-capped artifact (format is yours). Both files count
  toward the budget; keep the code small and put capacity in the artifact. Train your adapter with
  your own recipe in-session (e.g. via `train_adapter.py`); only the two files above are scored.
  Leave changes in the working tree; do not `git commit`.

---

## How you are graded (loop16: iterate in-session, at most 16 self-scored submissions)

This task uses the **loop16 protocol**: in your session you may score your current adapter and get
feedback **at most 16 times, at least once — you decide when to stop** (you need not use all 16).

### 1. Each round
After (re)building `/app/submission/{adapter.bin, adapter_entry.py}`, run

```
bash /opt/loop/submit.sh
```

It re-measures the declared-file byte budget, loads the frozen base + your adapter via
`build_adapted_model`, runs the anti-spoof self-checks (param floor, plausible CE band, logits vary,
OOD real << arbitrary), and reports the RAW **dev gain** (`base_CE - candidate_CE`) on a PUBLIC dev
corpus, best_so_far and the remaining budget.

🔴 **That dev gain is a PUBLIC proxy, not your graded score.** The overlay contains NO held-out corpus
and NO calibrated anchor. The grade normalises the gain against a HIDDEN `strong_baseline_ce` through
a calibrated log curve on a HIDDEN held-out split, and runs a whole-filesystem stash sweep the dev
proxy does not. The dev gain rises as your adapter improves (use it for **direction**), but the
absolute number is not the graded reward, and bytes stashed outside `adapter.bin`/`adapter_entry.py`
that pass the dev byte gate can still fail the grade's sweep. Do not overfit to the public corpus.

### 2. Finalize
When satisfied (or at the 16th round, which auto-finalizes) run

```
bash /opt/loop/submit.sh --finalize --reason "<one sentence on why you are stopping>"
```

`--reason` is mandatory and audited. The graded submission is your **best** correctness-passing round
(best-of-k, chosen by dev gain and planted for grading; `/app/repo` HEAD stays at the baked baseline
commit). A justified stop at k=1 scores the same as using all 16; there is no minimum and no "keep
going until N rounds without gain" rule.

## Red line (any one makes the whole task score 0)

- Do **not** read, `cat`, `grep`, `find`, copy or edit any verifier / hidden-test / evaluation file
  (under `/tests`, `/opt/loop/private`), the frozen base weights, or try to infer their contents —
  `bash /opt/loop/submit.sh` is the only sanctioned way to score.
- Do **not** run the grader directly, reproduce or reverse-engineer it.
- Do **not** exceed the byte budget or stash adaptation bytes outside the two declared files; do not
  read a side-channel file from the load hook (the grade quarantines every solver-writable path).
- Do **not** forge the loss, ship a degenerate/constant model, or tamper with the base weights.
- Do **not** fetch an upstream reference solution or access the internet at runtime. Public PEFT/LoRA
  methods are allowed; the MEASUREMENT is protected.
