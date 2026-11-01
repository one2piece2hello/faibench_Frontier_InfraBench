# vLLM Serving Optimization — Request Scheduling & Continuous Batching (H20) — loop16 protocol

You have a vLLM OpenAI-compatible serving instance serving a fixed language model on a single
**NVIDIA H20** GPU. Your goal: make it serve requests as fast as possible **without changing the
outputs**. To beat the strong, well-tuned baseline you must go **beyond configuration** — improve the
request-scheduling / continuous-batching subsystem itself (admission control, prefill/decode
interleave and chunk policy, batching order, preemption/requeue, per-step token-budget accounting),
write custom kernels, or modify vLLM source at `/app/repo`.

## Files & contract

- `/app/submission/launch_server.sh` — the entrypoint you own and modify; the harness runs it with
  `PORT` and `MODEL_PATH` and it must bring up an OpenAI-compatible server on `PORT` answering
  `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/completions`. Start from the shipped
  file; anything needed at scoring time must persist under `/app/submission/`.
- `/app/repo` — the full vLLM source tree, editable. `/app/model` — the fixed bf16 model.

## Correctness gate (runs before speed, at grading)

Greedy decoding (temperature 0) outputs must match the baseline **token by token** over hidden
prompts (normal, degenerate, adversarial). A scheduling change that corrupts request state changes
the outputs and scores 0 regardless of speed. Single-request latency must not materially regress.

---

## How you are graded (loop16: iterate in-session, at most 16 self-scored submissions)

Let `s` be your measured speedup over the strong baseline and `r` the calibrated reference
speedup over that same baseline. A failed validity check, `r <= 1`, or `s <= r` scores 0.
Otherwise the final reward is `min(1, ln(s/r) / ln(r))`, capped at 1 when `s >= r²`.

This task uses the **loop16 protocol**: in your session you may score the current
`/app/submission/launch_server.sh` (+ your `/app/repo` edits) and get feedback **at most 16 times, at
least once — you decide when to stop** (you need not use all 16).

### 1. Each round
After editing, run

```
bash /opt/loop/submit.sh
```

It launches your server (the SAME `launch_server.sh` the grade launches), confirms it serves the
**PUBLIC** dev prompts (a liveness/usability check), and returns: a server OK/FAIL, a **dev
throughput** (1000 / median single-request latency in ms on public prompts), best_so_far and the
remaining budget.

🔴 **That dev throughput is a COARSE PUBLIC proxy, not your graded score.** Grading uses HIDDEN
concurrent-burst workloads, an ABBA-paired median speedup against a strong baseline re-measured
in-session, and a **welded greedy token-parity gate** — none of which is in this container. The dev
proxy measures only single-request latency + server liveness on public prompts; it does NOT run the
concurrent bursts and CANNOT check token parity. So a server that is fast on the dev proxy but
changes greedy outputs, or that only helps single requests, can still score 0 at grading. Use the dev
number for **direction** (did the server get faster and still serve correctly?) — keep your greedy
outputs identical to the baseline.

### 2. Finalize
To stop after 1–15 submissions, run the command below. Round 16 finalizes automatically.

```
bash /opt/loop/submit.sh --finalize --reason "<one sentence on why you are stopping>"
```

Give a `--reason` when finalizing early. Among rounds that pass the dev checks, the one with
the highest public single-request throughput is selected for the held-out evaluation. There is no reward for using
more rounds; you may stop after the first submission.

## Constraints & red line (any one makes the whole task score 0)

- You may modify `launch_server.sh`, add helpers under `/app/submission`, modify vLLM source at
  `/app/repo`, use torch.compile / Triton / custom kernels, and install from the pre-built cache.
- Do **not** read, `cat`, `grep`, `find`, copy or edit any verifier / hidden-test file (under
  `/tests`, `/opt/loop/private`) — `bash /opt/loop/submit.sh` is the only sanctioned way to score.
- Do **not** disable, kill or tamper with the timer daemon or the measurement path.
- Do **not** access the internet at runtime.

Analyze GPU memory before changing configuration (`nvidia-smi`); an OOM kills the server and scores 0.
Keep a working `launch_server.sh` at all times.
