# vLLM Serving Optimization — Request Scheduling & Continuous Batching (H20)

You have a vLLM OpenAI-compatible serving instance serving a fixed language model
on a single **NVIDIA H20** GPU. Your goal is to make it serve requests as fast as
possible **without changing the outputs**.

The verifier launches your server using `/app/submission/launch_server.sh`, sends
hidden request workloads, and measures end-to-end latency against a **strong,
well-tuned vLLM baseline**. The baseline is already hardened on the scheduling side:
it uses CUDA graphs, chunked prefill with a tuned token budget, a tuned admission
width, prefix caching, and tuned GPU-memory allocation. Those configuration wins are
therefore already priced into the baseline — to beat it you must go **beyond
configuration**: improve the request-scheduling / continuous-batching subsystem itself
(admission control, how prefill and decode are interleaved and chunked, the batching
order, preemption/requeue, per-step token-budget accounting), write custom kernels, or
modify vLLM source.

Your score is a **bounded** function of your serving **speedup** over that baseline. Let

```
speedup = median over the held-out concurrent-burst workloads of ( baseline_time / your_time )
```

measured with paired baseline/candidate runs (the baseline is measured twice, before and
after your server, and the pair uses the mean of the two). Then

```
reward = min( 1.0 ,  ln(speedup / REF) / ln(REF) )   if speedup > REF   else 0.0    in [0.0, 1.0]
```

where `REF` is a constant fixed when this task was authored (the speedup a strong reference
solution achieved on this exact GPU). So:

- **`speedup <= REF` scores `0.0`.** Matching the strong baseline is worth nothing, and so is
  merely *tying the reference* — you have to **beat** `REF` before the reward leaves zero.
- `speedup == REF^1.5` ⇒ `0.5`.
- `speedup >= REF^2` ⇒ `1.0` (the cap).

**What is timed.** The score comes from **concurrent bursts of dissimilar requests** at high
concurrency (short interactive requests, long prefills, and sustained decode streams arriving
together, in several mixes and at several burst widths). Your optimization must help across the
**whole mix** — a single knob that helps one request shape while hurting another will not move the
median.

**Timing used only as a gate.** Sequential single-request latency is also measured, and your
server must not make it materially worse than the baseline's. You cannot win the bursts by
starving single requests: failing the latency gate scores the whole task `0.0`.

## How many times you may submit

In the default loop16 image, run `bash /opt/loop/submit.sh` once per candidate (1–16 submissions).
To stop before round 16, run `bash /opt/loop/submit.sh --finalize --reason "..."`; round 16 finalizes automatically.
Of the candidates that pass the dev checks, the one with the highest public single-request throughput is selected for the
single held-out evaluation. The dev metric ranks candidates; it is not the final reward.
If you run the base image without the submit helper, the grader uses the final working tree instead.

## Correctness gate (runs before speed)

The verifier runs your server and the baseline on the same hidden prompts with
**greedy decoding (temperature = 0)** and compares outputs **token by token**. The
prompt set includes normal text, degenerate inputs, and adversarial edge cases. Your
outputs must match the baseline's. A scheduling change that corrupts a request's state
(a dropped or duplicated sequence, a mis-restored preempted request, a wrong token
budget) will change the outputs and fail this gate. **If the correctness gate fails,
your score is zero regardless of speed.**

## Model

- A fixed instruction-tuned language model, bfloat16, pre-baked at `/app/model`.
- The tokenizer and weights are fixed; you may quantize/modify a copy, but outputs
  must still match the baseline under the greedy correctness gate.

## Files

- `/app/submission/launch_server.sh` — **the entrypoint you own and modify.** The
  verifier executes it to start your candidate server; it receives `PORT` and
  `MODEL_PATH` as environment variables and must bring up an OpenAI-compatible server
  on `PORT`.
- `/app/submission/` — your owned submission root. Anything needed at verification
  time (helper scripts, custom scheduler modules, vLLM source patches, kernels, config)
  **must live here** and be (re)applied by `launch_server.sh` at startup. Direct edits
  to site-packages alone may be lost during replay.
- `/app/repo` — the full vLLM source tree, editable. Find the installed package with
  `python3 -c "import vllm; print(vllm.__path__[0])"`.
- `/app/run_dev_bench.py` — a public dev benchmark (launches your server, sends a few
  requests, reports latency). The hidden verifier uses **different** workloads and
  more iterations, so a good dev number does not guarantee your score.

## The design space (anything that keeps outputs correct)

- **Request scheduling / continuous batching**: admission control (which and how many
  waiting requests to admit per step), the prefill/decode interleave and chunk policy,
  the batching / ordering strategy, preemption and requeue, and the per-step token-
  budget accounting. Public references you may consult: the vLLM `vllm/v1/core/sched/`
  scheduler code and the continuous-batching / chunked-prefill literature.
- **Server configuration**: CUDA graphs, chunked prefill, memory allocation, batch
  limits, quantization (must preserve greedy outputs).
- **Custom kernels / vLLM source modifications**: Triton or CUDA kernels; modify the
  installed vLLM source.

## Constraints

You CAN: modify `/app/submission/launch_server.sh` and add helper files under
`/app/submission`; modify vLLM source; use torch.compile / Triton / custom kernels;
install packages from the pre-built cache (no internet).

You CANNOT: access or reference `/tests/` or any hidden verifier file; disable or
tamper with the timer daemon; access the internet at runtime.

## Resource constraints

Analyze GPU memory before changing configuration (`nvidia-smi`, server logs). vLLM
pre-allocates KV-cache, CUDA-graph, and runtime buffers that compete with model
weights for H20 HBM. An out-of-memory error kills the server and scores zero. Verify
the server starts before committing to a configuration.

## Time budget

Your wall-clock budget is enforced and exposed through a timer daemon:

```bash
cat /app/.timer/remaining_secs
cat /app/.timer/elapsed_secs
```

Keep a working `/app/submission/launch_server.sh` at all times; leave time for a final
correctness + benchmark run. Everything needed to run your submission at verification
time must persist under `/app/submission/`.

## 提交方式

请保持 `/app/repo` 的 HEAD 在初始提交，不要 `git commit`。默认镜像用 `bash /opt/loop/submit.sh` 提交 1–16 次；仅基础镜像直接评测最终工作树。
