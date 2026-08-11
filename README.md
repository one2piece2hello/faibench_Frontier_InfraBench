<div align="center">

<h1>Φ-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?</h1>

*A reproducible benchmark for frontier LLMs & autonomous coding agents on **ML-systems / LLM-infrastructure engineering** — also written **Phi-Bench** / **ΦBench**.*

**85 open-source LLM-infrastructure engineering tasks** — build a public Docker image, solve the task offline, and score against a grader shipped with the package.

### 🔗 [llminfrabench.com](http://llminfrabench.com/)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
&nbsp;![tasks](https://img.shields.io/badge/tasks-85-brightgreen.svg)
&nbsp;[![website](https://img.shields.io/badge/website-llminfrabench.com-8A2BE2.svg)](http://llminfrabench.com/)

###### 🌐&nbsp; **English** &nbsp;·&nbsp; [简体中文](README.zh-CN.md)

</div>

---

Every task ships a **self-contained public Dockerfile** — `git clone` + `docker build` reproduces the environment, and the scoring surface is released with the package. All tasks are `allow_internet = false`: **both solving and scoring run offline**, so every dependency (including model weights and datasets) is baked into the image at `docker build` time.

| Subset | Tasks | Task type |
|---|---|---|
| `tasks/kfc/` | 55 | Excavate-and-reimplement: given a **functionally correct but slow** implementation, edit only the declared scope files and make it fast |
| `tasks/lh/` | 20 | Same idea, long-horizon (mostly kernels / protocol layers of large upstream libraries) |
| `tasks/e2e/` | 10 | Type-3 end-to-end: the whole working tree is editable, with only a small sha256-frozen scoring surface |

Machine-readable index: [`tasks_index.json`](tasks_index.json) (85 entries — package root / layout / GPU / scope / anchor / oracle availability per task). Scoring formulas: [`SCORING.md`](SCORING.md).

## Contents

- [Directory shape](#directory-shape)
- [How to run one task](#how-to-run-one-task)
- [Run with the bundled runner](#run-with-the-bundled-runner)
- [Two hard conventions](#two-hard-conventions)
- [Reproducibility of the excavated tree](#reproducibility-of-the-excavated-tree)
- [Self-check](#self-check)
- [License](#license)

## Directory shape

```
fai_bench/
├── tasks_index.json            index of 85 tasks
├── SCORING.md                  formulas for the two reward classes
├── scripts/verify_package.py   self-check: package structure + Dockerfile parseability + excavated-tree self-attestation
└── tasks/
    ├── kfc/<dir>/task/  ┐
    ├── e2e/<dir>/task/  ├ the three subsets place their **package root** differently — see package_root in tasks_index.json
    └── lh/<dir>/        ┘  (lh is flat; kfc/e2e have an extra task/ level)
```

The **package root** (i.e. the `docker build` context) contains exactly these, and nothing else:

```
<package_root>/
├── instruction.md        the only input visible to the model
├── task.toml             resources, scoring entrypoint, primary_metric, docker_image
├── .dockerignore         must live at the context root (docker does not read environment/.dockerignore)
├── environment/
│   ├── Dockerfile        ★ self-contained public recipe (public base + public sources + pinned versions)
│   ├── repo/             the excavated working tree (vendored; present for 75 tasks)
│   ├── runtime/          entrypoint.sh / timer.sh / run_dev_bench.sh
│   ├── loop/             in-session self-eval harness (76 tasks have it; 26 of those actually run 1–16 rounds,
│   │                     the other 50 are kfc with MIN=MAX=1, single submission — see SCORING.md)
│   └── …                 submission / dev_bench / stubs / workspace (per task)
├── tests/                scoring surface: test.sh + compute_reward.py + workloads + anchors
└── solution/             ★ reviewer-only: reference implementation / oracle patch (83 tasks have it)
```

## How to run one task

```bash
cd <package_root>
docker buildx build -f environment/Dockerfile -t <docker_image from task.toml> .
docker run --rm [--gpus all] -it <image>                                     # agent solves inside the container
docker run --rm [--gpus all] -v "$PWD/tests:/tests:ro" <image> bash /tests/test.sh
cat /logs/verifier/reward.json        # the reward field is this task's score
```

Every task's Dockerfile header spells out the four copy-paste commands (**build / run / score / re-calibrate**), along with that task's version anchors and working-tree provenance (the `PROVENANCE` block: which image digest it was restored from, by what method, and how it was verified).

**Network egress needed at build time** (varies by task; cross-check `tasks_index.json`): all tasks need apt and PyPI; 5 tasks need `git clone` from public GitHub; 5 e2e tasks pull model weights (~GB) from HuggingFace. **BuildKit is required**: the packages use `RUN <cmd> <<'PY' … PY` heredocs, so use Docker ≥ 23 + buildx (`docker buildx build`) — the classic builder cannot parse this syntax.

> Mirror-network note: 11 tasks write `pip install --index-url https://download.pytorch.org/whl/...`. `--index-url` **replaces** the primary index, so if your network can only reach an internal PyPI mirror, these 11 will fail to fetch torch — change it to `--extra-index-url` for compatibility with both networks (behavior is unchanged when public internet is available). The other 45 tasks already use `--extra-index-url` and are unaffected.

## Run with the bundled runner

`scripts/run_task.py` strings the manual commands above into one flow (build → agent → score → reward). It is **single-file and stdlib-only** (runs on any Python ≥ 3.8; below 3.11 it falls back to a built-in TOML parser, no `tomli` needed) and does build → start container → invoke agent → collect artifacts → mount `tests/` for scoring → aggregate reward in one command:

```bash
# Solve a task with claude-code / codex (the agent CLI is not in the image; it is injected at runtime)
python3 scripts/run_task.py --task tasks/kfc/<dir> --agent claude-code --model claude-opus-5 \
    --agent-bin /path/to/claude          # or --agent-install to install it inside the container
python3 scripts/run_task.py --task tasks/e2e/<dir> --agent codex --model gpt-5.6 --agent-install

# Two self-check paths that don't call a model:
python3 scripts/run_task.py --task tasks/lh/<dir> --agent oracle   # runs solution/solve.sh, should hit the task's reference score
python3 scripts/run_task.py --task tasks/kfc/<dir> --agent none     # scores the pristine baseline, should be ≈ 0

# Sample / full sweep:
python3 scripts/run_task.py --tasks-root tasks --n-tasks 10 --sample-seed 0 --agent claude-code
```

Key points:

- **The submission contract is "working-tree style,"** not `git commit`. The agent edits the allowed files directly and **leaves the changes in the working tree** — scoring reads the tree's diff against the baked-in baseline commit (`pre_artifacts.sh` captures it with `git add -AN` + `git diff HEAD`, without moving HEAD). This lets any file-editing scaffold plug in — swapping between claude-code / codex / mini-swe-agent needs no task change. **Do not let the agent commit** — once HEAD moves, the scope gate and the `git checkout HEAD -- <scope>` baseline capture both break, and a correct solution gets scored 0.
- **The agent CLI is injected at runtime**: the open-source images deliberately ship no agent (verified: zero hits across 91/91 Dockerfiles). `--agent-bin <host path>` mounts the CLI in read-only, or `--agent-install` installs a public npm package inside the container. GPT-family models use `codex`; everything else uses `claude-code`.
- **Network**: all tasks are `allow_internet = false`. Scoring is **always** `--network=none`; the agent step is offline by default too, opening only that agent's minimal allowlist when a model API must be called (`--agent-net proxy --net-proxy URL`, enforced by the proxy).
- **Mirror-network builds**: add `--build-network host` at build time so `RUN` steps can reach the PyPI/apt mirrors your host can see (off by default; not needed on public internet).
- **Outputs**: `runs/<task>-<agent>-<model>-<ts>/{build.log,agent.log,verify.log,artifacts/,verifier/,run.json}`, plus one line per task in `runs/summary.jsonl`.

The `--agent oracle` path doubles as the runner's own self-check: it invokes each task's `solution/solve.sh`, lands the reference implementation the same way an agent submission would (working tree, no commit), then runs normal candidate scoring — and should hit that task's reference score exactly.

## Two hard conventions

**1. `tests/` is released with the package but is NEVER baked into the image.** It is mounted at `/tests` only at scoring time. Reason: hidden cases, strong baselines, and calibration anchors all live inside — baking it in would make them readable/editable while solving. `solution/` is the same: reviewer-only, not in the image, not run during scoring.

**2. Performance anchors are calibration constants — changing hardware requires re-calibration.** The reward has the form `min(1, ln(speedup/ref_speedup)/ln(ref_speedup))` (**tie the ref_speedup and you get 0 — you must exceed it**; see [`SCORING.md`](SCORING.md)). `ref_speedup` is read-only at scoring time; the oracle is not re-run. 77 tasks have anchors; the calibration conditions are written in each task's `tests/ref_speedup.caveat.md` or the manifest's `hardware_caveat` field, with a copy-paste re-calibration command alongside. **Calibration runs on two channels**: GPU tasks on an NVIDIA H20, CPU tasks on the authors' CPU channel (Intel Sapphire Rapids) — each task records its own real calibration environment, so do not treat them as one unified setup.

```bash
# Patch form (most kfc / lh):
docker run --rm [--gpus all] -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/patches:ro" \
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
  <image> bash /tests/test.sh
# Single-file form (some tasks' reference is a variant of a whole file, not a patch):
  -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_FILE=/patches/kernel_oracle.py
```

Self-attestation: `noop` (no change) should score ≈ the no-op value; `negative` must score 0. **83 tasks ship a reference patch/oracle** and can use this path directly; only 2 tasks (`kfc/wro-offload-layer-prefetch-ring-pipeline-loop16`, `kfc/wro-offload-policy-grid-search-loop16`) do not, and their caveats note "anchor is not comparable across hardware — calibrate it yourself."

**Note the anchor resolution chain**: `tests/ref_speedup.txt` → in-image `/opt/verifier-correctness-manifest.json` → 1.0. The in-image copy **deliberately omits the real anchor** (it is readable by the solver), and `ref_speedup <= 1` is a hard gate, so **you must mount `tests/`** — otherwise it fails loudly with "anchor invalid" rather than emitting a wrong score. Also, `tests/ref_speedup.txt` is parsed with `tr -dc '0-9.'` — **do not add any comments to it**; any text containing digits or a decimal point will pollute the anchor value.

## Reproducibility of the excavated tree

For the kfc / lh subsets, the starting implementation is "an upstream library with a chunk excavated out." It is **not cloned** — the original image was assembled from a prebuilt tarball with no recorded upstream commit, so a clone cannot pin the scored bytes. Therefore `environment/repo/` is **restored from the original image and vendored**, and it is **self-attesting**:

> `git apply --check -p1 solution/oracle.patch` must apply cleanly **forward** onto `environment/repo/` and fail to apply in **reverse** — proving this tree is exactly the excavated baseline the reference patch was generated against. `scripts/verify_package.py` runs this gate per task.

When several tasks in one family share an upstream tree, **every** family member's scope files in that tree are held in the excavated state (otherwise one task's image would contain another's answer); a build-time assertion checks this.

The vendored tree is upstream code byte-for-byte — the third-party URLs, sample configs, even upstream's own committed internal proxy hints that appear inside it are all upstream content and, **per the self-attestation gate, must stay unchanged**.

## Self-check

```bash
python3 scripts/verify_package.py            # full self-check (85 tasks; archived delete_* tasks are auto-skipped)
python3 scripts/verify_package.py kfc lh     # check only some subsets
```

The self-check is **read-only** and derives all paths from the script's own location, so it runs anywhere you clone it. It verifies:
required files present · `task.toml` parses · `tests/test.sh` present and, for performance tasks, the anchor resolves (no silent fallback to 1.0) ·
**Dockerfile parses** (heredoc pairing, no dangling continuations, all `COPY` sources in the build context and not excluded by `.dockerignore`,
every `RUN` shell body passes `bash -n`) · anchor and caveat consistent · **excavated-tree self-attestation** (`oracle.patch` must apply forward
and fail in reverse) · no `__pycache__` / `*.bak` leftovers · **the runnable triad** (each task's `pre_artifacts.sh`
and `solution/solve.sh` exist, are executable, pass `bash -n`; `solve.sh` has a four-state CLI; `task.toml` is
`schema_version="2.0"`). `tasks_index.json` ships with the package; you do not need to rebuild it.

## License

fai_bench's **own work** — task specs (`instruction.md` / `task.toml`), graders (`tests/**`), reference solutions (`solution/**`), the loop16 harness (`environment/loop*/**`), the runner and self-check (`scripts/**`), and the documentation — is licensed under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).

The **vendored upstream code** under `environment/repo/` (nanoGPT, torchtitan, vLLM, llama.cpp, Megatron-LM, ColossalAI, flash-linear-attention, …) and the model weights / datasets fetched from public sources at build time (Qwen2.5, all-MiniLM-L6-v2, wikitext, …) **each retain their original license and copyright** and are **not** covered by this repository's Apache-2.0 grant — see [`NOTICE`](NOTICE). The `LICENSE` / `COPYING` file inside each vendored tree is its authoritative license.

---

<div align="center">

### Φ-Bench · Phi-Bench · ΦBench — LLM Infrastructure Engineering Benchmark

</div>

**About the name.** *Φ-Bench* (pronounced and also written **Phi-Bench** or **ΦBench**) is a benchmark that asks whether large language models and autonomous coding agents can engineer the ML systems infrastructure that powers LLMs themselves. Website & leaderboard: **[llminfrabench.com](http://llminfrabench.com/)**.

**Topics:** `llm` · `benchmark` · `llm-agents` · `coding-agents` · `ml-systems` · `llm-infrastructure` · `gpu` · `cuda-kernels` · `inference` · `training` · `quantization` · `agent-evaluation` · `leaderboard`

**What it covers.** GPU/CUDA kernel optimization, distributed training, inference & serving (vLLM), low-precision & quantization, communication/collectives, checkpointing & storage, MoE routing, attention & state-space kernels — 85 self-contained, Docker-reproducible tasks with offline graders and reference solutions.

**Cite as:**

```bibtex
@misc{faibench2026,
  title  = {$\Phi$-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?},
  author = {Φ-Bench contributors},
  year   = {2026},
  url    = {http://llminfrabench.com/}
}
```
