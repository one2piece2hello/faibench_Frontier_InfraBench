# syntax=docker/dockerfile:1
# =============================================================================
# e2e-vllm-scheduler-mixed-batch-serving — OPEN-SOURCE, self-contained task image
#
#   Task: maximise mixed-batch serving throughput of an OpenAI-compatible vLLM server.
#   The agent owns /app/submission (starting point: launch_server.sh) and may also edit the
#   editable vLLM tree; the OPEN verifier under /tests runs an ABBA paired benchmark
#   (baseline server -> candidate server -> baseline re-check) and scores log-speedup.
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup is a FROZEN constant read
#   from tests/reward_manifest.json (-> ref_speedup.value) and was calibrated on NVIDIA H20 (see
#   that file's calibrated_on + hardware_caveat). The scorer NEVER runs the oracle.
#   This is a SERVING-type ABBA paired benchmark (baseline server -> candidate server -> baseline
#   re-check), so to re-measure the anchor you run a FULL scored pass with the reviewer oracle
#   standing in as the candidate, and take the median over >= 3 runs:
#     1. mount the reviewer solutions:  -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/opt/negative:ro"
#     2. put the oracle launch policy in place of /app/submission/launch_server.sh, then run
#        `bash /tests/test.sh`
#     3. read the `speedup` median out of /logs/verifier/ and write it into
#        tests/reward_manifest.json -> ref_speedup.value (also update status / calibrated_on)
#   Anchors on serving tasks are node-sensitive (cross-launch variance dominates), so DO NOT reuse
#   the shipped constant on different hardware without re-measuring.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image, no
#   vendored model weights (Qwen2.5-1.5B-Instruct is pulled from Hugging Face at BUILD time).
#   Platform: linux/amd64 ONLY (the official vLLM wheel is manylinux x86_64).
#
# BUILD   (context = task/; network required — downloads the model + wheels. ~3 GB of weights)
#     cd <...>/vllm-scheduler-mixed-batch-serving/task
#     docker build -f environment/Dockerfile -t fai/vllm-scheduler-mixed-batch-serving:oss .
#   Non-H20 GPUs: --build-arg TORCH_CUDA_ARCH_LIST=8.0   # A100 (8.9 L40S · 9.0 H100/H20 · 10.0 B200)
#   Behind a HF mirror:  --build-arg HF_ENDPOINT=https://hf-mirror.com
# RUN     (needs 1 NVIDIA GPU, driver >= 535)
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -it fai/vllm-scheduler-mixed-batch-serving:oss
# SCORE   (inside the container; tests/ is MOUNTED, never baked)
#     bash /tests/test.sh                        # -> /logs/verifier/reward.json
#
# VERSION ANCHORS: CUDA 12.6 · Python 3.10 · torch 2.7.1 (cu126) · triton 3.3.1 · vllm 0.10.1.1
#   vLLM 0.10.1.1 hard-requires torch==2.7.1; the constraint file pins torch/triton so vLLM's
#   resolver can never move them (its compiled _C.so must match the torch ABI).
#
# ⚠️ ref_speedup in tests/reward_manifest.json was CALIBRATED ON H20. On other hardware the
#   anchor is not comparable — see that file's hardware_caveat / recalibrate_howto.
#
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
#   (That version FROM'd a private vLLM-serving base, COPY'd pre-staged model weights from the
#    build context and injected an internal agent CLI; all three are removed here.)
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG HF_ENDPOINT=https://huggingface.co
ARG MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TASK_BUDGET_SECS=14400

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-10 g++-10 clang cmake ninja-build pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
        git git-lfs curl wget ca-certificates procps psmisc coreutils \
        tar gzip unzip xz-utils vim less file jq tini \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

# ---- venv. Keep this exact path: tests/test.sh pins PATH=/opt/kernelbench-venv/bin ----
ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
RUN pip install --no-cache-dir -U pip setuptools wheel

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.7.1" "triton==3.3.1"

# ---- vLLM from the OFFICIAL PUBLIC WHEEL, WITH its dependencies. constraint pins torch/triton.
#      No `|| true`: a failed dependency install must fail the build.
RUN printf 'torch==2.7.1\ntriton==3.3.1\n' > /tmp/vllm-constraints.txt \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        --constraint /tmp/vllm-constraints.txt "vllm==0.10.1.1" "transformers>=4.55,<5" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__; print('torch pin held:', torch.__version__)"

# ---- served model weights from Hugging Face (public, Apache-2.0). NOT vendored in git. ----
ENV HF_ENDPOINT=${HF_ENDPOINT} HF_HUB_DISABLE_TELEMETRY=1
RUN pip install --no-cache-dir "huggingface_hub[cli]>=0.24,<1" \
 && huggingface-cli download "${MODEL_ID}" --local-dir /app/model --exclude "original/*" \
 && test -f /app/model/config.json \
 && ls /app/model/*.safetensors >/dev/null \
 && du -sh /app/model

# ---- editable vLLM tree.
#      This task's verifier restores the pristine snapshot into SITE-PACKAGES
#      (`tar xf /app/.vllm-baseline.tar -C "$SITE_PKG"`), so site-packages stays the
#      authoritative import location and /app/repo/vllm is a COPY that serves as the
#      git diff base for "what did the agent change".
#      NOTE: no inline `#` comments inside this RUN — shell-form RUN joins the \-continued
#      lines into ONE command, where a `#` would comment out everything after it.
RUN set -eu \
 && SITE_PKG="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__path__[0]))')" \
 && echo "SITE_PKG=$SITE_PKG" \
 && echo "$SITE_PKG" > /app/.vllm-site-packages-path \
 && tar cf /app/.vllm-baseline.tar -C "$SITE_PKG" vllm \
 && chmod 0644 /app/.vllm-baseline.tar \
 && rm -rf /app/repo && mkdir -p /app/repo \
 && cp -a "$SITE_PKG/vllm" /app/repo/vllm \
 && printf '__pycache__/\n*.pyc\n*.so.tmp\n' > /app/repo/.gitignore \
 && git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo && git init -q && git add -A && git commit -qm "vllm baked baseline" \
 && rm -rf /app/repo/.git/logs \
 && test "$(git -C /app/repo rev-list --count HEAD)" = "1" \
 && test -z "$(git -C /app/repo remote)" \
 && echo "GIT baseline OK commits=1"

# ---- solver-owned submission + runtime contract (entrypoint/timer/profile shipped in-package) ----
COPY submission/ /app/submission/
COPY environment/entrypoint.sh /app/entrypoint.sh
COPY environment/timer.sh /app/timer.sh
COPY scripts/run_dev_bench.py /app/run_dev_bench.py
COPY instruction.md /app/instruction.md
RUN chmod +x /app/entrypoint.sh /app/timer.sh /app/submission/launch_server.sh \
 && mkdir -p /app/.timer /logs/agent /logs/verifier /logs/artifacts \
 && find /app -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
RUN test -z "$(find /app -name '__pycache__' -o -name '*.pyc' 2>/dev/null)" \
 && echo "OK: zero __pycache__ / *.pyc under /app"
COPY environment/frontier-task-init.sh /etc/profile.d/frontier-task-init.sh
RUN chmod 0644 /etc/profile.d/frontier-task-init.sh
ENV BASH_ENV=/etc/profile.d/frontier-task-init.sh

# ---- OPEN grading suite is NOT baked: tests/ (test.sh + compute_reward.py + prompts.jsonl +
#      reward_manifest.json with the H20-calibrated ref_speedup) ships in this package and is
#      MOUNTED at /tests when scoring, so the solver cannot read or edit it while solving.
RUN rm -rf /opt/verifier

WORKDIR /app

# ---- build-time smoke. HARD: model + submission + runtime files + git base.
#      SOFT: `import vllm` (a GPU-less build host can fail vLLM's platform probe).
RUN set -eu \
 && python3 -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)" \
 && test -f /app/model/config.json \
 && test -x /app/submission/launch_server.sh \
 && test -x /app/entrypoint.sh && test -x /app/timer.sh \
 && test -f /app/.vllm-baseline.tar \
 && test "$(git -C /app/repo rev-list --count HEAD)" = "1" \
 && ( python3 -c "import vllm; print('vllm', vllm.__version__, 'from', vllm.__file__)" \
      || echo "WARN: import vllm failed at BUILD time (often needs a GPU/driver); verify at runtime" ) \
 && echo "BUILD-ASSERTS OK"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/bin/bash"]
