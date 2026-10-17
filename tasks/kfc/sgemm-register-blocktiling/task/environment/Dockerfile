# syntax=docker/dockerfile:1
# =============================================================================
# sgemm-register-blocktiling — OPEN-SOURCE, self-contained task image
#
#   Task: make the declared scope faster while preserving its observable behaviour exactly.
#         Editable scope (the ONLY files the scope gate allows to change):
##     - /app/repo/sgemm_kernel.cu
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   Needs 1 NVIDIA GPU to RUN (task.toml: gpus = 1, anchor calibrated on H20); no GPU to BUILD.
#   Platform: linux/amd64 ONLY (official torch wheels are manylinux x86_64).
#   Build context = this task directory (the Dockerfile lives in environment/).
#
# PROVENANCE of environment/repo/ (so anyone can re-derive it independently)
#   recovered from: <internal registry>/kernelbench/insea-sgemm-register-blocktiling-loop16:loop-20260720T180039Z@sha256:801dfb5d08e671fb8f4d43f42f080444a44e12c31fb663be465c716a502b6c84
#   method: `crane manifest` -> the layers under 80 MB -> `crane blob` + tar, overlaid in
#           manifest order; .git and any .wh..wh..opq marker dropped.
#   verification: `git apply --check -p1 oracle.patch` succeeds against these bytes and `-R` does NOT — i.e. this tree is the exact baseline tree the reference patch was generated against.
##
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network for PyPI only — the working tree ships in this package)
#     cd <...>/sgemm-register-blocktiling/task
#     docker build -f environment/Dockerfile -t fai/kfc-sgemm-register-blocktiling:oss .
#
# RUN     docker run --rm --gpus all -it fai/kfc-sgemm-register-blocktiling:oss
# SCORE   docker run --rm --gpus all -v "$PWD/tests:/tests:ro" fai/kfc-sgemm-register-blocktiling:oss bash /tests/test.sh
#         # -> /logs/verifier/reward.json   (modes: candidate | noop | oracle | negative)
#
# 🔴 RE-CALIBRATE THE ANCHOR ON YOUR HARDWARE
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; ref_speedup = 2.99 .
#   tests/test.sh resolves it from tests/ref_speedup.txt FIRST, then falls back to
#   /opt/verifier-correctness-manifest.json, then to 1.0 (which would make every reward 0).
#   The value was calibrated on NVIDIA H20. The verifier supports mode dispatch natively, so
#   re-measuring is ONE command (the reference patch is passed BY PATH and never baked):
#     docker run --rm --gpus all -v "$PWD/tests:/tests:ro" -v "<dir holding oracle.patch>:/patches:ro" \
#       -e KERNELBENCH_VERIFY_MODE=oracle -e KERNELBENCH_ORACLE_PATCH=/patches/oracle.patch \
#       fai/kfc-sgemm-register-blocktiling:oss bash /tests/test.sh
#   then write the measured speedup into tests/ref_speedup.txt. See tests/ref_speedup.caveat.md.
#   Sanity: noop (no patch) ~= 1.0 -> gated to 0; negative MUST score 0.
#
# VERSION ANCHORS
#     CUDA 12.6 · Python 3.10 · torch 2.7.0 (cu126, brings its matching triton 3.3.x)
#   The original private base declared torch 2.5.1 / triton 3.1.0 and the task overlay then ran
#   `pip install torch==2.7.0` — 2.7.0 is therefore the version the anchor was calibrated under.
# =============================================================================
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Compute capability you will RUN on (Triton JITs for the live device; this only affects
# hand-built CUDA extensions): 8.0 A100 / 8.9 L40S-4090 / 9.0 H100-H20 / 10.0 B200
ARG TORCH_CUDA_ARCH_LIST="9.0"
ARG TORCH_VERSION=2.7.0

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-10 g++-10 cmake ninja-build pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
        git ca-certificates curl procps coreutils tar gzip vim less file jq \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

# ---- venv at this EXACT path: tests/test.sh puts /opt/kernelbench-venv/bin at the head of PATH,
#      and a bare python3 from anywhere else would not have torch. ----
ENV VENV=/opt/kernelbench-venv
RUN python3.10 -m venv "$VENV"
ENV PATH=$VENV/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
        "torch==${TORCH_VERSION}" \
 && python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__); assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- the graded working tree. A REAL directory (not a symlink): the scope gate runs
#      `git -C /app/repo status --porcelain`, which cannot see through a symlink. ----
RUN mkdir -p /app/repo /app/.timer /app/workspace /logs/verifier /logs/agent /logs/artifacts
COPY environment/repo/ /app/repo/
COPY instruction.md /app/instruction.md

# ---- the baseline commit = the diff base the scope gate compares against.
#      🔴 core.fileMode=false is LOAD-BEARING, not hygiene: a shipped image in this corpus had
#      core.fileMode=true while some files were 0775 on disk but 100644 in the index, so
#      `git status --porcelain` reported phantom out-of-scope paths in EVERY mode and forced
#      reward to 0. Setting it false at creation makes that class of defect impossible. ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && git init -q \
 && git config core.fileMode false \
 && printf '__pycache__/\n*.pyc\n' >> .gitignore \
 && git add -A \
 && git commit -qm "baseline (graded starting point)" \
 && rm -rf .git/logs \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && test "$(git config --get core.fileMode)" = "false" \
 && test -z "$(git status --porcelain)" \
 && echo "[build] /app/repo: 1 commit, no remote, fileMode=false, clean tree"

# ---- runtime contract (copied verbatim from the shipped image) ----
COPY environment/runtime/entrypoint.sh /app/entrypoint.sh
COPY environment/runtime/timer.sh /app/timer.sh
RUN chmod 0755 /app/entrypoint.sh /app/timer.sh && chown root:root /app/entrypoint.sh /app/timer.sh
COPY environment/runtime/run_dev_bench.sh /app/workspace/run_dev_bench.sh
RUN chmod 0755 /app/workspace/run_dev_bench.sh
COPY environment/loop/ /opt/loop/
RUN if [ -f /opt/loop/submit.sh ]; then chmod 0755 /opt/loop/submit.sh; fi \
 && chown -R root:root /opt/loop \
 && if [ -f /opt/loop/score_engine.sh ]; then chmod 0700 /opt/loop/score_engine.sh; fi \
 && if [ -f /opt/loop/sanitize_feedback.py ]; then chmod 0700 /opt/loop/sanitize_feedback.py; fi \
 && if [ -d /opt/loop/private ]; then chmod -R go-rwx /opt/loop/private; fi

# ---- the FROZEN anchor fallback. tests/test.sh prefers tests/ref_speedup.txt (mounted) and falls
#      back to THIS path; keeping both consistent makes the resolution explicit. Only the constant. ----
COPY environment/verifier-correctness-manifest.json /opt/verifier-correctness-manifest.json

# ---- OPEN grading suite is NOT baked: tests/ (test.sh + compute_reward.py + the workload/harness
#   + ref_speedup.txt) is MOUNTED FRESH at /tests when scoring. It ships in this package —
#   open-source, but outside the image so the graded harness cannot be edited from the session.

# ---- 🔴 ANTI-LEAK ASSERTS (author-side artifact denylist) ----
RUN set -eu; rc=0; \
    hits="$(find / -xdev -type f \( -name 'oracle*.patch' -o -name 'negative*.patch' \
              -o -name 'baseline2*.patch' -o -name 'scope_card.json' -o -name 'kernel_oracle.py' \
              -o -name 'degradation_manifest.json' \) 2>/dev/null | head -10)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FAIL: a reference/authoring file is baked:"; echo "$hits"; rc=1; fi; \
    for p in /opt/verifier /opt/strong_baseline /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FAIL: frozen-surface path baked: $p"; rc=1; fi; \
    done; \
    test "$rc" = 0; \
    echo "[build] ANTI-LEAK ASSERTS PASS"

ENV PYTHONPATH=/app/repo \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/kernelbench-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64
WORKDIR /app/repo

# ---- build-time smoke: the graded module imports FROM /app/repo (the verifier asserts the same
#      thing at scoring time via its import-origin gate) and the scope files are present. ----
# ---- frozen naive baseline for the verifier's SPEED DENOMINATOR.
#      tests/test.sh resolves KB_BASELINE_MODULE=/opt/verifier-baseline/<file>; when that file
#      is absent the whole benchmark block is SKIPPED and the run hard-fails with
#      `validation_incomplete` (reward 0) no matter how good the submission is.
#      It is a pristine copy of the shipped, pre-edit scope files, taken here at build time —
#      exactly what the original task image baked. `cmp` proves the copy is byte-identical.
RUN mkdir -p /opt/verifier-baseline \
 && cp "/app/repo/sgemm.py" "/opt/verifier-baseline/sgemm.py" \
 && cmp "/app/repo/sgemm.py" "/opt/verifier-baseline/sgemm.py" \
 && cp "/app/repo/sgemm_kernel.cu" "/opt/verifier-baseline/sgemm_kernel.cu" \
 && cmp "/app/repo/sgemm_kernel.cu" "/opt/verifier-baseline/sgemm_kernel.cu" \
 && chmod 0444 "/opt/verifier-baseline/sgemm.py" "/opt/verifier-baseline/sgemm_kernel.cu" \
 && echo "frozen baseline(s) baked: $(ls -l /opt/verifier-baseline | tail -n +2 | wc -l) file(s)"

RUN python - <<'PY'
import os, subprocess, sys
sys.path.insert(0, "/app/repo")
for f in ('/app/repo/sgemm_kernel.cu',):
    p = os.path.join("/app/repo", f)
    assert os.path.isfile(p), f"scope file missing: {p}"
    print("scope file present:", f, os.path.getsize(p), "B")
for _b in ('sgemm.py',):
    _p = os.path.join('/opt/verifier-baseline', _b)
    assert os.path.isfile(_p), f"frozen baseline missing: {_p} (tests/test.sh would skip the benchmark and hard-fail validation_incomplete)"
    print('frozen baseline:', _p, os.path.getsize(_p), 'B')
n = subprocess.run(["git","-C","/app/repo","rev-list","--count","HEAD"],
                   capture_output=True, text=True).stdout.strip()
assert n == "1", f"git diff base broken (commits={n!r})"
print("OK: scope files, import origin and git base all in place")
PY

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
