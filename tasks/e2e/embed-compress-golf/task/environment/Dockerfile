# syntax=docker/dockerfile:1
# =============================================================================
# e2e-g2-embed-compress-golf — OPEN-SOURCE, self-contained task image
#
#   Task: build a retrieval encoder that fits in a HARD 64 bytes per vector and still ranks
#         well (nDCG@10). The agent writes /app/submission/submission_encoder.py and may edit
#         the whole sentence-transformers tree at /app/repo.
#
#   FULLY PUBLIC — no internal mirror, no private base image, no prebuilt image.
#   CPU ONLY — task.toml declares gpus = 0; a CPU-only torch wheel is installed on purpose.
#   Platform: linux/amd64 or arm64.
#   Build context = the task/ directory (this Dockerfile lives in task/environment/).
#
# -----------------------------------------------------------------------------
# BUILD  (needs outbound network: PyPI + github.com + huggingface.co)
#     cd <...>/embed-compress-golf/task
#     docker build -f environment/Dockerfile -t fai/e2e-g2-embed-compress-golf:oss .
#
# RUN
#     docker run --rm -it fai/e2e-g2-embed-compress-golf:oss
#
# SCORE (inside the container; tests/ is MOUNTED, never baked into the image)
#     docker run --rm -v "$PWD/tests:/tests:ro" \
#       fai/e2e-g2-embed-compress-golf:oss bash /tests/test.sh   # -> /logs/verifier/reward.json
#
# 🔴 RE-CALIBRATE THE ANCHORS ON YOUR MACHINE
#   reward = min(1.0, ln(gain/ref_speedup)/ln(ref_speedup)) if gain > ref_speedup else 0.0 with the FROZEN constants in
#   tests/verifier-correctness-manifest.json:
#       ref_speedup = 1.4290238072817099   (best in-budget variant: 32 dims x float16 = 64 B)
#       strong_baseline_ndcg = 0.459151
#   Both were measured on the authoring CPU lane (Intel Sapphire Rapids). The QUALITY numbers
#   are hardware-independent as long as the corpus reproduces (the build asserts that below);
#   only re-calibrate if you change the corpus, the base embedding model, or the byte budget.
#   To re-measure, score a reference encoder as the candidate:
#     docker run --rm -v "$PWD/tests:/tests:ro" -e VERIFIER_MODE=<mode> \
#       fai/e2e-g2-embed-compress-golf:oss bash /tests/test.sh
#     # read ndcg / gain from /logs/verifier/benchmark_results.json and write the new values into
#     # tests/verifier-correctness-manifest.json -> ref_speedup.value / thresholds.strong_baseline_ndcg
#   (ref_speedup <= 1 or missing => HARD FAIL 0 by design.)
#
# VERSION ANCHORS
#     Python 3.10 · sentence-transformers @ 28e7946 (editable at /app/repo) · torch 2.7.0+cpu
#     base embedding model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0, ~90 MB)
#
# THE DATASET IS REGENERATED, NOT SHIPPED
#   The original private base baked /data/retrieval as opaque blobs. This build downloads the
#   PUBLIC WikiText-103-raw parquet shard the package's own generator already points at
#   (environment/build_dataset.py, seed 20240725) and regenerates both splits, then ASSERTS that
#   the held-out split is byte-identical to this package's tests/heldout_*.{jsonl,json}
#   (digests in environment/tests_heldout_sha256.txt). If that assert passes, your corpus is
#   provably the one the anchors were calibrated on. The regenerated held-out copy is DELETED IN
#   THE SAME LAYER — the graded copy comes from tests/ at scoring time. Note that
#   heldout_qrels.json IS the relevance answer key, which is exactly why it must never be baked.
#
#   The original private overlay is kept beside this file as Dockerfile.orig_private_base.
# =============================================================================
FROM python:3.10-slim-bookworm

ARG ST_REF=28e79467c49b5bedff9b1492950504af04d0b74b
ARG MINILM_REPO=sentence-transformers/all-MiniLM-L6-v2
ARG WIKITEXT_PARQUET_URL=https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates curl procps coreutils \
        tar gzip vim less file jq \
    && rm -rf /var/lib/apt/lists/*

# ---- venv. Keep this exact path: tests/test.sh pins /opt/kernelbench-venv/bin ----
ENV VENV=/opt/kernelbench-venv
RUN python3 -m venv "$VENV"
ENV PATH=$VENV/bin:$PATH
RUN pip install --no-cache-dir -U pip setuptools wheel

# ---- CPU-only torch first, so sentence-transformers' resolver cannot pull a multi-GB CUDA
#      wheel into a task that declares gpus = 0. ----
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.7.0" \
 && python -c "import torch; print('torch', torch.__version__); assert torch.__version__.startswith('2.7.0')"

# ---- /app/repo: the WHOLE sentence-transformers tree at the recorded commit, editable-installed.
#      The commit is the one recorded in tests/verifier-correctness-manifest.json
#      (source_commit_app_repo), fetched by SHA so the tree is exactly the calibrated one.
#      Installed WITH its declared dependencies (transformers, scikit-learn, scipy, ...) under a
#      torch constraint so the CPU wheel above is kept. ----
RUN set -eu; \
    mkdir -p /app/repo; cd /app/repo; \
    git init -q; \
    git remote add origin https://github.com/UKPLab/sentence-transformers.git; \
    git fetch -q --depth 1 origin "${ST_REF}"; \
    git checkout -q FETCH_HEAD; \
    test -f /app/repo/pyproject.toml || { echo "BUILD_ASSERT_FATAL: sentence-transformers pyproject.toml missing"; exit 1; }; \
    n="$(find /app/repo -type f -not -path '*/.git/*' | wc -l)"; \
    echo "[build] /app/repo file count = $n"; \
    test "$n" -ge 100 || { echo "BUILD_ASSERT_FATAL: /app/repo has only $n files - clone mutilated"; exit 1; }
RUN printf 'torch==2.7.0\n' > /tmp/constraints.txt \
 && pip install --no-cache-dir --constraint /tmp/constraints.txt "numpy>=1.26,<3" "pyarrow==17.0.0" \
 && pip install --no-cache-dir --constraint /tmp/constraints.txt "transformers>=4.41,<5" -e /app/repo \
 && python -c "import sentence_transformers, pathlib; p=pathlib.Path(sentence_transformers.__file__).resolve(); print('sentence-transformers', sentence_transformers.__version__, p); assert str(p).startswith('/app/repo/'), p" \
 && python -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__"

# ---- the frozen base embedding model. The agent may use / truncate / quantise / fine-tune it,
#      but the 64-bytes-per-vector budget is enforced by the verifier. Downloaded from the PUBLIC
#      Hugging Face repo (Apache-2.0) instead of being vendored in the task package. ----
RUN pip install --no-cache-dir "huggingface_hub[cli]>=0.24,<1" \
 && huggingface-cli download "${MINILM_REPO}" --local-dir /opt/models/all-MiniLM-L6-v2 \
        --exclude "onnx/*" "openvino/*" "*.onnx" "rust_model.ot" "tf_model.h5" \
 && python -c "import json; json.load(open('/opt/models/all-MiniLM-L6-v2/config.json')); print('minilm config OK')" \
 && test -f /opt/models/all-MiniLM-L6-v2/modules.json \
 && du -sh /opt/models/all-MiniLM-L6-v2

# ---- the retrieval dataset. The agent-visible DEV split lands in /data/retrieval; the held-out
#      split is regenerated only to PROVE the corpus matches the calibration, then deleted.
#      The parquet shard is the exact public URL the generator itself names. ----
COPY environment/build_dataset.py /tmp/e2e_build_dataset.py
COPY environment/tests_heldout_sha256.txt /tmp/tests_heldout_sha256.txt
RUN set -eu; \
    curl -fSL --retry 3 -o /tmp/g2_embed_wikitext_00.parquet "${WIKITEXT_PARQUET_URL}"; \
    python3 /tmp/e2e_build_dataset.py; \
    cd /tmp; \
    sha256sum heldout_corpus.jsonl heldout_queries.jsonl heldout_qrels.json | sort > /tmp/built_heldout.txt; \
    sort /tmp/tests_heldout_sha256.txt > /tmp/want_heldout.txt; \
    cat /tmp/built_heldout.txt; \
    diff /tmp/built_heldout.txt /tmp/want_heldout.txt || { \
      echo "BUILD_ASSERT_FATAL: regenerated held-out split != this package's tests/heldout_* surface."; \
      echo "  ref_speedup=1.4290238072817099 and strong_baseline_ndcg=0.459151 were calibrated on"; \
      echo "  the archived one; re-calibrate before scoring (see the header)."; exit 1; }; \
    echo "OK: regenerated held-out split is byte-identical to the CALIBRATED surface"; \
    rm -f /tmp/heldout_corpus.jsonl /tmp/heldout_queries.jsonl /tmp/heldout_qrels.json \
          /tmp/tests_heldout_sha256.txt /tmp/built_heldout.txt /tmp/want_heldout.txt \
          /tmp/e2e_build_dataset.py /tmp/g2_embed_wikitext_00.parquet; \
    test -f /data/retrieval/dev_corpus.jsonl; \
    test -f /data/retrieval/dev_queries.jsonl; \
    test -f /data/retrieval/dev_qrels.json; \
    echo "[build] agent-visible DEV split: $(wc -l < /data/retrieval/dev_corpus.jsonl) passages, $(wc -l < /data/retrieval/dev_queries.jsonl) queries"

# ---- solver-owned submission skeleton + the starter template + the task statement ----
RUN mkdir -p /app/submission /logs/verifier /logs/artifacts
COPY environment/workspace/submission_encoder_template.py /app/submission/submission_encoder_template.py
COPY instruction.md /app/instruction.md

# ---- git baseline for the editable tree: ONE commit as the diff base, no upstream remote ----
RUN git config --global user.email "reviewer@fai-bench.local" \
 && git config --global user.name "fai-bench" \
 && git config --global init.defaultBranch main \
 && git config --global --add safe.directory /app/repo \
 && cd /app/repo \
 && rm -rf .git \
 && printf '__pycache__/\n*.pyc\n' > .gitignore \
 && git init -q \
 && git add -A \
 && git commit -qm "sentence-transformers @ pinned commit (upstream history stripped)" \
 && test "$(git rev-list --count HEAD)" = "1" \
 && test -z "$(git remote)" \
 && echo "git diff base ready: $(git rev-parse --short HEAD)"

# ---- OPEN grading suite is NOT baked into the image (by design).
#   tests/ (test.sh + compute_reward.py + the held-out split + the frozen manifest carrying the
#   calibrated anchors and heldout_qrels.json, the relevance answer key) is MOUNTED FRESH at /tests when scoring. It ships in this
#   package — open-source, but deliberately outside the image so the solver can read neither the
#   held-out data nor the anchors while solving.

# ---- 🔴 ANTI-LEAK ASSERTS: any regression that re-bakes the frozen surface or the held-out
#      split dies HERE, at build time, instead of silently invalidating the task. ----
RUN set -eu; \
    for p in /opt/verifier /opt/verifier-correctness-manifest.json /opt/strong_baseline \
             /opt/negative /opt/ceiling /tests; do \
      if [ -e "$p" ]; then echo "BUILD_ASSERT_FATAL: frozen surface leaked into the image at $p"; exit 3; fi; \
    done; \
    hits="$(find / -xdev -type f \( -name 'heldout_*' -o -name compute_reward.py \
              -o -name 'verifier-correctness-manifest.json' \) 2>/dev/null | head -20)"; \
    if [ -n "$hits" ]; then echo "BUILD_ASSERT_FATAL: baked frozen-surface files:"; echo "$hits"; exit 3; fi; \
    echo "[build] ANTI-LEAK ASSERTS PASS: no frozen surface, no held-out split, no anchor in the image"

# ---- runtime contract: minimal entrypoint + elapsed stub (this family has no wall-clock budget) ----
RUN printf '#!/bin/bash\nset -e\nif [ "$#" -gt 0 ]; then exec "$@"; fi\nexec /bin/bash -l\n' > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && printf '#!/bin/bash\nSTART=$(date +%%s)\nwhile true; do echo "elapsed=$(( $(date +%%s) - START ))s"; sleep 60; done\n' > /app/timer.sh \
 && chmod +x /app/timer.sh

ENV DATA_PATH=/data/retrieval \
    BASE_EMBED_MODEL=/opt/models/all-MiniLM-L6-v2 \
    SUBMISSION_DIR=/app/submission \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app/submission

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["tail","-f","/dev/null"]
