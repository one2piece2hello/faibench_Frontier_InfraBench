#!/usr/bin/env bash
# Verifier for wro-ssm-ssd-chunkscan (Mamba-2 SSD state-space scan subsystem).
# Type-2 Long-horizon, PERFORMANCE lane. Scope = 5 files under
#   vllm/model_executor/layers/mamba/ops/.
# Modes: candidate | noop | oracle | negative.
# reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; speedup = ABBA-paired median(base_ms/cand_ms)
# over >=5 pairs. Hard-fail (reward=0) if speedup<=1 or ref_speedup<=1 -- so noop/negative score 0.
# NOTE: the gate uses a 1.02 threshold, not a bare 1.0. the authoring
# measurements document a noop noise floor of 0.987-1.006 (wall-clock e2e timing);
# a real noop run measured speedup=1.001776 -> reward=0.000144 (nonzero) against a
# bare >1.0 gate. 1.02 comfortably absorbs that documented noise ceiling while remaining negligible
# next to ref_speedup=467.4, so no genuine optimization can be mistaken for noise.
# (baseline2 = N/A: single algorithmic lever — chunked reformulation; see scope card.)
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"   # pin venv (non-login exec has no torch on bare python3)
git config --global --add safe.directory '*' 2>/dev/null || true              # crane/root-owned /app/repo
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
MB="vllm/model_executor/layers/mamba/ops"
SCOPE=("$MB/ssd_combined.py" "$MB/ssd_chunk_state.py" "$MB/ssd_state_passing.py" "$MB/ssd_chunk_scan.py" "$MB/ssd_bmm.py")

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
speedup=0.0; ref_speedup=467.4; base_ms=-1; cand_ms=-1   # 467.4 = oracle-calibrated headroom (matches the baked /opt/verifier-correctness-manifest.json)

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }
RUNDIR="$TESTS_DIR"

# ---- apply oracle / negative patch by mode (candidate & noop: no patch) ----
if [ "$MODE" = "oracle" ]   && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];   then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"   2>"$LOG/oracle_apply.log"   || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "negative" ] && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH" 2>"$LOG/negative_apply.log" || true; fi

# ---- (1) scope-diff HARD GATE: only the 5 scope files may change in /app/repo ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$f" = "$s" ] && keep=true; done
  case "$f" in *__pycache__*|*.pyc) keep=true;; esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$f"; }
done < "$LOG/changed_files.txt"

# ---- (2) import-origin assert: vllm under test must be the baked /app/repo tree ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import importlib, os, sys
m = importlib.import_module("vllm.model_executor.layers.mamba.ops.ssd_chunk_scan")
loc = os.path.realpath(m.__file__)
print("SCOPE_LOC", loc)
sys.exit(0 if loc.startswith("/app/repo/") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked) ----
for f in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$f" ] || { add_hard "hidden_supplement_missing:$f"; correctness_ok=false; }
done

# ---- ref_speedup (oracle-calibrated). Priority: image-baked manifest > task-local mirror > hardcoded fallback above ----
[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',$ref_speedup))" 2>/dev/null || echo "$ref_speedup")
[ -f "$TESTS_DIR/verifier-correctness-manifest.json" ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('$TESTS_DIR/verifier-correctness-manifest.json')).get('ref_speedup',$ref_speedup))" 2>/dev/null || echo "$ref_speedup")

# ---- (4) CORRECTNESS GATE: scope output must match the independent fp32 reference ----
#      negative mode is EXPECTED to fail this gate.
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$RUNDIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  corr_rc=$?
  cok=$(grep WRO_SSM_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_SSM_RESULT ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING: ABBA-paired candidate vs baseline, >=5 pairs -> median speedup + cv ----
N_PAIRS=5
cand_arr=(); base_arr=()
if [ "$correctness_ok" = true ] && { [ "${#HARD[@]}" -eq 0 ] || [ -z "${HARD[*]:-}" ]; }; then
  for i in $(seq 1 "$N_PAIRS"); do
    ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/cand_timing_$i.out" 2>&1 || add_hard "candidate_timing_failed"
    c=$(grep WRO_SSM_RESULT "$LOG/cand_timing_$i.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_SSM_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
    cand_arr+=("$c")
    # baseline = the frozen degraded tree (clean checkout of the scope files at HEAD)
    git -C "$REPO" stash -q 2>/dev/null || true
    git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || true
    ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/base_timing_$i.out" 2>&1 || add_hard "baseline_timing_failed"
    b=$(grep WRO_SSM_RESULT "$LOG/base_timing_$i.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_SSM_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
    base_arr+=("$b")
    git -C "$REPO" stash pop -q 2>/dev/null || true
  done
  result_line=$(python3 -c "
import statistics, sys
k = int(sys.argv[1])
cand = [float(x) for x in sys.argv[2:2+k]]
base = [float(x) for x in sys.argv[2+k:2+2*k]]
ok = k > 0 and all(c > 0 for c in cand) and all(b > 0 for b in base)
if not ok:
    print('0 -1 -1 -1 -1 0')
else:
    ratios = [b / c for b, c in zip(base, cand)]
    sp = statistics.median(ratios)
    cv_c = (statistics.pstdev(cand) / statistics.mean(cand)) if statistics.mean(cand) else -1
    cv_b = (statistics.pstdev(base) / statistics.mean(base)) if statistics.mean(base) else -1
    print(round(sp,6), round(statistics.median(cand),4), round(statistics.median(base),4), round(cv_c,4), round(cv_b,4), 1)
" "$N_PAIRS" "${cand_arr[@]}" "${base_arr[@]}")
  read -r speedup cand_ms base_ms cv_cand cv_base timing_ok <<< "$result_line"
  [ "$timing_ok" = "1" ] || { benchmark_ok=false; add_hard "timing_invalid"; }
fi

# ---- verdict: reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; named hard-fail gates ----
reward=0.0
if [ "${#HARD[@]}" -eq 0 ] && [ "$correctness_ok" = true ]; then
  sp_gt1=$(python3 -c "print(1 if $speedup > 1.02 else 0)" 2>/dev/null || echo 0)  # 1.02 noise margin, see header NOTE
  ref_gt1=$(python3 -c "print(1 if $ref_speedup > 1.0 else 0)" 2>/dev/null || echo 0)
  if [ "$sp_gt1" != "1" ]; then
    add_hard "speedup_not_above_1"
  elif [ "$ref_gt1" != "1" ]; then
    add_hard "ref_speedup_not_above_1"
  else
    reward=$(python3 -c "import math; print(round(min(1.0, max(0.0, min(1.0, math.log($speedup) / math.log($ref_speedup) - 1.0))), 6))")
  fi
fi
nhard="${#HARD[@]}"
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRO_MODE="$MODE" WRO_REWARD="$reward" WRO_SPEEDUP="$speedup" \
       WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" WRO_REF="$ref_speedup" \
       WRO_CV_CAND="${cv_cand:--1}" WRO_CV_BASE="${cv_base:--1}" \
       WRO_HARD="$hard_str" WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" \
       WRO_CORR="$correctness_ok" WRO_BENCH="$benchmark_ok"
mkdir -p "$LOG"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except: return -1.0
ref = f(os.environ.get("WRO_REF", "1")); sp = f(os.environ.get("WRO_SPEEDUP", "0"))
v = {
  "task_type": "performance",
  "reward": f(os.environ.get("WRO_REWARD")),
  "hard_fail_reasons": os.environ.get("WRO_HARD", "").split(),
  "speedup": sp,
  "ref_speedup": ref,
  "cv": {"candidate": f(os.environ.get("WRO_CV_CAND")), "baseline": f(os.environ.get("WRO_CV_BASE"))},
  "mode": os.environ.get("WRO_MODE"),
  "baseline_ms": f(os.environ.get("WRO_BASE_MS")),
  "candidate_ms": f(os.environ.get("WRO_CAND_MS")),
  "metadata": {"vs_oracle_ratio": (sp/ref) if ref > 0 else None},
  "gates": {"scope_ok": os.environ.get("WRO_SCOPE") == "true",
            "import_origin_ok": os.environ.get("WRO_IMP") == "true",
            "correctness_ok": os.environ.get("WRO_CORR") == "true",
            "benchmark_ok": os.environ.get("WRO_BENCH") == "true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
