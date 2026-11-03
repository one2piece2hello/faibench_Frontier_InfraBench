#!/usr/bin/env bash
# Verifier for wre-diloco-comm-chunk-balance-loop1.
# Scope = submission/kernel.py (the ONLY solver-editable file). The solver places <= num_chunks
# CONTIGUOUS chunk boundaries over a gradient-tensor size stream to minimize the LARGEST chunk's
# bytes (the ring-all-reduce bottleneck), to the disclosed contract, then optimizes.
#
# reward = min(1, ln(speedup/ref_speedup)/ln(ref_speedup)); oracle -> 0.5, naive baseline -> 0.0,
# behind a HARD correctness gate (every emitted partition must be structurally VALID; pure host
# logic, no float tolerance). oracle_ms is a calibrated
# held-out constant (the empty/stub start cannot be timed). noop(stub)=NotImplementedError -> 0.
#
# Modes: candidate | noop | oracle | negative | baseline2.  loop16 calls candidate only.
# Emits ONE single-line JSON verdict: {gates, hard_fails, reward, speedup, ...}  (score_engine §X).
set -uo pipefail
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
git config --global --add safe.directory '*' 2>/dev/null || true   # §N crane/root-owned /app/repo

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
SUB="submission/kernel.py"
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
speedup=0.0; oracle_ms=-1; cand_ms=-1

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }

# ---- oracle_ms: calibrated held-out constant (baked at build; draft default pre-calibration) ----
if [ -f /opt/verifier-correctness-manifest.json ]; then
  oracle_ms=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('oracle_ms',-1))" 2>/dev/null || echo -1)
fi

# ---- ref_speedup = oracle-vs-BASELINE anchor, needed by the log reward formula ----
# The measured metric here is vs_oracle = oracle_ms/candidate_ms (the oracle is the 1.0
# anchor because an empty/stub start cannot be timed). The reward formula is defined on
# speedup-vs-BASELINE, and speedup = vs_oracle * ref_speedup, so:
#     reward = min(1, ln(vs_oracle) / ln(ref)), and 0 unless vs_oracle > 1 (beat the oracle)
# => oracle scores 0.5, the naive baseline scores 0.0, oracle^2 caps at 1.0.
# Prefer the tests/-local override (uploaded fresh -> recalibrate without a rebuild).
ref_speedup=1.0
if [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
elif [ -f /opt/verifier-correctness-manifest.json ]; then
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
fi
[ -n "$ref_speedup" ] || ref_speedup=1.0

# ---- mode dispatch: swap the reviewer variant over the scope file (NEVER in candidate/noop) ----
case "$MODE" in
  oracle)    [ -n "${KERNELBENCH_ORACLE_FILE:-}" ]    && cp "$KERNELBENCH_ORACLE_FILE"    "$REPO/$SUB" ;;
  negative)  [ -n "${KERNELBENCH_NEGATIVE_FILE:-}" ]  && cp "$KERNELBENCH_NEGATIVE_FILE"  "$REPO/$SUB" ;;
  baseline2) [ -n "${KERNELBENCH_BASELINE2_FILE:-}" ] && cp "$KERNELBENCH_BASELINE2_FILE" "$REPO/$SUB" ;;
  candidate|noop) : ;;   # score the live tree (candidate=solver edit, noop=baked stub)
esac

# ---- (1) scope-diff HARD GATE: only submission/kernel.py may change vs the baked baseline ----
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    case "$f" in
      "$SUB") : ;;
      *__pycache__*|*.pyc) : ;;
      *) scope_ok=false; add_hard "out_of_scope_edit:$f" ;;
    esac
  done < "$LOG/changed_files.txt"
fi

# ---- (2) import-origin assert: the scored kernel must be the baked /app/repo copy ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import importlib.util, os, sys
p = "/app/repo/submission/kernel.py"
print("KERNEL_PATH", os.path.realpath(p))
sys.exit(0 if os.path.realpath(p).startswith("/app/repo") and os.path.exists(p) else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked into the solver-visible tree) ----
for f in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$f" ] || { add_hard "hidden_supplement_missing:$f"; correctness_ok=false; }
done

# ---- (4) CORRECTNESS GATE (HARD): scope output must match the fp32 reference. noop/negative FAIL. ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  cok=$(grep WRE_RESULT "$LOG/correctness.out" | tail -1 | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRE_RESULT ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING (only if correct): candidate_ms; speedup = oracle_ms / candidate_ms ----
if [ "$correctness_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  read cand_ms flat_ok stable_ok < <(grep WRE_RESULT "$LOG/cand_timing.out" | tail -1 | python3 -c "import json,sys
try:
    d=json.loads(sys.stdin.read().split('WRE_RESULT ',1)[1]); print(d.get('timing_ms',-1), d.get('flat_ok',False), d.get('stable_ok',False))
except Exception: print('-1 False False')" 2>/dev/null || echo "-1 False False")
  # NOTE: flat_ok/stable_ok are RECORDED diagnostics, not a hard gate — the CSPRNG cache-probe
  # in the correctness stage is the real anti-cache gate (it subsumes constant-output cheats).
  # Only an invalid measurement (candidate_ms<=0) hard-fails timing.
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $oracle_ms>0 else 0)")" = "1" ]; then
    speedup=$(python3 -c "print(round($oracle_ms/$cand_ms,6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict (SINGLE-LINE JSON; §X score_engine parses the last {gates,hard_fails} line) ----
# PERFORMANCE reward formula. `speedup` here is vs_oracle (oracle_ms/cand_ms);
# absolute speedup vs the baseline = vs_oracle * ref_speedup, so
#   reward = min(1.0, ln(vs_oracle)/ln(ref_speedup)), 0 unless vs_oracle > 1.
# Pre-gates (any hit -> reward 0, formula not entered): a hard_fail, correctness FAIL,
# absolute speedup <= 1 (did not cross the baseline), ref_speedup <= 1 (bad anchor).
nhard="${#HARD[@]}"
abs_speedup=$(python3 -c "print(round($speedup*$ref_speedup,6))" 2>/dev/null || echo 0)
if [ "$correctness_ok" = true ] && [ "$nhard" -eq 0 ]; then
  if [ "$(python3 -c "print(1 if $ref_speedup<=1.0 else 0)" 2>/dev/null || echo 1)" = "1" ]; then
    add_hard "ref_speedup_invalid"
  elif [ "$(python3 -c "print(1 if $abs_speedup<1.0 else 0)" 2>/dev/null || echo 1)" = "1" ]; then
    add_hard "speedup_below_baseline"
  fi
  nhard="${#HARD[@]}"
fi
reward=0.0
if [ "$nhard" -eq 0 ] && [ "$correctness_ok" = true ]; then
  reward=$(python3 -c "
import math
sp=$abs_speedup; ref=$ref_speedup
r=0.5 * math.log(sp) / math.log(ref) if sp > 1.0 else 0.0
print(round(max(0.0, min(1.0, r)), 6))" 2>/dev/null || echo 0.0)
fi
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRE_MODE="$MODE" WRE_REWARD="$reward" WRE_SPEEDUP="$abs_speedup" \
       WRE_VS_ORACLE="$speedup" WRE_REF="$ref_speedup" \
       WRE_ORACLE_MS="$oracle_ms" WRE_CAND_MS="$cand_ms" \
       WRE_HARD="$hard_str" WRE_SCOPE="$scope_ok" WRE_IMP="$import_origin_ok" \
       WRE_CORR="$correctness_ok" WRE_BENCH="$benchmark_ok"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except Exception: return -1.0
sp = f(os.environ.get("WRE_SPEEDUP","0"))          # ABSOLUTE speedup vs the baseline
vo = f(os.environ.get("WRE_VS_ORACLE","0"))        # oracle_ms/cand_ms (oracle == 1.0)
v = {
  "task_type": "performance",
  "mode": os.environ.get("WRE_MODE"),
  "reward": f(os.environ.get("WRE_REWARD")),
  "speedup": sp,                                   # vs the baseline (reward semantics)
  "ref_speedup": f(os.environ.get("WRE_REF","1")),
  "cv": {"baseline": None, "candidate": None},
  "vs_oracle": vo,
  "oracle_ms": f(os.environ.get("WRE_ORACLE_MS")),
  "candidate_ms": f(os.environ.get("WRE_CAND_MS")),
  "hard_fails": os.environ.get("WRE_HARD","").split(),
  "hard_fail_reasons": os.environ.get("WRE_HARD","").split(),
  "gates": {"scope_ok": os.environ.get("WRE_SCOPE")=="true",
            "import_origin_ok": os.environ.get("WRE_IMP")=="true",
            "correctness_ok": os.environ.get("WRE_CORR")=="true",
            "benchmark_ok": os.environ.get("WRE_BENCH")=="true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
