#!/usr/bin/env bash
# Verifier for wro-lora-punica-fused (Punica multi-LoRA shrink/expand fused-GEMM subsystem).
# Type-2 Long-horizon, PERFORMANCE lane. Scope = the 3 files under vllm/lora/ops/triton_ops/
# listed in SCOPE below. Modes: candidate | noop | oracle | negative.
# reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; speedup = ABBA-paired median(base_ms/cand_ms)
# over >=5 pairs. Hard-fail (reward=0) on any named gate below, on ref_speedup<=1, or on
# speedup<=NOOP_FLOOR -- so noop/negative score exactly 0.
# (baseline2 = N/A: single algorithmic lever -- fused grouped shrink/expand.)
#
# Design notes -- each item is backed by a measurement or a build record:
#  (a) scope gate whitelists exactly `gated_gelu.py` (bare repo-root path, no directory prefix).
#      That file is a stray artifact of the shared GPU build base; it is
#      untracked in the non-git-baked base image, so `git status --porcelain` listed it in EVERY
#      mode and zeroed noop/oracle/negative alike (measured in a 4-mode
#      validation run). Whitelisting one exact filename is the narrowest possible exemption:
#      loosening the *matching* instead (e.g. ignoring repo-root files) would exempt every future
#      root-level file and hand solvers an out-of-scope code container. The compensating control
#      is the `gated_gelu` entry in BANNED below: the only way in-scope code can reach the
#      whitelisted file is to name it, and naming it is a hard fail.
#  (b) gate (1c): the ABBA baseline is now materialized from the baked baseline COMMIT
#      (`git show HEAD:<path>`) and asserted to carry BASELINE_MARKER. Before, the baseline was
#      `git checkout -- <scope>` = the INDEX, which is silently wrong twice over:
#        * /opt/loop/submit.sh --finalize plants the winning submission with
#          `read-tree` + `checkout-index` (peeked from the shipped loop16 image), which leaves the
#          INDEX holding the CANDIDATE tree -- so the authoritative end-of-session score compared
#          the candidate against itself (speedup ~1 -> reward 0 for a correct, fast solve);
#        * in a base image that was never git-baked, the index/HEAD holds the PRISTINE upstream
#          implementation (a crane-built task image has HEAD=pristine),
#          so the "baseline" was the reference solution itself.
#      `--finalize` does not move HEAD (it resets to HEAD and then plants into index+worktree), so
#      HEAD stays the frozen degraded baseline; asserting the marker makes the pristine-HEAD case a
#      NAMED failure instead of a plausible-looking 0.
#  (c) gate (1b): BANNED route-around scan over the scope files' CODE lines (comments stripped, so
#      merely mentioning a token in prose cannot trip it). Every token was verified to occur 0
#      times in BOTH the shipped degraded tree and the reference implementation, and each one
#      enforces a rule instruction.md states verbatim ("All work must stay inside the declared
#      editable scope" / "Solve from the code present").
#  (d) NOOP_FLOOR 1.0 -> 1.10. The unedited degraded tree really does measure above 1.0 on this
#      image/lane: 0.9896 and 1.023 on two consecutive unedited submissions of the loop16 image
#      and 0.998 in the historical 4-mode run. A bare `>1.0` gate therefore lets an unedited tree
#      collect a small positive reward. At ref_speedup=16.59 a 1.10 floor costs a genuine
#      submission at most 0.017 reward, so it buys "noop == exactly 0" almost for free.
#  (e) the verdict line now carries BOTH `hard_fail_reasons` (the bench reward spec schema) and
#      `hard_fails` -- /opt/loop/score_engine.sh only recognises a verdict line that has
#      `gates` AND `hard_fails`, and treats anything else as harness_error.
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"   # pin venv (non-login exec has no torch on bare python3)
git config --global --add safe.directory '*' 2>/dev/null || true              # crane/root-owned /app/repo
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
TT="vllm/lora/ops/triton_ops"
SCOPE=("$TT/kernel_utils.py" "$TT/lora_shrink_op.py" "$TT/lora_expand_op.py")
# Files that must carry the degraded-baseline marker in the baked baseline commit. (kernel_utils.py
# is excluded on purpose: its degraded form is stub bodies, it has no loop marker.)
BASELINE_FILES=("$TT/lora_shrink_op.py" "$TT/lora_expand_op.py")
# Marker of the DEGRADED baseline (the per-adapter eager loop the solver must replace). Verified
# present exactly once in each BASELINE_FILES of the shipped degraded tree and absent from the
# reference implementation (oracle.patch removes both occurrences). Used ONLY to assert that the
# tree we time as "baseline" really is the shipped baseline -- never to score a submission.
BASELINE_MARKER="for lid in active"
# Route-around ban, scanned over the scope files' CODE lines only. Each token enforces a rule the
# instruction states: stay inside the declared scope, and solve from the code present (no external
# copy, no second copy from the container, no implementation recovered from git history, no
# dynamic loading of code from outside the scope). All 13 occur 0 times in the shipped tree AND 0
# times in the reference implementation, so none of them can fire on honest work.
BANNED=("site-packages" "dist-packages" "importlib" "spec_from_file_location" "import_module" \
        "subprocess" "os.system" "os.popen" "exec(" ".git/" "gated_gelu" \
        "ops.torch_ops" "ops.xla_ops")
NOOP_FLOOR=1.10   # see header (d): measured unedited-tree band on this image/lane is 0.99-1.03

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
ban_ok=true; baseline_ok=true
speedup=0.0; ref_speedup=16.59; base_ms=-1; cand_ms=-1   # 16.59 = oracle-calibrated headroom (matches the baked /opt/verifier-correctness-manifest.json)

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }
RUNDIR="$TESTS_DIR"

purge_pyc(){ rm -rf "$REPO/$TT/__pycache__" 2>/dev/null || true; }

# ---- apply oracle / negative patch by mode (candidate & noop: no patch) ----
if [ "$MODE" = "oracle" ]   && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];   then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"   2>"$LOG/oracle_apply.log"   || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "negative" ] && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH" 2>"$LOG/negative_apply.log" || true; fi

# ---- (1) scope-diff HARD GATE: only the 3 scope files may change in /app/repo ----
git -C "$REPO" update-index -q --refresh 2>/dev/null || true
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$f" = "$s" ] && keep=true; done
  case "$f" in
    *__pycache__*|*.pyc) keep=true;;
    gated_gelu.py) keep=true;;   # stray build-base artifact at the repo root (see header (a)); exempt by exact path only -- `vllm/gated_gelu.py` etc. still fail the gate. Companion BANNED token stops it being used as a code container.
  esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$f"; }
done < "$LOG/changed_files.txt"

# ---- (1b) BANNED route-around scan (code lines only; comments stripped) ----
: > "$LOG/banned_hits.txt"
for f in "${SCOPE[@]}"; do
  [ -f "$REPO/$f" ] || continue
  code=$(sed -e 's/#.*$//' "$REPO/$f" 2>/dev/null || true)
  for t in "${BANNED[@]}"; do
    if printf '%s\n' "$code" | grep -Fq -- "$t"; then
      echo "$f :: $t" >> "$LOG/banned_hits.txt"
      ban_ok=false; add_hard "banned_token:$t"
    fi
  done
done

# ---- (1c) BASELINE-TREE assert: the tree we will time as "baseline" must BE the shipped baseline.
#      Source of truth = the baked baseline commit (HEAD), which --finalize never moves. ----
for f in "${BASELINE_FILES[@]}"; do
  blob=$(git -C "$REPO" show "HEAD:$f" 2>>"$LOG/baseline_probe.err" || true)
  if [ -z "$blob" ]; then
    baseline_ok=false; add_hard "baseline_tree_unreadable:$f"
  elif [ "$(printf '%s\n' "$blob" | grep -Fc -- "$BASELINE_MARKER")" = "0" ]; then
    baseline_ok=false; add_hard "baseline_tree_not_degraded:$f"
  fi
done

# ---- (2) import-origin assert: vllm under test must be the baked /app/repo tree ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import importlib, os, sys
m = importlib.import_module("vllm.lora.ops.triton_ops.lora_shrink_op")
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
if [ "${#HARD[@]}" -eq 0 ] && [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  purge_pyc
  ( cd "$RUNDIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  corr_rc=$?
  cok=$(grep WRO_LORA_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_LORA_RESULT ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING: ABBA-paired candidate vs baseline, >=5 pairs -> median speedup + cv ----
#      baseline = the frozen degraded tree materialized from HEAD (see header (b)).
#      The candidate half is restored from a plain file snapshot: `git stash` needs a git identity
#      and silently no-ops without one, which would leave the BASELINE in place for every later
#      candidate half and quietly drive the measured speedup to ~1.
N_PAIRS=5
cand_arr=(); base_arr=()
CSNAP="$LOG/cand_snapshot"
if [ "$correctness_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  rm -rf "$CSNAP"; mkdir -p "$CSNAP"
  for f in "${SCOPE[@]}"; do
    mkdir -p "$CSNAP/$(dirname "$f")"
    cp -p "$REPO/$f" "$CSNAP/$f" 2>/dev/null || { add_hard "candidate_snapshot_failed:$f"; benchmark_ok=false; }
  done
  for i in $(seq 1 "$N_PAIRS"); do
    # --- A: candidate ---
    purge_pyc
    ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/cand_timing_$i.out" 2>&1 || add_hard "candidate_timing_failed"
    c=$(grep WRO_LORA_RESULT "$LOG/cand_timing_$i.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_LORA_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
    cand_arr+=("$c")
    # --- B: baseline (materialize the baked baseline commit's scope files) ---
    for f in "${SCOPE[@]}"; do
      if git -C "$REPO" show "HEAD:$f" > "$LOG/base_blob.tmp" 2>>"$LOG/baseline_probe.err" && [ -s "$LOG/base_blob.tmp" ]; then
        cat "$LOG/base_blob.tmp" > "$REPO/$f"
      else
        add_hard "baseline_materialize_failed:$f"; benchmark_ok=false
      fi
    done
    purge_pyc
    ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/base_timing_$i.out" 2>&1 || add_hard "baseline_timing_failed"
    b=$(grep WRO_LORA_RESULT "$LOG/base_timing_$i.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_LORA_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
    base_arr+=("$b")
    # --- restore the candidate for the next A half (and for the caller) ---
    for f in "${SCOPE[@]}"; do
      cp -p "$CSNAP/$f" "$REPO/$f" 2>/dev/null || { add_hard "candidate_restore_failed:$f"; benchmark_ok=false; }
    done
  done
  purge_pyc
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
  sp_gt=$(python3 -c "print(1 if $speedup > $NOOP_FLOOR else 0)" 2>/dev/null || echo 0)
  ref_gt1=$(python3 -c "print(1 if $ref_speedup > 1.0 else 0)" 2>/dev/null || echo 0)
  if [ "$sp_gt" != "1" ]; then
    add_hard "speedup_not_above_1"          # canonical reward.md gate name; threshold = NOOP_FLOOR (noise margin, see header (d))
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
       WRO_CORR="$correctness_ok" WRO_BENCH="$benchmark_ok" \
       WRO_BAN="$ban_ok" WRO_BASEOK="$baseline_ok" WRO_FLOOR="$NOOP_FLOOR"
mkdir -p "$LOG"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except: return -1.0
ref = f(os.environ.get("WRO_REF", "1")); sp = f(os.environ.get("WRO_SPEEDUP", "0"))
hard = os.environ.get("WRO_HARD", "").split()
v = {
  "task_type": "performance",
  "reward": f(os.environ.get("WRO_REWARD")),
  "hard_fail_reasons": hard,
  "hard_fails": hard,                      # /opt/loop/score_engine.sh only parses a verdict line carrying BOTH `gates` and `hard_fails`
  "speedup": sp,
  "ref_speedup": ref,
  "cv": {"candidate": f(os.environ.get("WRO_CV_CAND")), "baseline": f(os.environ.get("WRO_CV_BASE"))},
  "mode": os.environ.get("WRO_MODE"),
  "baseline_ms": f(os.environ.get("WRO_BASE_MS")),
  "candidate_ms": f(os.environ.get("WRO_CAND_MS")),
  "metadata": {"vs_oracle_ratio": (sp/ref) if ref > 0 else None,
               "noop_floor": f(os.environ.get("WRO_FLOOR")),
               "baseline_source": "git show HEAD:<scope> (baked degraded baseline commit)"},
  "gates": {"scope_ok": os.environ.get("WRO_SCOPE") == "true",
            "import_origin_ok": os.environ.get("WRO_IMP") == "true",
            "correctness_ok": os.environ.get("WRO_CORR") == "true",
            "benchmark_ok": os.environ.get("WRO_BENCH") == "true",
            "ban_ok": os.environ.get("WRO_BAN") == "true",
            "baseline_ok": os.environ.get("WRO_BASEOK") == "true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
