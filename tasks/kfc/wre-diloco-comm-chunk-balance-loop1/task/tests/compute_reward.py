#!/usr/bin/env python3
"""diloco-comm-chunk-balance compute_reward — PERFORMANCE reward formula.

    reward = min(1.0, 0.5 * ln(speedup)/ln(ref_speedup)) if speedup > 1.0 else 0.0          range [0.0, 1.0]

The verifier measures vs_oracle = oracle_ms / candidate_ms (the oracle is the 1.0
anchor because the empty/stub start cannot be timed). Absolute speedup vs the
BASELINE is therefore  speedup = vs_oracle * ref_speedup, where ref_speedup is the
oracle's speedup over the naive baseline. So the oracle scores 0.5, the naive
baseline scores 0.0, and oracle^2 caps at 1.0.

The metric is a deterministic byte total, so no candidate can beat the oracle:
the gate must therefore be inclusive at the oracle (`speedup > 1.0`, i.e. cross the
baseline) instead of `speedup > ref_speedup`, which no submission can satisfy.

Pre-gates (any one hit => reward 0.0, the formula is not entered): the correctness
gate in test.sh (which owns scope_ok / import_origin_ok / correctness_ok and writes
the authoritative verdict), speedup <= 1, ref_speedup <= 1.

NOTE: test.sh computes the authoritative reward inline and only presence-checks this
file. This module is the same formula, kept for standalone/manual use.
"""
import json, math, sys

TOKEN = "WRE_RESULT "


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith(TOKEN):
                return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRE_RESULT in {path}")


def main():
    # argv: candidate_timing_out oracle_ms [ref_speedup]
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else None
    ref_speedup = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cand_ms = cand.get("timing_ms", -1)
    vs_oracle = (oracle_ms / cand_ms) if (oracle_ms and cand_ms and cand_ms > 0) else 0.0
    speedup = vs_oracle * ref_speedup
    hard = []
    if not (math.isfinite(speedup) and math.isfinite(ref_speedup)):
        hard.append("non_finite_metric")
    elif ref_speedup <= 1.0:
        hard.append("ref_speedup_invalid")
    elif speedup <= 1.0:
        hard.append("speedup_not_above_baseline")
    reward = 0.0
    if not hard:
        reward = 0.5 * math.log(speedup) / math.log(ref_speedup) if speedup > 1.0 else 0.0
        reward = max(0.0, min(1.0, reward))
    out = {
        "task_type": "performance",
        "reward": round(reward, 6),
        "hard_fail_reasons": hard,
        "speedup": round(speedup, 6),
        "ref_speedup": ref_speedup,
        "cv": {"baseline": None, "candidate": None},
        "vs_oracle": round(vs_oracle, 6),
        "candidate_ms": round(cand_ms, 4),
        "oracle_ms": round(oracle_ms, 4) if oracle_ms else None,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
