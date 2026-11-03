#!/usr/bin/env python3
"""Off-chip-traffic metric for the tile-size planning task.

Deterministic COMPUTED byte total (no valgrind, no timing): for a fixed corpus of
(problems, cap) instances, each mode's ``plan_tiling`` returns a plan, the harness
INDEPENDENTLY validates it and sums the modeled off-chip bytes moved. ``test.sh`` takes
the ratio baseline_traffic / candidate_traffic. The candidate's plan is materialised once
(``tile_harness.freeze_plan``) before it is validated and charged, so both steps always
see the same tiles.

Modes:
  candidate  plan every instance with /app/repo/tile_planner.py; validate; print
             TRAFFIC=<total bytes> VALID=1  (or INVALID and exit non-zero)
  baseline   same, using the frozen baseline module (KB_BASELINE_MODULE)
  verify     confirm candidate AND baseline plans are valid on the corpus and report
             the naive-vs-reference headroom; print VERIFY_OK / VERIFY_FAIL
"""

import os
import sys

from tile_harness import (
    freeze_plan,
    load_candidate,
    load_module,
    make_bench_corpus,
    naive_traffic,
    plan_is_valid,
    plan_traffic,
    reference_best_plan,
)


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/tile_planner.py")
    return load_module(path)


def _plan_corpus_traffic(module, corpus):
    total = 0
    for ii, (problems, cap) in enumerate(corpus):
        plan = freeze_plan(module.plan_tiling(problems, cap))
        ok, reason = plan_is_valid(problems, cap, plan)
        if not ok:
            return total, False, f"instance[{ii}]: {reason}"
        total += plan_traffic(problems, plan)
    return total, True, "ok"


def mode_candidate():
    corpus = make_bench_corpus()
    total, ok, reason = _plan_corpus_traffic(load_candidate(), corpus)
    if not ok:
        print(f"INVALID {reason}")
        return 1
    print(f"TRAFFIC={total} VALID=1")
    return 0


def mode_baseline():
    corpus = make_bench_corpus()
    total, ok, reason = _plan_corpus_traffic(_baseline_module(), corpus)
    if not ok:
        print(f"INVALID {reason}")
        return 1
    print(f"TRAFFIC={total} VALID=1")
    return 0


def mode_verify():
    corpus = make_bench_corpus()
    cand = load_candidate()
    base = _baseline_module()
    bad = 0
    for ii, (problems, cap) in enumerate(corpus):
        cok, creason = plan_is_valid(problems, cap, freeze_plan(cand.plan_tiling(problems, cap)))
        bok, breason = plan_is_valid(problems, cap, freeze_plan(base.plan_tiling(problems, cap)))
        if not cok:
            print(f"VERIFY_FAIL candidate instance[{ii}]: {creason}")
            bad += 1
        if not bok:
            print(f"VERIFY_FAIL baseline instance[{ii}]: {breason}")
            bad += 1
    if bad:
        return 1
    naive_total = sum(naive_traffic(problems, cap) for problems, cap in corpus)
    ref_total = 0
    for problems, cap in corpus:
        ref_total += plan_traffic(problems, reference_best_plan(problems, cap))
    print(f"VERIFY_OK instances={len(corpus)} naive_traffic={naive_total} "
          f"reference_best_traffic={ref_total}")
    return 0


MODES = {"candidate": mode_candidate, "baseline": mode_baseline, "verify": mode_verify}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "candidate"
    fn = MODES.get(mode)
    if fn is None:
        print(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
        sys.exit(2)
    sys.exit(fn())


if __name__ == "__main__":
    main()
