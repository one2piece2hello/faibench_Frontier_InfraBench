"""Correctness suite for the tile-size planning contract (ACCELERATION lane).

Correctness here = plan VALIDITY, checked against an INDEPENDENT model
(tile_harness.plan_is_valid), never against a reference plan. A candidate may reach any
valid plan; it is scored on validity + its own plan's total traffic. Meta / anti-hacking
cases confirm the checker REJECTS an illegal plan (a low-traffic tiling that exceeds the
on-chip capacity, or one drawn from outside the allowed choices), so a tiny traffic
bought by an invalid tiling cannot score, and that a plan is validated and charged as
the SAME materialised tiles, so a lazy plan object cannot be checked on one set of tiles
and charged for another.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
"""

import sys
import traceback

from tile_harness import (
    footprint_of,
    freeze_plan,
    load_candidate,
    make_bench_corpus,
    naive_plan,
    naive_traffic,
    plan_is_valid,
    plan_traffic,
    reference_best_plan,
    traffic_of,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _divs(n, limit):
    return [d for d in range(1, limit + 1) if n % d == 0]


def _prob(M, N, K, esz, lm, ln, lk):
    return {"M": M, "N": N, "K": K, "esz": esz,
            "tm_choices": _divs(M, lm), "tn_choices": _divs(N, ln), "tk_choices": _divs(K, lk)}


def _assert_valid(mod, problems, cap, label=""):
    plan = freeze_plan(mod.plan_tiling(problems, cap))
    ok, reason = plan_is_valid(problems, cap, plan)
    if not ok:
        raise AssertionError(f"candidate plan INVALID {label}: {reason}")
    return plan


@case
def normal_valid_complete(mod):
    problems = [_prob(256, 256, 256, 2, 128, 128, 64)]
    plan = _assert_valid(mod, problems, 24 * 1024, "[normal]")
    if len(plan) != 1 or len(list(plan[0])) != 3:
        raise AssertionError("plan must be one [Tm,Tn,Tk] per problem")


@case
def multi_problem_valid(mod):
    problems = [_prob(256, 256, 256, 2, 128, 128, 64),
                _prob(512, 128, 256, 2, 128, 128, 64),
                _prob(128, 384, 192, 2, 128, 128, 64)]
    _assert_valid(mod, problems, 24 * 1024, "[multi]")


@case
def capacity_reject(mod):
    # meta: a low-traffic tiling that exceeds cap must be rejected; candidate valid.
    problems = [_prob(256, 256, 256, 2, 256, 256, 256)]
    cap = 8 * 1024
    over = [[256, 256, 256]]                    # footprint = (256*256*3)*2 >> 8KiB
    ok, reason = plan_is_valid(problems, cap, over)
    if ok:
        raise AssertionError("checker FAILED to reject an over-capacity tiling")
    if "cap" not in reason and "exceed" not in reason:
        raise AssertionError(f"rejection should cite capacity; got: {reason}")
    _assert_valid(mod, problems, cap, "[cap-guard]")


@case
def out_of_choices_reject(mod):
    # meta: a tile not drawn from the choice lists must be rejected.
    problems = [_prob(256, 256, 256, 2, 64, 64, 64)]
    bad = [[96, 64, 64]]                         # 96 divides 256? no -> not in choices
    ok, reason = plan_is_valid(problems, 64 * 1024, bad)
    if ok:
        raise AssertionError("checker FAILED to reject an out-of-choices tile")
    _assert_valid(mod, problems, 64 * 1024, "[choice-guard]")


@case
def smallest_tile_always_fits(mod):
    # the smallest tile of each axis must be within any positive cap (footprint tiny).
    problems = [_prob(512, 512, 512, 4, 256, 256, 256)]
    cap = 4 * 1024
    small = [[min(problems[0]["tm_choices"]), min(problems[0]["tn_choices"]),
              min(problems[0]["tk_choices"])]]
    ok, _ = plan_is_valid(problems, cap, small)
    if not ok:
        raise AssertionError("smallest tile must fit any positive cap")
    _assert_valid(mod, problems, cap, "[smallest fits]")


@case
def reference_headroom_exists(mod):
    # a capacity-respecting tiling moves strictly less than the smallest-tile baseline.
    problems = [_prob(256, 256, 256, 2, 128, 128, 64)]
    cap = 24 * 1024
    naive = naive_traffic(problems, cap)
    ref = plan_traffic(problems, reference_best_plan(problems, cap))
    if not (ref < naive):
        raise AssertionError(f"expected traffic headroom; ref={ref} naive={naive}")
    _assert_valid(mod, problems, cap, "[headroom]")


@case
def degenerate_single_choice(mod):
    # each axis offers exactly one tile -> the plan is forced; still must be valid.
    problems = [{"M": 64, "N": 64, "K": 64, "esz": 2,
                 "tm_choices": [64], "tn_choices": [64], "tk_choices": [64]}]
    cap = footprint_of(problems[0], [64, 64, 64])   # exactly fits
    plan = _assert_valid(mod, problems, cap, "[single-choice]")
    if list(plan[0]) != [64, 64, 64]:
        raise AssertionError(f"forced tiling must be [64,64,64], got {plan[0]}")


@case
def degenerate_single_problem(mod):
    problems = [_prob(128, 128, 128, 4, 64, 64, 64)]
    _assert_valid(mod, problems, 32 * 1024, "[single problem]")


@case
def error_bad_dimension(mod):
    _expect(mod, [{"M": 0, "N": 64, "K": 64, "esz": 2, "tm_choices": [1],
                   "tn_choices": [1], "tk_choices": [1]}], 4096, ValueError, "M=0")
    _expect(mod, [{"M": 64, "N": 64, "K": 64, "esz": -1, "tm_choices": [1],
                   "tn_choices": [1], "tk_choices": [1]}], 4096, ValueError, "esz<0")
    _expect(mod, [{"M": True, "N": 64, "K": 64, "esz": 2, "tm_choices": [1],
                   "tn_choices": [1], "tk_choices": [1]}], 4096, TypeError, "bool M")


@case
def error_bad_choices(mod):
    _expect(mod, [{"M": 64, "N": 64, "K": 64, "esz": 2, "tm_choices": [],
                   "tn_choices": [1], "tk_choices": [1]}], 4096, ValueError, "empty choices")
    # 5 does not divide 64 -> invalid choice
    _expect(mod, [{"M": 64, "N": 64, "K": 64, "esz": 2, "tm_choices": [5],
                   "tn_choices": [1], "tk_choices": [1]}], 4096, ValueError, "non-divisor choice")


@case
def error_bad_cap(mod):
    problems = [_prob(64, 64, 64, 2, 64, 64, 64)]
    _expect(mod, problems, 0, ValueError, "cap=0")
    _expect(mod, problems, -8, ValueError, "cap<0")
    _expect(mod, problems, True, TypeError, "bool cap")


@case
def metamorphic_traffic_scale_esz(mod):
    # doubling esz doubles both the traffic and the footprint of any fixed tile.
    p1 = _prob(128, 128, 128, 2, 64, 64, 64)
    p2 = _prob(128, 128, 128, 4, 64, 64, 64)
    tile = [64, 64, 64]
    if traffic_of(p2, tile) != 2 * traffic_of(p1, tile):
        raise AssertionError("traffic not linear in esz")
    if footprint_of(p2, tile) != 2 * footprint_of(p1, tile):
        raise AssertionError("footprint not linear in esz")


@case
def work_evidence_traffic_matches_model(mod):
    # the candidate's own plan traffic must equal the independent model on its tiles.
    problems = [_prob(256, 256, 256, 2, 128, 128, 64)]
    cap = 24 * 1024
    plan = _assert_valid(mod, problems, cap, "[work-evidence]")
    recomputed = sum(traffic_of(p, list(t)) for p, t in zip(problems, plan))
    if recomputed != plan_traffic(problems, plan):
        raise AssertionError("plan traffic disagrees with the independent model")


@case
def anti_hacking_min_traffic_over_cap_rejected(mod):
    # the globally-minimal-traffic tiling (largest tiles) is illegal when it exceeds
    # cap; the checker must reject it and the candidate must still be valid.
    problems = [_prob(256, 256, 256, 2, 256, 256, 256)]
    cap = 6 * 1024
    biggest = [[max(problems[0]["tm_choices"]), max(problems[0]["tn_choices"]),
                max(problems[0]["tk_choices"])]]
    ok, _ = plan_is_valid(problems, cap, biggest)
    if ok:
        raise AssertionError("checker FAILED to reject the over-cap minimal-traffic tiling")
    _assert_valid(mod, problems, cap, "[anti-hack]")


@case
def anti_hacking_plan_materialized_once(mod):
    # meta: a plan must be validated and charged as the SAME materialised tiles. Scoring
    # iterates the candidate's return value twice (once to check validity, once to sum
    # the traffic), so a lazy object that yields valid tiles first and cheaper ones
    # afterwards would be checked on one set of tiles and charged for another. Both
    # consumers therefore have to go through tile_harness.freeze_plan.
    problems = [_prob(256, 256, 256, 2, 128, 128, 64)]
    cap = 24 * 1024
    valid = reference_best_plan(problems, cap)
    cheap = [[10 ** 9, 10 ** 9, 1]]  # far below the traffic minimum, and far over cap

    class Shifting:
        """Valid tiles on the first iteration, cheap ones afterwards."""

        def __init__(self, first, later):
            self.first, self.later, self.rounds = first, later, 0

        def __len__(self):
            return len(self.first)

        def __getitem__(self, index):
            return self.first[index]

        def __iter__(self):
            self.rounds += 1
            return iter(self.first if self.rounds == 1 else self.later)

    frozen = freeze_plan(Shifting(valid, cheap))
    if plan_traffic(problems, frozen) != plan_traffic(problems, valid):
        raise AssertionError("freeze_plan must keep the tiles a validity pass would see")
    if not plan_is_valid(problems, cap, frozen)[0]:
        raise AssertionError("materialised plan must validate on the tiles it froze")
    _assert_valid(mod, problems, cap, "[materialised]")


def _expect(mod, problems, cap, exc, label):
    try:
        mod.plan_tiling(problems, cap)
    except exc:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"{label}: raised {type(other).__name__}, expected {exc.__name__}")
    raise AssertionError(f"{label}: did not raise {exc.__name__}")


def main():
    mod = load_candidate()
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            fn_case(mod)
            passed += 1
            print(f"CASE_PASS {name}")
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            print(f"CASE_FAIL {name}: {reason.splitlines()[0][:300]}")
            traceback.print_exc(file=sys.stderr)
    total = len(CASES)
    print(f"CASES_PASSED={passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
