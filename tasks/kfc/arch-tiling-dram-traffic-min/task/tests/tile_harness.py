"""Shared ground-truth harness for the tile-size planning task (CPU, pure Python).

Provides candidate/module loaders, an INDEPENDENT off-chip-traffic model and plan-
validity checker (the correctness ground truth), the total-traffic metric, an
exhaustive reference optimizer (headroom evidence only -- NOT a scoring oracle), and a
deterministic problem corpus. Standard library only (no numpy / torch) so the metric is
a deterministic COMPUTED byte total with no measurement noise.
"""

import hashlib
import importlib.util
import os


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    path = os.path.join(repo_dir(), "tile_planner.py")
    spec = importlib.util.spec_from_file_location("candidate_tile_planner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "plan_tiling"):
        raise AttributeError(f"{path} does not define plan_tiling")
    return mod


def load_module(path):
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location("kb_tile_mod_" + digest, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Independent off-chip traffic model and capacity footprint.
#   moved   = esz * ( M*K*(N//Tn) + K*N*(M//Tm) + 2*M*N )
#   footprint = (Tm*Tk + Tk*Tn + Tm*Tn) * esz
# --------------------------------------------------------------------------- #
def traffic_of(problem, tile):
    M, N, K, esz = problem["M"], problem["N"], problem["K"], problem["esz"]
    Tm, Tn, Tk = tile
    return esz * (M * K * (N // Tn) + K * N * (M // Tm) + 2 * M * N)


def footprint_of(problem, tile):
    Tm, Tn, Tk = tile
    return (Tm * Tk + Tk * Tn + Tm * Tn) * problem["esz"]


def _tile_allowed(problem, tile):
    Tm, Tn, Tk = tile
    return (Tm in problem["tm_choices"] and Tn in problem["tn_choices"]
            and Tk in problem["tk_choices"])


def plan_is_valid(problems, cap, plan):
    """Return (ok, reason). A plan is valid iff it assigns one [Tm,Tn,Tk] per problem,
    each drawn from that problem's choice lists, each fitting the capacity."""
    try:
        items = list(plan)
    except Exception:
        return False, "plan is not a list"
    if len(items) != len(problems):
        return False, f"plan has {len(items)} tiles for {len(problems)} problems"
    for pi, (problem, tile) in enumerate(zip(problems, items)):
        try:
            t = list(tile)
        except Exception:
            return False, f"problem {pi}: tile is not a sequence"
        if len(t) != 3:
            return False, f"problem {pi}: tile must be [Tm, Tn, Tk], got {t}"
        for x in t:
            if isinstance(x, bool) or not isinstance(x, int):
                return False, f"problem {pi}: tile entries must be ints, got {t}"
        if not _tile_allowed(problem, t):
            return False, f"problem {pi}: tile {t} not drawn from the allowed choices"
        fp = footprint_of(problem, t)
        if fp > cap:
            return False, f"problem {pi}: tile {t} footprint {fp} exceeds cap {cap}"
    return True, "ok"


def freeze_plan(plan):
    """Materialise a plan exactly once, for validation AND for scoring.

    ``plan_is_valid`` consumes the object it is handed and so does ``plan_traffic``, so a
    caller that passes the candidate's raw return value to both can be fooled by a lazy
    object whose first iteration yields valid tiles and whose later iterations yield
    something cheaper: validity would be checked on one set of tiles and the traffic
    charged for another. Materialise here and hand the frozen list to both.
    """
    try:
        return [list(tile) for tile in plan]
    except TypeError:
        return plan  # not iterable -- plan_is_valid reports the reason


def plan_traffic(problems, plan):
    """Total off-chip bytes moved by a (assumed valid) plan."""
    return sum(traffic_of(p, list(t)) for p, t in zip(problems, plan))


def naive_plan(problems, cap):
    """Smallest tile of each axis (the baseline behaviour) -> maximal traffic."""
    return [[min(p["tm_choices"]), min(p["tn_choices"]), min(p["tk_choices"])] for p in problems]


def naive_traffic(problems, cap):
    return plan_traffic(problems, naive_plan(problems, cap))


# --------------------------------------------------------------------------- #
# Independent reference: exhaustive minimum-traffic valid tiling. Used only to report
# that real headroom exists; NOT a scoring oracle (the candidate is scored on validity
# + its own plan's traffic, never on matching this reference).
# --------------------------------------------------------------------------- #
def reference_best_plan(problems, cap):
    plan = []
    for p in problems:
        best_tile = None
        best_traf = None
        for Tm in p["tm_choices"]:
            for Tn in p["tn_choices"]:
                for Tk in p["tk_choices"]:
                    tile = [Tm, Tn, Tk]
                    if footprint_of(p, tile) > cap:
                        continue
                    traf = traffic_of(p, tile)
                    # tie-break deterministically: fewer bytes, then larger Tm, Tn, Tk
                    key = (traf, -Tm, -Tn, -Tk)
                    if best_traf is None or key < best_traf:
                        best_traf = key
                        best_tile = tile
        plan.append(best_tile)
    return plan


# --------------------------------------------------------------------------- #
# Deterministic problem corpus.
# --------------------------------------------------------------------------- #
def _divisors_upto(n, limit):
    return [d for d in range(1, limit + 1) if n % d == 0]


def make_bench_corpus():
    """A fixed list of (problems, cap) instances. Deterministic; the traffic is a pure
    computed byte total so there is no measurement noise. Shapes are chosen so a
    capacity-respecting large tiling moves far less data than the smallest-tile
    baseline, but the largest tiles do NOT fit -> a real constrained search.

    Every axis offers ALL divisors of its dimension. The traffic depends only on Tm and
    Tn, and the two compete for the same capacity, so a bigger Tm is paid for with a
    smaller Tn: choosing them one axis at a time (grow Tm as far as it fits, then Tn)
    overshoots. Capping the choice ranges hid that trade-off -- the one-axis-at-a-time
    choice landed exactly on the traffic minimum, which left the metric with no headroom
    over its own reference solution (see tests/ref_speedup.caveat.md)."""
    instances = []
    # instance A: three square-ish GEMMs, esz=2 (fp16), cap = 24 KiB.
    probsA = []
    for (M, N, K) in ((256, 256, 256), (512, 128, 256), (128, 384, 192)):
        probsA.append({
            "M": M, "N": N, "K": K, "esz": 2,
            "tm_choices": _divisors_upto(M, M),
            "tn_choices": _divisors_upto(N, N),
            "tk_choices": _divisors_upto(K, 64),
        })
    instances.append((probsA, 24 * 1024))
    # instance B: tall-skinny + wide, esz=4 (fp32), cap = 48 KiB.
    probsB = []
    for (M, N, K) in ((768, 64, 320), (96, 640, 256)):
        probsB.append({
            "M": M, "N": N, "K": K, "esz": 4,
            "tm_choices": _divisors_upto(M, M),
            "tn_choices": _divisors_upto(N, N),
            "tk_choices": _divisors_upto(K, 64),
        })
    instances.append((probsB, 48 * 1024))
    return instances
