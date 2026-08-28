"""Per-entry cost tolerances — the layer this package exists for.

For a serving entry (i0, j0) of an optimal set S*, the upper tolerance is
the largest increase of C[i0, j0] that keeps S* optimal. It equals
d_min = min_{S != S*}(margin_S + u_S) when that is below the serving margin
u*, and is infinite (saturated) otherwise. Computing it exactly costs one
constrained re-solve per entry in general (and is NP-hard to approximate,
van Hoesel-Wagelmans); this module implements the branch-split algorithm
validated on the OR-Library/TSPLIB benchmark set (438/438 entries exact,
median 7.3x faster than the exact re-solve on the same solver stack):

    d_min = min(A_{i0}, B),
    A_{i0} = OPT(C; y_{i0} = 0) - f*      (the classical FACILITY tolerance,
                                           client-independent: one certified
                                           row-deleted solve per facility),
    B      = the entry-deleted solve with y_{i0} forced open, S* excluded,
             and client j0's column capped at v_{l*} — exact for the
             tolerance decision by the level-restriction lemma, with the
             column collapsing to l* options.

Guarantee: the returned tolerance is ALWAYS exact; the returned d_min is
exact whenever the tolerance is finite, and a certified lower bound in
[u*, d_min] on saturated entries (the classification is unaffected).
"""

import numpy as np

from .core import EPS, serving_split
from .model import solve_full
from .solve import solve_expand


def facility_tolerance(C, p, i0, f_star, time_limit=300.0, engine="expand"):
    """A_{i0} = OPT(C; y_{i0} = 0) - f*, via the row-deleted submatrix.
    engine='expand' uses the certified expansion loop; 'full' the full model
    (recommended on LP-loose instances, where the reduction does not pay).
    Returns (A, seconds) or (None, seconds) on timeout."""
    C = np.asarray(C, dtype=float)
    Cr = np.delete(C, i0, axis=0)
    if engine == "full":
        val, _, t, st = solve_full(Cr, p, time_limit=time_limit)
    else:
        val, _, t, st, _ = solve_expand(Cr, p, time_limit=time_limit)
    if st != 0:
        return None, t
    return max(val - f_star, 0.0), t


def entry_tolerance(C, p, S_star, f_star, i0, j0, A_i0, time_limit=300.0):
    """Exact upper tolerance of serving entry (i0, j0).
    Returns (tol, d_min, B, seconds, status)."""
    C = np.asarray(C, dtype=float)
    m, n = C.shape
    i0_chk, a, u_star = serving_split(C, S_star, j0)
    if i0_chk != i0:
        raise ValueError(f"({i0},{j0}) is not the serving entry; "
                         f"client {j0} is served by {i0_chk}")
    if u_star <= EPS:
        return np.inf, np.nan, np.nan, 0.0, 0   # degenerate: no strict margin
    C2 = C.copy()
    C2[i0, j0] = C[:, j0].max()
    others = [i for i in S_star if i != i0]
    v_lstar = C[others, j0].min()
    C2[:, j0] = np.minimum(C2[:, j0], v_lstar)
    # pin j0's depth to cover everything strictly below v_{l*}: the overflow
    # price then equals v_{l*}, the exact price the lemma justifies
    k = max(int((C2[:, j0] < v_lstar - 1e-9).sum()), 1)
    cap_val = np.sort(C2[:, j0])[k] if k < m else None
    assert cap_val is None or abs(cap_val - v_lstar) <= 1e-9, (
        "legal-cap price != v_{l*} — pinning failed")
    val, _, t_s, st, _ = solve_expand(
        C2, p, exclude=S_star, force_open=[i0], time_limit=time_limit,
        legal_cap=j0, depth_overrides={j0: k})
    if st != 0:
        return np.nan, np.nan, np.nan, t_s, st
    B = max(val - f_star, 0.0)
    d_min = min(A_i0, B)
    tol = d_min if d_min < u_star - EPS else np.inf
    return tol, d_min, B, t_s, 0


def margin2(C, p, S_star, f_star, time_limit=300.0):
    """margin_2 = f(second-best set) - f*, one exclusion solve."""
    val, _, t, st = solve_full(C, p, exclude=S_star, time_limit=time_limit)
    if st != 0:
        return None, t
    return max(val - f_star, 0.0), t


def f2_screen(u_star, m2):
    """The free safe screen min(u*, margin_2): a certified LOWER bound on the
    tolerance (infinite when it reaches u*)."""
    b = min(u_star, m2)
    return np.inf if b >= u_star - EPS else b


def tolerance_profile(C, p, S_star=None, f_star=None, clients=None,
                      time_limit=300.0, a_engine="expand"):
    """Exact upper-tolerance profile over serving entries.

    Solves the base problem if S* is not supplied (certified expand),
    computes one facility tolerance per touched facility (cached), then one
    level-restricted entry solve per client. Returns a dict of aligned
    arrays: client, facility, u_star, d_min, tol, plus f_star/S_star.
    Degenerate clients (u* <= EPS) are reported with tol = inf, d_min = nan.
    """
    C = np.asarray(C, dtype=float)
    m, n = C.shape
    if S_star is None:
        f_star, S_star, _, st, _ = solve_expand(C, p, time_limit=time_limit)
        if st != 0:
            raise RuntimeError("base solve failed within the time limit")
    elif f_star is None:
        f_star = float(C[list(S_star), :].min(axis=0).sum())
    cache = {}
    out = {"client": [], "facility": [], "u_star": [], "d_min": [], "tol": []}
    for j0 in (range(n) if clients is None else clients):
        i0, a, u_star = serving_split(C, S_star, j0)
        if u_star <= EPS:
            out["client"].append(j0)
            out["facility"].append(i0)
            out["u_star"].append(u_star)
            out["d_min"].append(np.nan)
            out["tol"].append(np.inf)
            continue
        if i0 not in cache:
            cache[i0] = facility_tolerance(C, p, i0, f_star,
                                           time_limit=time_limit,
                                           engine=a_engine)[0]
        A = cache[i0]
        if A is None:
            continue    # facility solve timed out: entry not reported
        tol, d_min, _, _, st = entry_tolerance(
            C, p, S_star, f_star, i0, j0, A, time_limit=time_limit)
        if st != 0:
            continue
        out["client"].append(j0)
        out["facility"].append(i0)
        out["u_star"].append(u_star)
        out["d_min"].append(d_min)
        out["tol"].append(tol)
    return {"f_star": f_star, "S_star": tuple(S_star),
            **{k: np.array(v) for k, v in out.items()}}
