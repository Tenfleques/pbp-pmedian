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

import time

import numpy as np

from .backends import HAVE_HIGHSPY
from .core import EPS, serving_split
from .model import lp_prices, solve_full
from .solve import solve_expand


def b2_profile_warm(C, p, S_star, f_star, clients=None, warm=True,
                    time_limit=600.0):
    """The whole-instance B2 profile by a warm-started dual simplex.

    B2(i0, j0) = LP(C~; sum_{i in S*} y_i <= p-1) - f* is the strongest
    safe lower bound of the ladder (exact on 57.4% of benchmark entries,
    with a per-entry integrality certificate). Consecutive entries differ
    in ONE objective coefficient — the perturbed assignment variable's cost
    rises to its column maximum — so one exclusion LP is built once and
    each subsequent entry is a changeColCost + hot re-solve from the
    previous optimal basis (requires highspy). warm=False rebuilds the
    model per entry: the same-stack cold baseline the speedup is measured
    against.

    Returns a dict of per-client arrays (client, facility, a, u_star,
    degenerate, B2, cert, iters, t_s) and scalars (t_build_s, t_total_s).
    cert: the relaxation returned integral y, so B2 equals d_min exactly
    (the certificate proposition); iters: simplex iterations per re-solve.
    """
    if not HAVE_HIGHSPY:
        raise RuntimeError("b2_profile_warm needs highspy — "
                           "pip install pbp-pmedian[highs]")
    import highspy

    C = np.asarray(C, dtype=float)
    m, n = C.shape
    nx = m * n
    S_sorted = sorted(int(i) for i in S_star)

    def build():
        from scipy import sparse
        lp = highspy.HighsLp()
        lp.num_col_ = m + nx
        # rows: n assignment eq, 1 cardinality eq, nx Balinski <=0, 1 excl
        rows = [np.repeat(np.arange(n), m), np.full(m, n)]
        cols = [(m + np.arange(m)[None, :] * n
                 + np.arange(n)[:, None]).ravel(), np.arange(m)]
        vals = [np.ones(n * m), np.ones(m)]
        r_bal = n + 1 + np.arange(nx)
        rows += [r_bal, r_bal, np.full(len(S_sorted), n + 1 + nx)]
        cols += [m + np.arange(nx), np.repeat(np.arange(m), n),
                 np.array(S_sorted)]
        vals += [np.ones(nx), -np.ones(nx), np.ones(len(S_sorted))]
        A = sparse.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(n + 1 + nx + 1, m + nx)).tocsc()
        lp.num_row_ = A.shape[0]
        lp.col_cost_ = np.concatenate([np.zeros(m), C.ravel()])
        lp.col_lower_ = np.zeros(m + nx)
        lp.col_upper_ = np.ones(m + nx)
        lp.row_lower_ = np.concatenate(
            [np.ones(n), [float(p)], np.full(nx, -highspy.kHighsInf),
             [-highspy.kHighsInf]])
        lp.row_upper_ = np.concatenate(
            [np.ones(n), [float(p)], np.zeros(nx), [float(p - 1)]])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.passModel(lp)
        return h

    t0 = time.perf_counter()
    h = build()
    t_build = time.perf_counter() - t0
    out = {k: [] for k in ("client", "facility", "a", "u_star",
                           "degenerate", "B2", "cert", "iters", "t_s")}
    prev = None  # (col, original cost) to restore
    cols_iter = clients if clients is not None else range(n)
    t_all0 = time.perf_counter()
    for j0 in cols_iter:
        i0, _, u_star = serving_split(C, S_star, j0)
        a = float(C[i0, j0])
        col = m + int(i0) * n + int(j0)
        cmax = float(C[:, j0].max())
        t1 = time.perf_counter()
        if not warm:
            h = build()
        if prev is not None:
            h.changeColCost(prev[0], prev[1])
        h.changeColCost(col, cmax)
        prev = (col, a)
        h.run()
        dt = time.perf_counter() - t1
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            b2, cert, iters = None, False, -1
        else:
            b2 = max(h.getInfo().objective_function_value - f_star, 0.0)
            y = np.array(h.getSolution().col_value[:m])
            cert = bool(np.abs(y - np.round(y)).max() <= 1e-6)
            iters = int(h.getInfo().simplex_iteration_count)
        out["client"].append(int(j0))
        out["facility"].append(int(i0))
        out["a"].append(a)
        out["u_star"].append(float(u_star))
        out["degenerate"].append(bool(u_star <= EPS))
        out["B2"].append(b2)
        out["cert"].append(cert)
        out["iters"].append(iters)
        out["t_s"].append(dt)
    res = {k: np.array(v) for k, v in out.items()}
    res["t_build_s"] = t_build
    res["t_total_s"] = time.perf_counter() - t_all0 + t_build
    return res


def dual_price_screen(C, p, S_star, f_star, lp=None, time_limit=300.0):
    """Free lower bounds on the closed facilities' forcing-in prices, read
    off the plain relaxation's reduced costs.

    For a facility i CLOSED at the optimum, its forcing-in price
    T_i = OPT(C; y_i = 1) - f* (the facility's lower tolerance) satisfies

        T_i  >=  rc_i - g,        g = f* - LP(C)  (the integrality gap),

    because OPT(C; y_i=1) >= LP(C; y_i=1) >= LP(C) + rc_i. When the
    relaxation is tight (g = 0, ``certified``) the reduced cost itself is a
    valid lower bound, at zero cost beyond the one LP the pipeline already
    solves. Calibration on pmed1-10 + synthetics (tolerance paper, the
    "textbook prices" section): safe 165/165 in the tight regime, Spearman
    0.97 where rc > 0, median rc/T = 0.68 — a ranker/screen, not a value;
    about a fifth of finite cases underestimate by more than 10x.

    Two structural silences, by design: OPEN facilities are basic (rc = 0
    identically — the report never prices losing a facility; use
    facility_tolerance for that), and the plain relaxation carries no
    usable ENTRY-level estimate (use the F2 screen / entry_tolerance).

    lp: a model.lp_prices(C, p) result to reuse (no extra solve); solved
    here if omitted. Returns a dict:
      facility   (k,) closed-facility indices (ascending)
      rc         (k,) their reduced costs
      lower      (k,) gap-corrected lower bounds max(rc - g, 0) on T_i
      gap_abs    g = max(f* - LP, 0)
      certified  True iff g <= EPS (then lower == rc, valid as-is)
      lp_value, informative (count rc > EPS), n_closed, seconds, status
    """
    C = np.asarray(C, dtype=float)
    m = C.shape[0]
    if lp is None:
        lp = lp_prices(C, p, time_limit=time_limit)
    if lp["status"] != 0:
        return {"facility": None, "rc": None, "lower": None,
                "gap_abs": None, "certified": False, "lp_value": None,
                "informative": None, "n_closed": None,
                "seconds": lp["seconds"], "status": lp["status"]}
    g = max(float(f_star) - lp["value"], 0.0)
    closed = np.array(sorted(set(range(m)) - set(S_star)), dtype=int)
    rc = lp["rc_y"][closed]
    return {"facility": closed, "rc": rc,
            "lower": np.maximum(rc - g, 0.0),
            "gap_abs": g, "certified": bool(g <= EPS),
            "lp_value": lp["value"],
            "informative": int((rc > EPS).sum()),
            "n_closed": int(len(closed)),
            "seconds": lp["seconds"], "status": 0}


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
