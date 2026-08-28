"""Canonical combinatorial kernel (pure Python/NumPy).

Everything here is solver-free. The C variant in ``pmedian_core.c`` mirrors
this module function-for-function; ``core_c.py`` auto-selects it when the
shared library is built and falls back here otherwise. Any semantic change
must land in this file first and be mirrored in C afterwards (house rule
inherited from tmc-pbp).

Instance convention: ``C`` is an (m, n) cost matrix — m candidate facilities
(rows), n clients (columns), ``C[i, j]`` the cost of facility i serving
client j. A solution S is a set of p open rows and
``f(S) = sum_j min_{i in S} C[i, j]``.

In Hammer-Beresnev terms each client's ascending-sorted column is its nested
monomial chain; the kernel exposes exactly the pieces of that structure the
solvers and the tolerance layer consume (sorted orders, truncation caps,
levels).
"""

import time

import numpy as np

EPS = 1e-6


def cost_of(C, S):
    """f(S) = sum over clients of the cheapest open facility."""
    return float(np.asarray(C)[list(S), :].min(axis=0).sum())


def column_order(C):
    """Stable ascending facility order per client column (m x n)."""
    return np.argsort(C, axis=0, kind="stable")


def truncated_matrix(C, D):
    """Cap every client at its (D+1)-th nearest facility — drop all
    Hammer-Beresnev monomials of degree > D. Exact lower-bounding transform:
    costs can only decrease."""
    m = C.shape[0]
    if D >= m - 1:
        return C.copy()
    cap = np.sort(C, axis=0)[D]
    return np.minimum(C, cap[None, :])


def swap_descent(C, p, S0=None, max_iters=200, seed=0):
    """Best-improvement 1-swap local search. Returns (S, f, iters).

    Vectorized: per removed position the column costs without that facility
    come from the top-2 structure; all insertions are evaluated in one
    (m x n) pass. Feasible incumbents for MIP starts and upper bounds.
    """
    C = np.asarray(C, dtype=float)
    m, n = C.shape
    rng = np.random.default_rng(seed)
    S = sorted(rng.choice(m, size=p, replace=False).tolist()) if S0 is None \
        else sorted(S0)
    f = float(C[S, :].min(axis=0).sum())
    for it in range(max_iters):
        sub = C[S, :]
        amin = np.argmin(sub, axis=0)
        if p >= 2:
            part = np.partition(sub, 1, axis=0)
            serve1, serve2 = part[0], part[1]
        else:
            serve1 = sub[0]
            serve2 = np.full(n, np.inf)
        outside = np.array([i for i in range(m) if i not in set(S)])
        if len(outside) == 0:
            return tuple(S), f, it
        best = (0.0, None, None)
        for k in range(p):
            cw = np.where(amin == k, serve2, serve1)
            f_all = np.minimum(cw[None, :], C[outside, :]).sum(axis=1)
            kk = int(np.argmin(f_all))
            if f_all[kk] < f + best[0] - 1e-9:
                best = (float(f_all[kk]) - f, k, int(outside[kk]))
        if best[1] is None:
            return tuple(S), f, it
        S[best[1]] = best[2]
        S = sorted(S)
        f = f + best[0]
    return tuple(S), f, max_iters


def swap_T1(C2, S_star, f_star):
    """Best 1-swap upper bound on min_{S != S*} f_{C2}(S) - f_star,
    restricted to the 1-swap neighbourhood of S*. Returns (T1, seconds)."""
    t0 = time.perf_counter()
    C2 = np.asarray(C2, dtype=float)
    m, n = C2.shape
    Sl = list(S_star)
    p = len(Sl)
    sub = C2[Sl, :]
    amin = np.argmin(sub, axis=0)
    if p >= 2:
        part = np.partition(sub, 1, axis=0)
        serve1, serve2 = part[0], part[1]
    else:
        serve1 = sub[0]
        serve2 = np.full(n, np.inf)
    others = np.array([i for i in range(m) if i not in set(Sl)])
    best = np.inf
    for k in range(p):
        cw = np.where(amin == k, serve2, serve1)
        f_all = np.minimum(cw[None, :], C2[others, :]).sum(axis=1)
        best = min(best, float(f_all.min()))
    return max(best - f_star, 0.0), time.perf_counter() - t0


def serving_split(C, S_star, j0):
    """For client j0 under S*: (i0, a, u_star) — the serving facility, its
    cost, and the strict serving margin u* = second-best-in-S* minus a."""
    i0 = min(S_star, key=lambda i: (C[i, j0], i))
    a = float(C[i0, j0])
    others = [i for i in S_star if i != i0]
    u_star = float(np.asarray(C)[others, j0].min() - a) if others else np.inf
    return i0, a, u_star


def labeled_distance(C1, C2):
    """d(C, C') = sum_j max_i |c_ij - c'_ij| — the labeled pBp distance.
    (The label-free distance does NOT control objective values.)"""
    return float(np.abs(np.asarray(C1) - np.asarray(C2)).max(axis=0).sum())


def reuse_certificate(d, margin2):
    """2 d < margin_2(C) certifies that a cached optimal set remains optimal
    for the perturbed instance — no solve needed. The constant 2 is tight."""
    return 2.0 * d < margin2 - 1e-9
