"""MILP model builders: the full Balinski model and the reduced
(BEAMR-style, per-client depth) model.

The reduced model keeps assignment variables only for each client's
``depths[j]`` nearest facilities plus one overflow variable priced at the
(depths[j]+1)-th cost — an EXACT solver for the depth-capped matrix with
``sum_j depths[j]`` assignment variables instead of m*n.
"""

import time

import numpy as np
from scipy import sparse

from . import backends
from .core import EPS

_STATE = {"solver": backends.BACKENDS[backends.DEFAULT]}


def set_backend(name):
    """Select 'scipy' or 'highs' for every model solve in this module."""
    _STATE["solver"] = backends.BACKENDS[name]


def solve_full(C, p, exclude=None, force_open=None, time_limit=300.0,
               start_S=None, relax=False):
    """Full Balinski model: min f(S), |S| = p. Returns
    (value, S, seconds, status). exclude: rows whose exact set is forbidden
    (sum_{i in exclude} y_i <= p-1). force_open: rows with y_i = 1.
    relax=True solves the LP relaxation and returns (value, y, seconds, 0).
    """
    C = np.asarray(C, dtype=float)
    m, n = C.shape
    nx = m * n
    c_obj = np.concatenate([np.zeros(m), C.ravel()])
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for j in range(n):
        rows.extend([r] * m)
        cols.extend(m + np.arange(m) * n + j)
        vals.extend([1.0] * m)
        lb.append(1.0)
        ub.append(1.0)
        r += 1
    xi = np.repeat(np.arange(m), n)
    ridx = r + np.arange(nx)
    rows.extend(np.repeat(ridx, 2))
    inter_c = np.empty(2 * nx, dtype=int)
    inter_c[0::2] = m + np.arange(nx)
    inter_c[1::2] = xi
    cols.extend(inter_c)
    inter_v = np.empty(2 * nx)
    inter_v[0::2] = 1.0
    inter_v[1::2] = -1.0
    vals.extend(inter_v)
    lb.extend([-np.inf] * nx)
    ub.extend([0.0] * nx)
    r += nx
    rows.extend([r] * m)
    cols.extend(range(m))
    vals.extend([1.0] * m)
    lb.append(float(p))
    ub.append(float(p))
    r += 1
    if exclude is not None:
        ex = sorted(exclude)
        rows.extend([r] * len(ex))
        cols.extend(ex)
        vals.extend([1.0] * len(ex))
        lb.append(-np.inf)
        ub.append(float(p - 1))
        r += 1
    nvar = m + nx
    lo = np.zeros(nvar)
    if force_open is not None:
        for i in force_open:
            lo[i] = 1.0
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsc()
    integrality = np.concatenate(
        [np.zeros(m) if relax else np.ones(m), np.zeros(nx)])
    start = None
    if start_S is not None and not relax:
        start = np.zeros(nvar)
        sset = set(start_S)
        for i in sset:
            start[i] = 1.0
        assign = C[sorted(sset), :].argmin(axis=0)
        srows = np.array(sorted(sset))
        for j in range(n):
            start[m + srows[assign[j]] * n + j] = 1.0
    x, fun, dt, st = _STATE["solver"](
        c_obj, A, np.array(lb), np.array(ub), lo, np.ones(nvar),
        integrality, time_limit=time_limit, start=start)
    if st != 0:
        return None, None, dt, st
    if relax:
        return float(fun), x[:m], dt, 0
    S = tuple(sorted(np.flatnonzero(x[:m] > 0.5).tolist()))
    return float(fun), S, dt, 0


def reduced_pmedian_ragged(C, depths, p, exclude=None, force_open=None,
                           time_limit=300.0, start_S=None):
    """Exact solver for the depth-capped p-median with PER-CLIENT depths.
    Returns (value, S, t_solve, status, info); info carries the per-client
    overflow flags the expansion loop consumes and the lists used.
    start_S builds a full feasible MIP start (y indicator, cheapest listed
    open facility per client, overflow otherwise)."""
    C = np.asarray(C, dtype=float)
    m, n = C.shape
    depths = np.minimum(np.asarray(depths, dtype=int), m)
    order = np.argsort(C, axis=0, kind="stable")
    t0 = time.perf_counter()
    lists, caps = [], np.full(n, np.nan)
    xvar_col, xvar_fac, xvar_cost = [], [], []
    for j in range(n):
        D = depths[j]
        lj = order[:D, j]
        lists.append(lj)
        xvar_col.extend([j] * D)
        xvar_fac.extend(lj.tolist())
        xvar_cost.extend(C[lj, j].tolist())
        if D < m:
            caps[j] = C[order[D, j], j]
    nx = len(xvar_col)
    has_z = np.isfinite(caps)
    z_idx = {j: m + nx + k for k, j in enumerate(np.flatnonzero(has_z))}
    nz = len(z_idx)
    c_obj = np.concatenate([np.zeros(m), np.array(xvar_cost), caps[has_z]])

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    xvar_col = np.array(xvar_col)
    xvar_fac = np.array(xvar_fac)
    rows.extend(xvar_col.tolist())
    cols.extend((m + np.arange(nx)).tolist())
    vals.extend([1.0] * nx)
    for j in range(n):
        if j in z_idx:
            rows.append(j)
            cols.append(z_idx[j])
            vals.append(1.0)
    lb.extend([1.0] * n)
    ub.extend([1.0] * n)
    r += n
    ridx = r + np.arange(nx)
    rows.extend(np.repeat(ridx, 2).tolist())
    inter_c = np.empty(2 * nx, dtype=int)
    inter_c[0::2] = m + np.arange(nx)
    inter_c[1::2] = xvar_fac
    cols.extend(inter_c.tolist())
    inter_v = np.empty(2 * nx)
    inter_v[0::2] = 1.0
    inter_v[1::2] = -1.0
    vals.extend(inter_v.tolist())
    lb.extend([-np.inf] * nx)
    ub.extend([0.0] * nx)
    r += nx
    rows.extend([r] * m)
    cols.extend(range(m))
    vals.extend([1.0] * m)
    lb.append(float(p))
    ub.append(float(p))
    r += 1
    if exclude is not None:
        ex = sorted(exclude)
        rows.extend([r] * len(ex))
        cols.extend(ex)
        vals.extend([1.0] * len(ex))
        lb.append(-np.inf)
        ub.append(float(p - 1))
        r += 1
    nvar = m + nx + nz
    lo = np.zeros(nvar)
    if force_open is not None:
        for i in force_open:
            lo[i] = 1.0
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsc()
    integrality = np.concatenate([np.ones(m), np.zeros(nx + nz)])
    start = None
    if start_S is not None:
        sset = set(start_S)
        start = np.zeros(nvar)
        for i in sset:
            start[i] = 1.0
        pos = 0
        for j in range(n):
            D = len(lists[j])
            in_S = [k for k in range(D) if lists[j][k] in sset]
            if in_S:
                start[m + pos + in_S[0]] = 1.0
            elif j in z_idx:
                start[z_idx[j]] = 1.0
            else:
                start = None
                break
            pos += D
    t_build = time.perf_counter() - t0
    x, fun, dt, status = _STATE["solver"](
        c_obj, A, np.array(lb), np.array(ub), lo, np.ones(nvar), integrality,
        time_limit=time_limit, start=start)
    info = {"t_build": t_build, "n_xvars": nx, "lists": lists,
            "overflow": np.zeros(n, dtype=bool)}
    if status != 0:
        return None, None, dt, status, info
    S = tuple(sorted(np.flatnonzero(x[:m] > 0.5).tolist()))
    for j, k in z_idx.items():
        info["overflow"][j] = x[k] > EPS
    return float(fun), S, dt, 0, info
