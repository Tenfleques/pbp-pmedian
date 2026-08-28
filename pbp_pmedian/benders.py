"""Two-phase Benders decomposition for the p-median.

Implemented from Duran-Mateluna, Ales & Elloumi, "An efficient Benders
decomposition for the p-median problem" (EJOR 308, 2023; arXiv 2111.13405):
their formulation (F3)/(MP), Lemma 2's closed-form subproblem optimum,
Corollary 1's cuts, and the O(mn) separation walk — which runs over exactly
the Hammer-Beresnev prefix order of each client's sorted column. This is a
callback-free variant: phase 1 is a cutting-plane loop on the LP relaxation,
phase 2 re-solves the MIP master with exact separation at integer points,
terminating with proof (the master value equals the true cost of its own
solution while the master is a relaxation).

Convention mapping: their client i = our column j; their site j = our row i.
"""

import time

import numpy as np
from scipy import sparse

from . import backends
from .core import swap_descent

EPS = 1e-7


class BendersPMedian:
    def __init__(self, C, p, backend=None):
        self.C = np.asarray(C, dtype=float)
        self.m, self.n = self.C.shape
        self.p = p
        name = backend or backends.DEFAULT
        self.solve_milp = backends.BACKENDS[name]
        self.order = np.argsort(self.C, axis=0, kind="stable")
        self.dsort = np.take_along_axis(self.C, self.order, axis=0)
        self.bounds_last = []
        self.radii = []
        for j in range(self.n):
            d = self.dsort[:, j]
            last = np.flatnonzero(np.diff(d) > 1e-12)
            self.bounds_last.append(np.append(last, self.m - 1))
            self.radii.append(d[np.append(last, self.m - 1)])
        self.cuts = []
        self.stats = {"p1_iters": 0, "p2_rounds": 0, "t_p1": 0.0,
                      "t_p2": 0.0, "t_sep": 0.0}

    def separate(self, ybar, theta_bar):
        """Closed-form separation (their Algorithms 1-2), vectorized per
        client. Appends violated cuts; returns (n_violated, sum_opt)."""
        t0 = time.perf_counter()
        n_viol, sum_opt = 0, 0.0
        for j in range(self.n):
            pref = np.cumsum(ybar[self.order[:, j]])
            last = self.bounds_last[j]
            radii = self.radii[j]
            covered = pref[last]
            open_idx = np.flatnonzero(1.0 - covered > 1e-9)
            kb = (int(open_idx[-1]) + 1) if len(open_idx) else 0
            if kb == 0:
                opt_j = radii[0]
                cut = (j, radii[0], np.empty(0, dtype=int), np.empty(0))
            else:
                Dnext = radii[kb]
                upto = last[kb - 1]
                sites = self.order[:upto + 1, j]
                coefs = Dnext - self.dsort[:upto + 1, j]
                opt_j = Dnext - float(coefs @ ybar[sites])
                cut = (j, Dnext, sites.copy(), coefs.copy())
            sum_opt += opt_j
            if theta_bar[j] < opt_j - 1e-6:
                self.cuts.append(cut)
                n_viol += 1
        self.stats["t_sep"] += time.perf_counter() - t0
        return n_viol, sum_opt

    def _master(self, relax, time_limit, start=None):
        m, n = self.m, self.n
        nv = m + n
        c = np.concatenate([np.zeros(m), np.ones(n)])
        lo = np.concatenate([np.zeros(m),
                             np.array([r[0] for r in self.radii])])
        hi = np.concatenate([np.ones(m), np.full(n, np.inf)])
        rows, cols, vals, lb, ub = [], [], [], [], []
        rows.extend([0] * m)
        cols.extend(range(m))
        vals.extend([1.0] * m)
        lb.append(float(self.p))
        ub.append(float(self.p))
        r = 1
        for (j, rhs, sites, coefs) in self.cuts:
            rows.append(r)
            cols.append(m + j)
            vals.append(1.0)
            if len(sites):
                rows.extend([r] * len(sites))
                cols.extend(np.asarray(sites).tolist())
                vals.extend(np.asarray(coefs).tolist())
            lb.append(float(rhs))
            ub.append(np.inf)
            r += 1
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsc()
        integrality = np.concatenate(
            [np.zeros(m) if relax else np.ones(m), np.zeros(n)])
        return self.solve_milp(c, A, np.array(lb), np.array(ub), lo, hi,
                               integrality, time_limit=time_limit,
                               start=start)

    def true_thetas(self, S):
        return self.C[list(S), :].min(axis=0)

    def solve(self, time_limit=600.0, p1_max_iters=300, p2_max_rounds=60,
              verbose=False):
        t_start = time.perf_counter()
        f = None
        for it in range(p1_max_iters):
            remaining = max(time_limit - (time.perf_counter() - t_start), 5.0)
            x, f, dt, st = self._master(relax=True, time_limit=remaining)
            self.stats["t_p1"] += dt
            if st != 0:
                return None, None, time.perf_counter() - t_start, st
            n_viol, _ = self.separate(x[:self.m], x[self.m:])
            self.stats["p1_iters"] = it + 1
            if n_viol == 0:
                break
        lp_bound = f
        Sseed, _, _ = swap_descent(self.C, self.p, seed=0)
        yseed = np.zeros(self.m)
        yseed[list(Sseed)] = 1.0
        start = np.concatenate([yseed, self.true_thetas(Sseed)])
        for rnd in range(p2_max_rounds):
            remaining = max(time_limit - (time.perf_counter() - t_start), 5.0)
            x, f, dt, st = self._master(relax=False, time_limit=remaining,
                                        start=start)
            self.stats["t_p2"] += dt
            self.stats["p2_rounds"] = rnd + 1
            if st != 0:
                return None, None, time.perf_counter() - t_start, st
            y = np.round(x[:self.m])
            S = tuple(sorted(np.flatnonzero(y > 0.5).tolist()))
            n_viol, sum_opt = self.separate(y, x[self.m:])
            if n_viol == 0 or abs(sum_opt - f) <= 1e-6:
                total = time.perf_counter() - t_start
                if verbose:
                    print(f"benders: LP {lp_bound:.1f} -> OPT {sum_opt:.1f} "
                          f"p1={self.stats['p1_iters']} "
                          f"p2={self.stats['p2_rounds']} "
                          f"cuts={len(self.cuts)} t={total:.1f}s")
                return float(sum_opt), S, total, 0
            start = np.concatenate([y, self.true_thetas(S)])
        return None, None, time.perf_counter() - t_start, 1
