"""Uniform MILP backend layer: scipy(HiGHS) always, highspy when installed.

Both solve  min c'x  s.t.  lb <= A x <= ub,  lo <= x <= hi,  x_k integer
where integrality[k] = 1. The highspy backend adds MIP starts (incumbent
injection) and is selected automatically when available; a rejected start
never fails a solve. Return convention: (x, fun, seconds, status) with
status 0 = proven optimal, 1 = limit reached, 4 = other.
"""

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

try:
    import highspy
    HAVE_HIGHSPY = True
except ImportError:
    HAVE_HIGHSPY = False


def solve_scipy(c, A, lb, ub, lo, hi, integrality, time_limit=300.0,
                start=None):
    """scipy.optimize.milp path; `start` accepted and ignored."""
    t0 = time.perf_counter()
    res = milp(
        c,
        constraints=[LinearConstraint(A, lb, ub)],
        integrality=integrality,
        bounds=Bounds(lo, hi),
        options={"mip_rel_gap": 0.0, "time_limit": time_limit},
    )
    dt = time.perf_counter() - t0
    if res.status != 0:
        return None, None, dt, res.status
    return res.x, float(res.fun), dt, 0


def solve_highs(c, A, lb, ub, lo, hi, integrality, time_limit=300.0,
                start=None):
    """highspy path with optional MIP start."""
    if not HAVE_HIGHSPY:
        raise RuntimeError("highspy not installed — pip install pbp-pmedian[highs]")
    t0 = time.perf_counter()
    Ac = A.tocsc() if not isinstance(A, np.ndarray) else None
    n = len(c)
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = np.asarray(c, dtype=float)
    lp.col_lower_ = np.asarray(lo, dtype=float)
    lp.col_upper_ = np.asarray(hi, dtype=float)
    lp.row_lower_ = np.asarray(lb, dtype=float)
    lp.row_upper_ = np.asarray(ub, dtype=float)
    if Ac is not None:
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = Ac.indptr
        lp.a_matrix_.index_ = Ac.indices
        lp.a_matrix_.value_ = Ac.data
    else:
        dense = np.asarray(A, dtype=float)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = np.arange(0, dense.size + 1, dense.shape[1])
        lp.a_matrix_.index_ = np.tile(np.arange(dense.shape[1]),
                                      dense.shape[0])
        lp.a_matrix_.value_ = dense.ravel()
    integ = np.asarray(integrality)
    lp.integrality_ = [highspy.HighsVarType.kInteger if integ[k] else
                       highspy.HighsVarType.kContinuous for k in range(n)]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.passModel(lp)
    if start is not None:
        sol = highspy.HighsSolution()
        sol.col_value = np.asarray(start, dtype=float)
        try:
            h.setSolution(sol)
        except Exception:
            pass
    h.run()
    dt = time.perf_counter() - t0
    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        x = np.array(h.getSolution().col_value)
        return x, float(h.getInfo().objective_function_value), dt, 0
    if status in (highspy.HighsModelStatus.kTimeLimit,
                  highspy.HighsModelStatus.kIterationLimit):
        return None, None, dt, 1
    return None, None, dt, 4


BACKENDS = {"scipy": solve_scipy}
if HAVE_HIGHSPY:
    BACKENDS["highs"] = solve_highs

DEFAULT = "highs" if HAVE_HIGHSPY else "scipy"
