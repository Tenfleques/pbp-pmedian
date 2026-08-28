"""The engine router.

Empirical rule (calibrated on 50 OR-Library / TSPLIB / synthetic / random
instances against the fastest-engine oracle, matching its total wall-clock
to within measurement noise; the rule is post-hoc calibrated and should be
re-confirmed out-of-sample before being treated as a law):

    rho = p / m >= 0.10        -> expand-until-certificate
    rho <  0.10                -> Benders
    LP-loose random instances  -> full model (the reduction does not pay
                                  there; detectable via the gap estimate)

The one-LP integrality gap additionally predicts the TOLERANCE-bound regime
(zero gap: cheap LP bounds certify; positive gap: they fail) — that is a
separate dispatch from the engine choice above.
"""

import numpy as np

from .benders import BendersPMedian
from .core import swap_descent
from .model import solve_full
from .solve import solve_expand


def gap_estimate(C, p, time_limit=300.0, seed=0):
    """(UB - LP)/UB with UB from swap descent: an UPPER estimate of the
    integrality gap (zero implies zero true gap; positive may be descent
    slack — refine with a better incumbent if it matters)."""
    lp, _, _, st = solve_full(C, p, relax=True, time_limit=time_limit)
    if st != 0:
        return None
    _, ub, _ = swap_descent(np.asarray(C, dtype=float), p, seed=seed)
    return max(ub - lp, 0.0) / max(ub, 1e-12)


def route(C, p):
    """Return the engine name for (C, p): 'expand' or 'benders'."""
    m = np.asarray(C).shape[0]
    return "expand" if p / m >= 0.10 else "benders"


def solve(C, p, time_limit=600.0, engine=None, backend=None):
    """Routed exact solve. Returns (value, S, seconds, status, engine)."""
    eng = engine or route(C, p)
    if eng == "expand":
        v, S, t, st, _ = solve_expand(C, p, time_limit=time_limit)
    elif eng == "benders":
        v, S, t, st = BendersPMedian(C, p, backend=backend).solve(
            time_limit=time_limit)
    else:
        v, S, t, st = solve_full(C, p, time_limit=time_limit)
    return v, S, t, st, eng
