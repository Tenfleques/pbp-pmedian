"""The engine router.

Empirical rule, out-of-sample tested 2026-08-30 on 23 fresh instances
(never used in calibration; oracle = fastest engine that proves
optimality):

    rho = p / m >= 0.10        -> expand-until-certificate
    rho <  0.10                -> Benders

This plain rho-rule scored 0.70 oracle agreement at 1.20x the oracle's
total wall-clock out-of-sample. The richer fitted variant that also sent
LP-loose random instances to the full model was DEMOTED by the same test
(0.65 / 1.32x — its random-class leg, calibrated on one instance, went
1-for-3 on fresh random instances). Treat route() as a good default, not a
law: disagreements concentrate where the engines tie (sub-second), plus a
residual rho-boundary band around 0.10-0.20 where either engine is fine.

The one-LP integrality gap still predicts the TOLERANCE-bound regime (zero
gap: cheap LP bounds certify; positive gap: they fail) — that is a separate
dispatch from the engine choice above.
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
