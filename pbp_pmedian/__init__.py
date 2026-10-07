"""pbp-pmedian — exact p-median solving and per-entry cost tolerances via
the Hammer-Beresnev (pseudo-Boolean) representation.

Quickstart::

    import numpy as np
    from pbp_pmedian import solve, tolerance_profile

    C = np.random.default_rng(0).uniform(1, 50, size=(30, 40))
    value, S, secs, status, engine = solve(C, p=5)
    prof = tolerance_profile(C, 5, S_star=S, f_star=value)

The C kernel (``bash build_pmedian.sh``) accelerates the solver-free
primitives; ``pbp_pmedian.core_c.HAVE_C`` reports whether it is active.
"""

from .backends import BACKENDS, DEFAULT as DEFAULT_BACKEND, HAVE_HIGHSPY
from .benders import BendersPMedian
from .core import (cost_of, labeled_distance, reuse_certificate,
                   serving_split, swap_T1, swap_descent, truncated_matrix)
from .hb import hb_polynomial, hb_vector
from .model import (lp_prices, reduced_pmedian_ragged, set_backend,
                    solve_full)
from .pipeline import analyze, format_report
from .router import gap_estimate, route, solve
from .solve import solve_expand
from .tolerance import (b2_profile_warm, certified_profile, dual_price_screen,
                        entry_label, entry_tolerance, f2_screen,
                        facility_tolerance, margin2, tolerance_profile)

__version__ = "0.2.4"

__all__ = [
    "BACKENDS", "DEFAULT_BACKEND", "HAVE_HIGHSPY", "BendersPMedian",
    "analyze", "format_report",
    "cost_of", "labeled_distance", "reuse_certificate", "serving_split",
    "swap_T1", "swap_descent", "truncated_matrix", "hb_polynomial",
    "hb_vector", "reduced_pmedian_ragged", "set_backend", "solve_full",
    "gap_estimate", "route", "solve", "solve_expand", "lp_prices",
    "dual_price_screen", "b2_profile_warm", "certified_profile", "entry_label",
    "entry_tolerance", "f2_screen", "facility_tolerance", "margin2",
    "tolerance_profile",
    "__version__",
]
