"""Expand-until-certificate solving.

The loop solves the reduced model, deepens the lists of clients whose
overflow variable is active at the optimum, and terminates when no
(non-legal) overflow is active. At that point every client pays a LISTED
(true) cost, and since the capped model underestimates every set, the
returned solution is optimal for the TRUE problem — an a-priori optimality
certificate that needs no reference value. This is the mechanics of
Church's BEAMR / ZEBRA / Goldengorin's CFIE ch.2 truncation lineage,
packaged with the certificate made explicit.
"""

from math import ceil

import numpy as np

from .model import reduced_pmedian_ragged


def solve_expand(C, p, exclude=None, force_open=None, time_limit=300.0,
                 legal_cap=None, depth_overrides=None, max_rounds=12):
    """Returns (value, S, seconds, status, detail); status 0 implies the
    no-overflow optimality certificate held (detail["certified"]).

    legal_cap: client index whose overflow never triggers expansion and does
    not block the certificate — SOUND ONLY IF the caller pins that column's
    depth (depth_overrides) so its overflow price equals an externally
    justified exact price (see tolerance.entry_tolerance).
    """
    C = np.asarray(C, dtype=float)
    m, n = C.shape
    depths = np.full(n, min(max(2, ceil(0.1 * m)), m), dtype=int)
    if depth_overrides:
        for j, dj in depth_overrides.items():
            depths[j] = min(max(int(dj), 1), m)
    t_total, rounds, prev_S = 0.0, [], None
    for _ in range(max_rounds):
        remaining = max(time_limit - t_total, 5.0)
        val, S, dt, st, info = reduced_pmedian_ragged(
            C, depths, p, exclude=exclude, force_open=force_open,
            time_limit=remaining, start_S=prev_S)
        t_total += dt + info["t_build"]
        rounds.append({"depths_max": int(depths.max()),
                       "n_xvars": info["n_xvars"], "t": dt})
        if st != 0:
            return None, None, t_total, st, {"rounds": rounds,
                                             "depths": depths}
        prev_S = S
        over = info["overflow"].copy()
        if legal_cap is not None:
            over[legal_cap] = False
        if not over.any():
            return val, S, t_total, 0, {"rounds": rounds, "depths": depths,
                                        "certified": True}
        depths = np.where(over, np.minimum(depths * 2, m), depths)
        if t_total >= time_limit:
            return None, None, t_total, 1, {"rounds": rounds,
                                            "depths": depths}
    return None, None, t_total, 1, {"rounds": rounds, "depths": depths}
