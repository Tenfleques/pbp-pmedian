"""The unified pipeline: one call (or one shell command) from a cost matrix
to a routed, certified solve with a tolerance report.

    report = analyze(C, p)          # gap -> route -> solve -> tolerances
    print(format_report(report))

Stages (each recorded in the report with its wall time):

1. one-LP gap estimate — (UB - LP)/UB with UB from swap descent; zero
   certifies the LP-tight regime (cheap tolerance certificates fire there),
   positive flags LP-loose (they will not; exact solves still work). The
   relaxation is solved once WITH duals (model.lp_prices); its reduced
   costs feed the free dual-price screen of stage 3b at no extra solve.
2. engine routing — rho = p/m >= 0.10 -> expand-until-certificate, else
   Benders; explicit ``engine=`` overrides. Out-of-sample (23 fresh
   instances, 2026-08-30) the rule scored 0.70 oracle agreement at 1.20x
   the oracle's total wall-clock — a good default, not a law; misses
   concentrate where engines tie. A failed routed solve falls back to the
   full model once (recorded).
3. base solve — with the engine's own optimality certificate: expand's
   no-overflow expansion is a-priori exact; Benders terminates only when
   the master value equals the true cost of its own set (the master being
   a relaxation, that proves optimality); the full model relies on the MIP
   solver's gap-zero proof.
4. tolerance layer — ``tolerances="screen"`` (default): the free certified
   lower bound F2 = min(u*, margin_2) per serving entry from ONE extra
   solve, which either bounds the tolerance from below or certifies it
   infinite; ``"exact"``: the branch-split profile (facility solves cached,
   one level-restricted entry solve per client — exact on every entry);
   ``"none"``: skip.
"""

import time

import numpy as np

from .core import EPS, serving_split
from .model import lp_prices, solve_full
from .router import gap_estimate, route, solve
from .tolerance import (dual_price_screen, f2_screen, margin2,
                        tolerance_profile)


def analyze(C, p, tolerances="screen", engine=None, clients=None,
            time_limit=600.0, seed=0):
    """Run the full pipeline on cost matrix C (m facilities x n clients).

    Returns a dict report; see format_report for the human-readable view.
    tolerances: "none" | "screen" | "exact".
    engine: None (routed) | "expand" | "benders" | "full".
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a 2-D cost matrix")
    m, n = C.shape
    if not 1 <= p < m:
        raise ValueError(f"need 1 <= p < m, got p={p}, m={m}")
    if tolerances not in ("none", "screen", "exact"):
        raise ValueError(f"unknown tolerances mode {tolerances!r}")
    report = {"m": m, "n": n, "p": p, "rho": p / m}

    t0 = time.perf_counter()
    lp = lp_prices(C, p, time_limit=time_limit)
    gap = (None if lp["status"] != 0 else
           gap_estimate(C, p, time_limit=time_limit, seed=seed,
                        lp_val=lp["value"]))
    report["gap_estimate"] = gap
    report["lp_tight"] = (gap is not None and gap <= 1e-9)
    report["t_gap_s"] = time.perf_counter() - t0

    eng = engine or route(C, p)
    report["engine_requested"] = engine or f"auto->{eng}"
    v, S, t_base, st, eng_used = solve(C, p, time_limit=time_limit,
                                       engine=eng)
    report["fallback_used"] = False
    if st != 0 and eng != "full":
        v, S, t_fb, st = solve_full(C, p, time_limit=time_limit)
        t_base += t_fb
        eng_used = "full"
        report["fallback_used"] = True
    report.update({"engine": eng_used, "t_base_s": t_base,
                   "base_status": st})
    if st != 0:
        report.update({"f_star": None, "S_star": None,
                       "certificate": "NONE — no engine proved optimality "
                                      f"within {time_limit:.0f}s"})
        return report
    report["f_star"] = float(v)
    report["S_star"] = tuple(int(i) for i in S)
    report["certificate"] = {
        "expand": "no-overflow expansion (a-priori exact)",
        "benders": "Benders master value == true cost (relaxation tight)",
        "full": "MIP solver optimality at zero gap",
    }[eng_used]

    # 3b. dual-price screen — free: reads the reduced costs of the LP the
    # gap stage already solved (theorem: T_i >= rc_i - gap; certified when
    # the gap is zero). t_prices_s is the marginal cost AFTER that LP.
    if lp["status"] == 0:
        t0 = time.perf_counter()
        report["dual_prices"] = dual_price_screen(C, p, S, v, lp=lp)
        report["t_prices_s"] = time.perf_counter() - t0

    report["tolerances"] = tolerances
    if tolerances == "none":
        return report
    t0 = time.perf_counter()
    if tolerances == "screen":
        m2, _ = margin2(C, p, S, v, time_limit=time_limit)
        if m2 is None:
            report["tolerance_status"] = ("margin_2 solve failed within "
                                          "the time limit")
            report["t_tol_s"] = time.perf_counter() - t0
            return report
        cols = clients if clients is not None else range(n)
        cj, fac, us, f2, deg = [], [], [], [], []
        for j0 in cols:
            i0, _, u_star = serving_split(C, S, j0)
            cj.append(int(j0))
            fac.append(int(i0))
            us.append(u_star)
            f2.append(f2_screen(u_star, m2))
            deg.append(bool(u_star <= EPS))
        report.update({"margin_2": m2, "client": np.array(cj),
                       "facility": np.array(fac), "u_star": np.array(us),
                       "f2_lower": np.array(f2),
                       "degenerate": np.array(deg)})
        sat = np.isinf(report["f2_lower"]) & ~report["degenerate"]
        report["screen_summary"] = {
            "clients": len(cj),
            "certified_saturated": int(sat.sum()),
            "degenerate": int(report["degenerate"].sum()),
            "finite_lower_bounds": int(np.isfinite(report["f2_lower"]).sum()),
        }
    else:  # exact
        prof = tolerance_profile(
            C, p, S_star=S, f_star=v, clients=clients,
            time_limit=time_limit,
            a_engine="full" if not report["lp_tight"] else "expand")
        report.update({k: prof[k] for k in
                       ("client", "facility", "u_star", "d_min", "tol")})
        report["degenerate"] = np.array(
            [u <= EPS for u in report["u_star"]])
        tol = report["tol"]
        finite = np.isfinite(tol)
        report["profile_summary"] = {
            "entries": len(tol),
            "finite": int(finite.sum()),
            "saturated": int((~finite & ~report["degenerate"]).sum()),
            "degenerate": int(report["degenerate"].sum()),
            "tol_min": float(tol[finite].min()) if finite.any() else None,
            "tol_median": (float(np.median(tol[finite]))
                           if finite.any() else None),
        }
    report["t_tol_s"] = time.perf_counter() - t0
    return report


def format_report(report):
    """Human-readable rendering of an analyze() report."""
    ln = [f"p-median analysis — {report['m']} facilities x "
          f"{report['n']} clients, p={report['p']} "
          f"(rho={report['rho']:.3f})"]
    g = report.get("gap_estimate")
    if g is None:
        ln.append("gap estimate: LP failed within the time limit")
    else:
        regime = ("LP-tight — cheap certificates apply" if report["lp_tight"]
                  else "positive — LP-loose suspect, expect exact solves "
                       "to carry the tolerance layer")
        ln.append(f"one-LP gap estimate: {g:.5f} ({regime}) "
                  f"[{report['t_gap_s']:.1f}s]")
    ln.append(f"engine: {report['engine']} "
              f"(requested {report['engine_requested']}"
              f"{', FELL BACK to full' if report['fallback_used'] else ''}) "
              f"— router calibration is post-hoc; override with engine=")
    if report.get("f_star") is None:
        ln.append(f"base solve FAILED: {report['certificate']}")
        return "\n".join(ln)
    S = report["S_star"]
    s_txt = (", ".join(map(str, S[:8])) + (", ..." if len(S) > 8 else ""))
    ln.append(f"optimum: f* = {report['f_star']:g}, S* = ({s_txt}) — "
              f"{report['certificate']} [{report['t_base_s']:.1f}s]")
    dp = report.get("dual_prices")
    if dp and dp.get("status") == 0:
        kind = ("certified lower bounds (gap 0)" if dp["certified"] else
                f"gap-corrected lower bounds (gap {dp['gap_abs']:g}; "
                "naive reduced costs NOT safe here)")
        ln.append(f"dual-price screen (free, from the gap LP): "
                  f"{dp['n_closed']} closed facilities priced — "
                  f"{dp['informative']} informative (rc>0), {kind} "
                  f"[{report.get('t_prices_s', 0.0):.3f}s]")
    if report.get("tolerances") == "screen" and "screen_summary" in report:
        s = report["screen_summary"]
        ln.append(f"tolerance screen (free F2 bound): {s['clients']} clients"
                  f" — {s['finite_lower_bounds']} finite lower bounds, "
                  f"{s['certified_saturated']} certified saturated, "
                  f"{s['degenerate']} degenerate (tied optimum) "
                  f"[{report['t_tol_s']:.1f}s]")
    elif report.get("tolerances") == "exact" and "profile_summary" in report:
        s = report["profile_summary"]
        rng_txt = (f" [min {s['tol_min']:g}, median {s['tol_median']:g}]"
                   if s["finite"] else "")
        ln.append(f"tolerance profile (exact): {s['entries']} serving "
                  f"entries — {s['finite']} finite{rng_txt}, "
                  f"{s['saturated']} saturated, {s['degenerate']} "
                  f"degenerate [{report['t_tol_s']:.1f}s]")
    elif report.get("tolerance_status"):
        ln.append(f"tolerance layer: {report['tolerance_status']}")
    return "\n".join(ln)
