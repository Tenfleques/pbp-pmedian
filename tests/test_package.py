"""pbp-pmedian test suite.

Everything is validated the way the research pipeline validated it:
against FULL ENUMERATION on small instances (the ground truth no solver
can argue with), plus cross-backend and C-vs-Python equality. These are
ports of the selftests that guarded the benchmark campaign (kit 106/106,
Benders 103/103).
"""

import itertools

import numpy as np
import pytest

from pbp_pmedian import (BendersPMedian, cost_of, entry_tolerance,
                         facility_tolerance, labeled_distance, margin2,
                         reuse_certificate, route, solve, solve_expand,
                         solve_full, swap_T1, swap_descent,
                         tolerance_profile, truncated_matrix)
from pbp_pmedian import backends, core, core_c


def enum_all(C, p):
    subsets = list(itertools.combinations(range(C.shape[0]), p))
    costs = np.array([cost_of(C, S) for S in subsets])
    return subsets, costs


def enum_tolerance(C, p, i0, j0, S_star):
    subsets, costs = enum_all(C, p)
    k_star = subsets.index(tuple(S_star))
    a = C[i0, j0]
    others = [i for i in S_star if i != i0]
    u_star = C[others, j0].min() - a
    d_min = np.inf
    for k, S in enumerate(subsets):
        if k == k_star:
            continue
        margin = costs[k] - costs[k_star]
        if i0 in S:
            rest = [i for i in S if i != i0]
            u_S = max(C[rest, j0].min() - a, 0.0)
        else:
            u_S = 0.0
        d_min = min(d_min, margin + u_S)
    tol = d_min if d_min < u_star - 1e-9 else np.inf
    return tol, u_star, d_min


@pytest.fixture(params=["int", "cont"])
def instance(request):
    rng = np.random.default_rng(1330 if request.param == "int" else 1331)
    C = (rng.integers(1, 21, size=(9, 12)).astype(float)
         if request.param == "int" else rng.uniform(1, 50, size=(9, 12)))
    return C, 3


def test_expand_equals_enumeration(instance):
    C, p = instance
    subsets, costs = enum_all(C, p)
    f_star = costs.min()
    v, S, _, st, d = solve_expand(C, p, time_limit=60.0)
    assert st == 0 and d.get("certified")
    assert abs(v - f_star) <= 1e-6
    assert abs(cost_of(C, S) - v) <= 1e-9


def test_full_and_exclusion(instance):
    C, p = instance
    subsets, costs = enum_all(C, p)
    v, S, _, st = solve_full(C, p, time_limit=60.0)
    assert st == 0 and abs(v - costs.min()) <= 1e-6
    v2, _, _, st = solve_full(C, p, exclude=S, time_limit=60.0)
    second = min(c for sub, c in zip(subsets, costs) if tuple(sub) != S)
    assert st == 0 and abs(v2 - second) <= 1e-6


def test_benders_equals_enumeration(instance):
    C, p = instance
    _, costs = enum_all(C, p)
    bd = BendersPMedian(C, p)
    v, S, _, st = bd.solve(time_limit=120.0)
    assert st == 0 and abs(v - costs.min()) <= 1e-6
    y = np.zeros(C.shape[0])
    y[list(S)] = 1.0
    th = bd.true_thetas(S)
    for (j, rhs, sites, coefs) in bd.cuts:
        lhs = th[j] + (coefs @ y[sites] if len(sites) else 0.0)
        assert lhs >= rhs - 1e-6, "cut invalid at the optimum"


def test_entry_tolerance_equals_enumeration(instance):
    C, p = instance
    v, S_star, _, st = solve_full(C, p, time_limit=60.0)
    assert st == 0
    for j0 in range(C.shape[1]):
        i0 = min(S_star, key=lambda i: (C[i, j0], i))
        others = [i for i in S_star if i != i0]
        if C[others, j0].min() - C[i0, j0] <= 1e-6:
            continue
        tol_e, u_e, dmin_e = enum_tolerance(C, p, i0, j0, S_star)
        A, _ = facility_tolerance(C, p, i0, v, time_limit=60.0)
        tol_k, dmin_k, _, _, st = entry_tolerance(
            C, p, S_star, v, i0, j0, A, time_limit=60.0)
        assert st == 0
        assert (np.isinf(tol_k) and np.isinf(tol_e)) or \
            abs(tol_k - tol_e) <= 1e-6
        if np.isfinite(tol_e):
            assert abs(dmin_k - dmin_e) <= 1e-6
        else:   # saturated: certified lower bound in [u*, d_min]
            assert u_e - 1e-6 <= dmin_k <= dmin_e + 1e-6


def test_entry_label_keeps_the_facility_bound_in_the_interval():
    from pbp_pmedian import entry_label
    status, tol, lo, hi = entry_label(10.0, 5.0, 2.0, False)
    assert status == "interval" and np.isnan(tol)
    assert lo == 2.0 and hi == 5.0
    status, tol, lo, hi = entry_label(10.0, 5.0, 5.0, False)
    assert status == "certified" and abs(tol - 5.0) <= 1e-9
    status, tol, lo, hi = entry_label(10.0, 8.0, 3.0, True)
    assert status == "certified" and abs(tol - 3.0) <= 1e-9
    status, tol, lo, hi = entry_label(4.0, 9.0, 4.0, False)
    assert status == "certified" and np.isinf(tol)
    status, tol, _, _ = entry_label(0.0, 1.0, 1.0, False)
    assert status == "degenerate" and np.isinf(tol)


def test_certified_profile_matches_enumeration(instance):
    from pbp_pmedian import certified_profile
    C, p = instance
    v, S, _, st = solve_full(C, p, time_limit=60.0)
    assert st == 0
    prof = certified_profile(C, p, S_star=S, f_star=v, time_limit=60.0,
                             a_engine="full")
    assert len(prof["client"]) == C.shape[1]
    for j0, i0, status, tol, lo, hi in zip(
            prof["client"], prof["facility"], prof["status"],
            prof["tol"], prof["lo"], prof["hi"]):
        _tol_e, u_e, d_e = enum_tolerance(C, p, int(i0), int(j0), S)
        if u_e <= 1e-6:
            assert status == "degenerate" and np.isinf(tol)
            continue
        assert status in ("certified", "interval")
        if status == "certified":
            if np.isinf(tol):
                assert d_e >= u_e - 1e-5
            else:
                assert abs(float(tol) - d_e) <= 1e-5
                assert abs(float(tol) - hi) <= 1e-5 or abs(float(tol) - lo) <= 1e-5
        else:
            assert np.isnan(tol)
            assert float(lo) - 1e-5 <= d_e <= float(hi) + 1e-5


def test_profile_runs(instance):
    C, p = instance
    prof = tolerance_profile(C, p, time_limit=60.0)
    n = C.shape[1]
    assert len(prof["client"]) == n
    fin = np.isfinite(prof["tol"])
    assert (prof["tol"][fin] >= 0).all()


def test_router_and_solve(instance):
    C, p = instance
    assert route(C, p) in ("expand", "benders")
    v, S, _, st, eng = solve(C, p, time_limit=120.0)
    _, costs = enum_all(C, p)
    assert st == 0 and abs(v - costs.min()) <= 1e-6


def test_screens_and_caching(instance):
    C, p = instance
    v, S_star, _, st = solve_full(C, p, time_limit=60.0)
    m2, _ = margin2(C, p, S_star, v, time_limit=60.0)
    assert m2 is not None and m2 >= -1e-9
    rng = np.random.default_rng(7)
    subsets, _ = enum_all(C, p)
    for scale in (0.01, 0.2):
        Cp = C + rng.uniform(-scale, scale, size=C.shape)
        d = labeled_distance(C, Cp)
        if reuse_certificate(d, m2):
            best = min(cost_of(Cp, S) for S in subsets)
            assert cost_of(Cp, S_star) <= best + 1e-9


def test_truncated_matrix_monotone(instance):
    C, p = instance
    prev = None
    for D in (2, 4, 8):
        TD = truncated_matrix(C, D)
        assert (TD <= C + 1e-12).all()
        if prev is not None:
            assert (TD >= prev - 1e-12).all()
        prev = TD


def test_c_kernel_matches_python(instance):
    C, p = instance
    if not core_c.HAVE_C:
        pytest.skip("C kernel not built (bash build_pmedian.sh)")
    rng = np.random.default_rng(4)
    for _ in range(5):
        S = sorted(rng.choice(C.shape[0], size=p, replace=False).tolist())
        assert abs(core_c.cost_of(C, S) - core.cost_of(C, S)) <= 1e-9
        f = core.cost_of(C, S)
        t1_py, _ = core.swap_T1(C, S, f)
        t1_c, _ = core_c.swap_T1(C, S, f)
        assert abs(t1_py - t1_c) <= 1e-9
        Spy, fpy, _ = core.swap_descent(C, p, S0=S)
        Sc, fc, _ = core_c.swap_descent(C, p, S0=S)
        assert abs(fc - core.cost_of(C, Sc)) <= 1e-9
        assert fc <= f + 1e-9
        assert abs(fpy - fc) <= 1e-6


@pytest.mark.skipif(not backends.HAVE_HIGHSPY, reason="highspy not installed")
def test_highs_backend_agrees(instance):
    from pbp_pmedian import set_backend
    C, p = instance
    v_s, _, _, st = solve_full(C, p, time_limit=60.0)
    set_backend("highs")
    try:
        v_h, _, _, st2 = solve_full(C, p, time_limit=60.0)
        vE, _, _, st3, _ = solve_expand(C, p, time_limit=60.0)
    finally:
        set_backend("scipy")
    assert st == st2 == st3 == 0
    assert abs(v_s - v_h) <= 1e-6 and abs(v_s - vE) <= 1e-6


@pytest.mark.skipif(not backends.HAVE_HIGHSPY, reason="needs highspy")
def test_benders_harvest_equals_enumeration(instance):
    C, p = instance
    subsets, costs = enum_all(C, p)
    b = BendersPMedian(C, p, backend="highs")     # harvest defaults on
    assert b.harvest
    f, S, _, st = b.solve(time_limit=120.0)
    assert st == 0 and abs(f - costs.min()) <= 1e-6
    # every cut (incumbent-harvested included) valid at the true optimum
    S_ref = subsets[int(np.argmin(costs))]
    y = np.zeros(C.shape[0])
    y[list(S_ref)] = 1.0
    th = C[list(S_ref), :].min(axis=0)
    for (j, rhs, sites, coefs) in b.cuts:
        lhs = th[j] + (coefs @ y[sites] if len(sites) else 0.0)
        assert lhs >= rhs - 1e-6
    # de-dup: keys unique across the pool
    keys = [(j, round(float(rhs), 9), sites.tobytes())
            for (j, rhs, sites, coefs) in b.cuts]
    assert len(keys) == len(set(keys))
    # explicit off-switch reproduces the callback-free result
    b2 = BendersPMedian(C, p, backend="highs", harvest=False)
    assert not b2.harvest
    f2, _, _, st2 = b2.solve(time_limit=120.0)
    assert st2 == 0 and abs(f2 - f) <= 1e-6


def test_lp_prices_matches_relaxation(instance):
    """lp_prices solves the same relaxation as solve_full(relax=True) and
    returns consistent dual information (basic facilities price at 0)."""
    from pbp_pmedian import lp_prices

    C, p = instance
    lp = lp_prices(C, p)
    assert lp["status"] == 0
    val_ref, y_ref, _, st = solve_full(C, p, relax=True)
    assert st == 0
    assert abs(lp["value"] - val_ref) <= 1e-6
    assert lp["rc_y"].shape == (C.shape[0],)
    assert (lp["rc_y"] >= 0).all()
    basic = lp["y"] > 1e-9
    assert (lp["rc_y"][basic] <= 1e-7).all(), "basic facility priced"


def test_dual_price_screen_theorem(instance):
    """T_i >= rc_i - gap for every closed facility, with T_i from the
    forced solve and, independently, from full enumeration."""
    from pbp_pmedian import dual_price_screen

    C, p = instance
    v, S, _, st = solve_full(C, p)
    assert st == 0
    dp = dual_price_screen(C, p, S, v)
    assert dp["status"] == 0
    assert dp["n_closed"] == C.shape[0] - p
    subsets, costs = enum_all(C, p)
    f_star = costs.min()
    for i, lower in zip(dp["facility"], dp["lower"]):
        vf, _, _, stf = solve_full(C, p, force_open=[int(i)])
        assert stf == 0
        T = max(vf - v, 0.0)
        assert T >= lower - 1e-6, f"theorem violated at facility {i}"
        T_enum = min(c for S_, c in zip(subsets, costs) if i in S_) - f_star
        assert abs(T - T_enum) <= 1e-6, "forced solve != enumeration"
        if dp["certified"]:
            assert dp["rc"][list(dp["facility"]).index(i)] <= T + 1e-6


def test_dual_price_screen_open_silence(instance):
    """Open facilities are basic: rc identically zero (never priced)."""
    from pbp_pmedian import dual_price_screen, lp_prices

    C, p = instance
    v, S, _, st = solve_full(C, p)
    assert st == 0
    lp = lp_prices(C, p)
    dp = dual_price_screen(C, p, S, v, lp=lp)
    assert set(dp["facility"]) == set(range(C.shape[0])) - set(S)
    if dp["certified"]:
        assert (lp["rc_y"][list(S)] <= 1e-7).all()


def test_b2_profile_warm(instance):
    """Warm-started B2 profile: warm == cold rebuild == safe vs enumeration,
    and the integrality certificate is never wrong."""
    from pbp_pmedian import b2_profile_warm

    C, p = instance
    v, S, _, st = solve_full(C, p)
    assert st == 0
    warm = b2_profile_warm(C, p, S, v)
    cold = b2_profile_warm(C, p, S, v, warm=False)
    assert np.allclose(warm["B2"].astype(float), cold["B2"].astype(float),
                       atol=1e-6), "warm != cold"
    for k in range(len(warm["client"])):
        j0, i0 = int(warm["client"][k]), int(warm["facility"][k])
        if warm["degenerate"][k]:
            continue
        _, _, d_min = enum_tolerance(C, p, i0, j0, S)
        b2 = float(warm["B2"][k])
        assert b2 <= d_min + 1e-6, f"B2 overestimates at client {j0}"
        if warm["cert"][k]:
            assert abs(b2 - d_min) <= 1e-6, f"certificate wrong at {j0}"
