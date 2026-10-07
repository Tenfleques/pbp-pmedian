"""Pipeline + CLI tests, enumeration-backed like the rest of the suite."""

import itertools
import json

import numpy as np
import pytest

from pbp_pmedian import analyze, cost_of, format_report
from pbp_pmedian.cli import main as cli_main


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


def test_analyze_exact_equals_enumeration(instance):
    C, p = instance
    _, costs = enum_all(C, p)
    rep = analyze(C, p, tolerances="exact")
    assert rep["base_status"] == 0
    assert abs(rep["f_star"] - costs.min()) <= 1e-9
    assert abs(cost_of(C, rep["S_star"]) - rep["f_star"]) <= 1e-9
    for j0, i0, tol, d_min, deg in zip(rep["client"], rep["facility"],
                                       rep["tol"], rep["d_min"],
                                       rep["degenerate"]):
        if deg:
            assert np.isinf(tol)
            continue
        tol_ref, _, d_ref = enum_tolerance(C, p, i0, j0, rep["S_star"])
        assert (np.isinf(tol) and np.isinf(tol_ref)) or \
            abs(tol - tol_ref) <= 1e-6, (j0, tol, tol_ref)
        if np.isfinite(tol_ref):
            assert abs(d_min - d_ref) <= 1e-6


def test_certified_mode_does_not_report_an_interval_as_a_number(instance):
    C, p = instance
    rep = analyze(C, p, tolerances="certified", engine="full")
    summary = rep["certified_summary"]
    assert (summary["certified"] + summary["interval"] + summary["degenerate"]
            + summary["unsolved"] == summary["entries"])
    assert "certified or interval" in format_report(rep)
    for status, tol in zip(rep["status"], rep["tol"]):
        if status == "interval":
            assert np.isnan(tol)
        elif status == "certified":
            assert np.isfinite(tol) or np.isinf(tol)


def test_screen_is_safe_lower_bound(instance):
    C, p = instance
    rep_s = analyze(C, p, tolerances="screen")
    rep_e = analyze(C, p, tolerances="exact")
    exact = {int(j): t for j, t in zip(rep_e["client"], rep_e["tol"])}
    for j0, f2, deg in zip(rep_s["client"], rep_s["f2_lower"],
                           rep_s["degenerate"]):
        if deg or int(j0) not in exact:
            continue
        tol = exact[int(j0)]
        if np.isinf(f2):
            assert np.isinf(tol), (j0, f2, tol)  # certified saturation
        else:
            assert f2 <= tol + 1e-9, (j0, f2, tol)  # lower bound


def test_engine_override_and_none_mode(instance):
    C, p = instance
    vals = set()
    for eng in ("expand", "benders", "full"):
        rep = analyze(C, p, tolerances="none", engine=eng)
        assert rep["engine"] == eng and rep["base_status"] == 0
        vals.add(round(rep["f_star"], 9))
        assert "tol" not in rep
    assert len(vals) == 1


def test_format_report_renders(instance):
    C, p = instance
    txt = format_report(analyze(C, p, tolerances="screen"))
    assert "optimum: f* =" in txt and "tolerance screen" in txt


def test_analyze_input_validation():
    with pytest.raises(ValueError):
        analyze(np.ones(5), 2)
    with pytest.raises(ValueError):
        analyze(np.ones((4, 4)), 4)
    with pytest.raises(ValueError):
        analyze(np.ones((4, 4)), 2, tolerances="bogus")


def test_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(7)
    C = rng.integers(1, 30, size=(8, 10)).astype(float)
    mat = tmp_path / "costs.csv"
    np.savetxt(mat, C, delimiter=",")
    out_json = tmp_path / "report.json"
    rc = cli_main([str(mat), "-p", "3", "--tolerances", "exact",
                   "--json", str(out_json)])
    assert rc == 0
    assert "optimum: f* =" in capsys.readouterr().out
    rep = json.loads(out_json.read_text())
    _, costs = enum_all(C, 3)
    assert abs(rep["f_star"] - costs.min()) <= 1e-9
    assert rep["m"] == 8 and rep["p"] == 3
    # whitespace-separated load path + engine override + client subset
    mat2 = tmp_path / "costs.txt"
    np.savetxt(mat2, C)
    rc = cli_main([str(mat2), "-p", "3", "--engine", "benders",
                   "--tolerances", "screen", "--clients", "0,3,7"])
    assert rc == 0


def test_analyze_dual_prices_block():
    """analyze() carries the free dual-price screen: consistent with a
    manual lp_prices call, zero extra LP solves (t_prices_s is marginal)."""
    import numpy as np
    from pbp_pmedian import analyze, dual_price_screen, format_report, lp_prices

    rng = np.random.default_rng(7)
    C = rng.uniform(1, 50, size=(12, 15))
    p = 4
    rep = analyze(C, p, tolerances="none")
    assert rep["base_status"] == 0
    dp = rep["dual_prices"]
    assert dp["status"] == 0 and dp["n_closed"] == 12 - p
    manual = dual_price_screen(C, p, rep["S_star"], rep["f_star"],
                               lp=lp_prices(C, p))
    assert np.allclose(dp["rc"], manual["rc"], atol=1e-9)
    assert np.allclose(dp["lower"], manual["lower"], atol=1e-9)
    assert dp["certified"] == manual["certified"]
    assert rep["t_prices_s"] < 0.1, "screen must be marginal, not a solve"
    assert dp["certified"] == rep["lp_tight"] or not dp["certified"]
    assert "dual-price screen" in format_report(rep)
