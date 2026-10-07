"""Command-line entry point: the whole pipeline as one shell command.

    pbp-pmedian costs.csv -p 5
    pbp-pmedian costs.csv -p 5 --tolerances exact --json report.json
    pbp-pmedian costs.txt -p 3 --engine benders --clients 0,4,17

The matrix file is plain numeric text (comma- or whitespace-separated,
rows = facilities, columns = clients, no header). The engine router is a
post-hoc calibration (see pipeline docs) — ``--engine`` overrides it.
"""

import argparse
import json
import sys

import numpy as np

from . import __version__
from .pipeline import analyze, format_report


def _load_matrix(path):
    try:
        return np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        return np.loadtxt(path, ndmin=2)


def _jsonable(obj):
    """Report dict -> JSON-safe (arrays to lists, non-finite to strings)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if np.isnan(f):
            return "nan"
        if np.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pbp-pmedian",
        description="Exact p-median solve + certified cost tolerances via "
                    "the Hammer-Beresnev representation.")
    ap.add_argument("matrix", help="cost matrix file (rows = facilities, "
                                   "columns = clients; CSV or whitespace)")
    ap.add_argument("-p", type=int, required=True,
                    help="number of medians to open")
    ap.add_argument("--tolerances",
                    choices=("none", "screen", "certified", "exact"),
                    default="screen",
                    help="tolerance layer: free F2 screen (default), "
                         "certified-or-interval profile, exact branch-split "
                         "profile, or none")
    ap.add_argument("--engine", choices=("auto", "expand", "benders", "full"),
                    default="auto", help="base engine (default: routed)")
    ap.add_argument("--clients", default=None,
                    help="comma-separated client indices to profile "
                         "(default: all)")
    ap.add_argument("--time-limit", type=float, default=600.0,
                    help="seconds per solve stage (default 600)")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="also write the full report as JSON to this path")
    ap.add_argument("--version", action="version",
                    version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    C = _load_matrix(args.matrix)
    clients = (None if args.clients is None else
               [int(s) for s in args.clients.split(",") if s.strip()])
    report = analyze(C, args.p, tolerances=args.tolerances,
                     engine=None if args.engine == "auto" else args.engine,
                     clients=clients, time_limit=args.time_limit)
    print(format_report(report))
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(_jsonable(report), fh, indent=2)
        print(f"report written to {args.json_path}")
    return 0 if report.get("f_star") is not None else 1


if __name__ == "__main__":
    sys.exit(main())
