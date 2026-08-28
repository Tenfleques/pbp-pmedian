"""ctypes wrapper for the C kernel, with automatic fallback to core.py.

Build the shared library with ``bash build_pmedian.sh`` (repo root); the
wheel ships it when present. ``HAVE_C`` reports which path is active; the
API mirrors core.py exactly so call sites can import either module.
"""

import ctypes
import time
from pathlib import Path

import numpy as np

from . import core as _py

_LIB = None
for _name in ("libpmedian_core.dylib", "libpmedian_core.so"):
    _p = Path(__file__).parent / _name
    if _p.exists():
        try:
            _LIB = ctypes.CDLL(str(_p))
        except OSError:
            _LIB = None
        break

HAVE_C = _LIB is not None

if HAVE_C:
    _dp = ctypes.POINTER(ctypes.c_double)
    _ip = ctypes.POINTER(ctypes.c_int)
    _LIB.pm_cost.restype = ctypes.c_double
    _LIB.pm_cost.argtypes = [_dp, ctypes.c_int, ctypes.c_int, _ip,
                             ctypes.c_int]
    _LIB.pm_swap_t1.restype = ctypes.c_double
    _LIB.pm_swap_t1.argtypes = [_dp, ctypes.c_int, ctypes.c_int, _ip,
                                ctypes.c_int, ctypes.c_double]
    _LIB.pm_swap_descent.restype = ctypes.c_double
    _LIB.pm_swap_descent.argtypes = [_dp, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, _ip, ctypes.c_int, _ip]


def _carr(C):
    a = np.ascontiguousarray(C, dtype=np.float64)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def cost_of(C, S):
    if not HAVE_C:
        return _py.cost_of(C, S)
    a, cp = _carr(C)
    m, n = a.shape
    s = np.ascontiguousarray(sorted(S), dtype=np.int32)
    return float(_LIB.pm_cost(cp, m, n,
                              s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              len(s)))


def swap_T1(C2, S_star, f_star):
    if not HAVE_C:
        return _py.swap_T1(C2, S_star, f_star)
    t0 = time.perf_counter()
    a, cp = _carr(C2)
    m, n = a.shape
    s = np.ascontiguousarray(sorted(S_star), dtype=np.int32)
    v = _LIB.pm_swap_t1(cp, m, n,
                        s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        len(s), float(f_star))
    if v < 0:
        return _py.swap_T1(C2, S_star, f_star)   # allocation failure
    return float(v), time.perf_counter() - t0


def swap_descent(C, p, S0=None, max_iters=200, seed=0):
    if not HAVE_C:
        return _py.swap_descent(C, p, S0=S0, max_iters=max_iters, seed=seed)
    a, cp = _carr(C)
    m, n = a.shape
    if S0 is None:
        rng = np.random.default_rng(seed)
        S0 = sorted(rng.choice(m, size=p, replace=False).tolist())
    s = np.ascontiguousarray(sorted(S0), dtype=np.int32)
    iters = ctypes.c_int(0)
    f = _LIB.pm_swap_descent(cp, m, n, p,
                             s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                             max_iters, ctypes.byref(iters))
    if f < 0:
        return _py.swap_descent(C, p, S0=S0, max_iters=max_iters, seed=seed)
    return tuple(int(x) for x in s), float(f), int(iters.value)
