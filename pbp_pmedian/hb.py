"""Interop with the tmc-pbp package (optional dependency).

The p-median objective's Hammer-Beresnev polynomial is exactly what
``pbp.core.create_pbp`` computes from the cost matrix; this module bridges
to it for interpretability and export without making tmc-pbp a hard
dependency. Install with ``pip install pbp-pmedian[full]``.
"""


def hb_polynomial(C):
    """Full Hammer-Beresnev decomposition of the instance via tmc-pbp.
    Returns the create_pbp DataFrame (monomials y, coefficients, degrees)."""
    try:
        from pbp.core import create_pbp
    except ImportError as exc:
        raise ImportError(
            "tmc-pbp is not installed; install pbp-pmedian[full] "
            "(or pip install tmc-pbp) for Hammer-Beresnev exports"
        ) from exc
    return create_pbp(C)


def hb_vector(C):
    """Fixed-length PBP vector of the instance via tmc-pbp."""
    try:
        from pbp.core import pbp_vector
    except ImportError as exc:
        raise ImportError(
            "tmc-pbp is not installed; install pbp-pmedian[full] "
            "(or pip install tmc-pbp) for Hammer-Beresnev exports"
        ) from exc
    return pbp_vector(C)
