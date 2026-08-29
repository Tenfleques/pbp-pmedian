# pbp-pmedian

Exact p-median solving and **per-entry cost tolerances** via the
Hammer–Beresnev (pseudo-Boolean) representation.

Given an m×n cost matrix (facilities × clients) and p, the package provides:

- **`solve(C, p)`** — routed exact solving: an expand-until-certificate
  reduced model (BEAMR/ZEBRA/CFIE truncation lineage, with the a-priori
  no-overflow optimality certificate made explicit) or a two-phase Benders
  decomposition (implemented from Durán-Mateluna–Ales–Elloumi, EJOR 2023,
  whose cut separation walks exactly the Hammer–Beresnev prefix order),
  chosen by a one-division router.
- **`tolerance_profile(C, p)`** — the exact upper tolerance of every serving
  cost entry: the largest increase of C[i,j] that keeps the optimal set
  optimal. Computed by the branch split `d_min = min(A, B)` — one certified
  facility solve per open facility plus one level-restricted entry solve per
  client — with saturation detected exactly.
- Screens and certificates: the free `f2_screen`, the caching rule
  `reuse_certificate` (2·d < margin₂ ⇒ a cached optimum survives a
  perturbation, no solve; the constant 2 is tight), truncation transforms,
  swap descent/bounds.
- A **C kernel** for the solver-free primitives (`bash build_pmedian.sh`),
  mirroring `core.py` function-for-function with automatic fallback.

## Install

```bash
pip install pbp-pmedian            # numpy + scipy (HiGHS via scipy)
pip install pbp-pmedian[highs]     # + native highspy backend with MIP starts
pip install pbp-pmedian[full]      # + tmc-pbp for Hammer–Beresnev exports
bash build_pmedian.sh              # optional: build the C kernel (from a repo checkout)
```

## Quickstart

```python
import numpy as np
from pbp_pmedian import solve, tolerance_profile, route

rng = np.random.default_rng(0)
C = rng.uniform(1, 50, size=(60, 80))     # 60 candidate facilities, 80 clients

value, S, secs, status, engine = solve(C, p=8)   # proven optimal, engine=route(C, 8)

prof = tolerance_profile(C, 8, S_star=S, f_star=value)
# prof["tol"][j]: exact upper tolerance of client j's serving entry
# (np.inf where the entry is saturated — provably unbounded)
```

One call (or one shell command) for the whole pipeline — gap estimate →
engine routing → certified solve → tolerance layer:

```python
from pbp_pmedian import analyze, format_report
print(format_report(analyze(C, 8, tolerances="exact")))
```

```bash
pbp-pmedian costs.csv -p 8                       # free F2 tolerance screen
pbp-pmedian costs.csv -p 8 --tolerances exact --json report.json
pbp-pmedian costs.csv -p 8 --engine benders --clients 0,4,17
```

The matrix file is plain numeric text (comma- or whitespace-separated, rows
= facilities, columns = clients). The engine router (ρ = p/m ≥ 0.10 →
expand, else Benders) scored 0.70 oracle agreement at 1.20× oracle
wall-clock on 23 fresh out-of-sample instances — a good default, not a
law; `--engine` overrides it. With `highspy` installed the Benders engine
also separates at every incumbent ("harvest"), measured −13.5 % on
OR-Library totals and 2.9× on the rw100 class at identical guarantees.

Hammer–Beresnev interop (optional, needs `tmc-pbp`):

```python
from pbp_pmedian import hb_polynomial
poly = hb_polynomial(C)      # the instance's pseudo-Boolean decomposition
```

## Validation and measured performance

Every algorithm here is a port of code validated against full enumeration
and against a 47-instance OR-Library/TSPLIB benchmark campaign (research
repository, 2026-08; all numbers from committed CSVs, measured on an
M-series MacBook Air with the same solver stack):

| component | validation | measured |
|---|---|---|
| expand-until-certificate | value == full model on every solved instance | 49/52 benchmark bases solved, median 3.7× vs the full model |
| two-phase Benders | == full model; every cut valid at the optimum | all 40 OR-Library instances proven, median 1.1 s |
| entry tolerance | == enumeration; 438/438 benchmark entries exact | median 7.3× vs one exact re-solve per entry |
| router | vs fastest-engine oracle | total wall-clock ≈ 1.00× oracle on the calibration set |

The test suite reruns the enumeration cross-checks (`pytest tests/`).

## Honest limits

- Per-entry tolerances of NP-hard problems admit no polynomial
  constant-factor approximation unless P=NP (van Hoesel–Wagelmans); this
  package is exact anyway on the instances it can solve, and inherits the
  solver's limits where it cannot.
- LP-loose random instances (asymmetric uniform-cost, n ≥ 250) defeat every
  engine here at hour-scale budgets — documented, not hidden. On such
  instances prefer `engine="full"` for moderate n.
- The router's rule is calibrated on the benchmark families above and should
  be re-confirmed on your instance family before being trusted blindly.
- The Benders implementation is callback-free (master re-solves); the
  original paper's branch-and-Benders-cut is faster on the largest
  instances.

## Relationship to tmc-pbp

This package operationalizes the p-median side of the pseudo-Boolean
polynomial (PBP) research programme: the reduced models are degree
truncations of the instance's Hammer–Beresnev polynomial, the entry solve's
level restriction is a theorem about its monomial chains, and `tmc-pbp`
(optional dependency) computes the polynomial itself for interpretability
and export.

## License

MIT — Tendai Mapungwana Chikake.
