# High-Order Time Integration: Options, Tradeoffs, and Direction

## Goal

Order 5/6 accuracy above RK4, **keeping communication low** (each RHS eval is a
ghost exchange — the solver's main bottleneck), and staying **RK-like** (single
solution vector, as ETS/ETS_MSRK already do; no state-propagation rewrite).

For explicit methods, high order, few RHS evals, and hyperbolic (imaginary-axis)
stability compete for the same coefficient freedom — the families below trade
these off differently.

Stability metric: apply the method to `y'=λy`; stable where `|R(z)|≤1`. The
**imaginary-axis** intercept is the CFL limit for advection/wave terms (usually
binding); the **real-axis** intercept is headroom for dissipation (Kreiss-Oliger).

## What's already in the tree

Single-step RK (single solution vector, pure Butcher tables in ETS):

| scheme | order | evals/step | imag stab | real stab | notes |
|---|---|---|---|---|---|
| RK4 | 4 | 4 | 2.83 | 2.79 | baseline |
| RK5 | 5 | 6 | **0.85** | 3.39 | best single-step hyperbolic choice |
| RK6 (Luther) | 6 | 7 | 0.079 | 2.86 | needs KO dissipation to be stable |

Multistep RK (single solution vector + reused base-point derivative — ETS_MSRK):

| scheme | order | fresh evals/step | imag stab |
|---|---|---|---|
| RK4-MSRK2(1) | 4 | 3 (−25%) | 2.54 |
| RK4-MSRK2(2) | 4 | 3 (−25%) | 2.46 |
| RK4-MSRK3 | 4 | 2 (−50%) | 1.31 |

All verified by `test/src/testTimeIntegratorOrder.cpp` (order + stability).
Note: single-step high order *raises* communication; MSRK *lowers* it.

## Exploration summary (Python prototypes, not committed methods)

Two derivation tools were built:
- `scripts/derive_msrk_coefficients.py` — MSRK/two-step-RK order conditions via
  B-series (validated against the RK4-MSRK paper), stability, order tests.
- `scripts/derive_peer_coefficients.py` — peer / general-linear methods.

Key numerical findings:

1. **Single-step high order kills imaginary-axis stability** (RK6 = 0.079) —
   only usable on hyperbolic terms with dissipation.
2. **Strict order-5 MSRK** (base-point reuse, 1 history) → order 5 at 5 evals,
   imag ~1.29; pushing to 4 evals (2 history) collapsed to ~0.06 — reusing *old
   integer-step-back* points is what breaks multistep stability.
3. **Peer methods** (a *cloud* of high-order stage values reused at fractional
   recent abscissae) hit order 6 at RK4's 4-eval budget, imag ~1.2–1.7, ~3–4
   orders more accurate than RK4 — the highest ceiling, but needs a stage-value
   cloud (a Dendro state-handling rewrite).

## Decision / direction

- **Now, for accuracy with zero structural risk:** use the in-tree **RK5**
  (order 5, single-step, better hyperbolic stability than RK6) or **RK6** where
  dissipation covers the imaginary axis. Cost: more communication than RK4.
- **The stated target ("RK4-MSRK style, but order 5/6, high stability, low
  communication"):** a **higher-order two-step / multistep Runge-Kutta (TSRK)**
  method — single propagated solution + reused prior-step stage derivatives (same
  storage model as ETS_MSRK, *no cloud*). The RK4-MSRK methods are the order-4
  case; the generalization is a bounded *derivation* project (the order boost and
  stability optimization use the TSRK order conditions). This is the primary
  forward path.
- **Peer methods:** documented as the future high-ceiling option; revisit if the
  TSRK ceiling is insufficient and a state-propagation rework is acceptable.

## TSRK result: RK-like, and it beats the peer cloud

Prototyped an explicit TSRK that **propagates a single solution vector** and
reuses the previous 1–2 steps' *stage* derivatives (recent, fractional history) —
the RK-like storage model of `ETS_MSRK`, no cloud. Order is enforced by uniform
high stage order (polynomial exactness), so the reused stages stay accurate.
Because every stage hangs off the single `y_n` and the reused-derivative modes
vanish at `z=0`, the methods are **automatically zero-stable** (no parasitic
roots near the unit circle) — the single-solution constraint turns into a
stability *advantage*.

Comparison at order 6, RK4's 4-eval budget (`scripts/derive_tsrk_coefficients.py`
vs `scripts/derive_peer_coefficients.py`):

| approach | imag stab | real stab | RK-like? | state |
|---|---|---|---|---|
| single-step RK6 | 0.079 | 2.86 | yes, but 7 evals | few temps |
| peer (cloud) | ~1.4–1.7 | ~1.1 | no (stage cloud) | 2s vectors |
| **TSRK (single vec)** | **~1.85–1.92** | ~1.1–1.3 | **yes** | y + 2s deriv vectors |

TSRK evals-vs-stability at order 6: 4 evals → imag ~1.9, 5 evals → ~2.2, 6 evals
→ ~2.5 (approaching RK4's 2.83). Accuracy ~4 orders better than RK4.

### Finalized order-6 TSRK, 4 fresh evals/step (design pass)

Reproducible (best of `derive_tsrk_coefficients.best_method` seeds 0–11, order 6,
s=4, depth 2). imag 1.85, real 1.31, zero-stable, stable into the 2nd quadrant to
`z ≈ −0.8+0.8i`; local error constant `C ≈ 0.88` (`err_loc ≈ C·h⁷`; RK4's is
O(1)), global `err@h=1/16 ≈ 2.1e-9`. Convergence rates 6.39 → 6.18 → 6.09.

```
c  = [0.52208859, 0.74526195, 0.95717378, 1.32694299]
A1 = [[ 0.01359319,  0.21175158, -0.12595036,  0.47571041],
      [-0.07083360,  0.17706374,  0.15598039, -0.04203496],
      [-0.07471584,  0.25898457,  0.12873016,  0.21921271],
      [-0.29574671,  0.43416092,  0.67832536,  0.07602501]]
A2 = [[-0.03771813,  0.14116513, -0.14028790, -0.01617532],
      [-0.05424334,  0.18847607, -0.17007719,  0.00377513],
      [-0.04246131,  0.10736428, -0.02311686, -0.13501934],
      [-0.19196398,  0.43391221, -0.02721906, -0.48287152]]
r  = [[ 0.0,         0.0,         0.0,        0.0],
      [ 0.55715570,  0.0,         0.0,        0.0],
      [ 0.03143044,  0.48676497,  0.0,        0.0],
      [-0.01615572, -0.38659089,  1.10506735, 0.0]]
b  = [ 0.18608798,  0.29389562,  0.11922370, -0.00392003]
v1 = [ 0.07838760,  0.17781688,  0.06160740,  0.23596605]
v2 = [-0.04795278,  0.13881943, -0.06754081, -0.17239103]
```

Per step (F = f(Y); F1..F4 are the 4 fresh evals = 4 ghost syncs):
```
Yi = y_n + h·( A1[i]·F^(n-1) + A2[i]·F^(n-2) + sum_{j<i} r[i,j]·Fj ),  Fi = f(Yi)
y_{n+1} = y_n + h·( b·F + v1·F^(n-1) + v2·F^(n-2) )
history slide:  F^(n-2) <- F^(n-1);  F^(n-1) <- [F1,F2,F3,F4]
```
Storage: `y_n` + 8 derivative vectors. Needs bootstrap + remesh/dt-change
re-seeding (same as `ETS_MSRK`). Coefficients are numerically optimized (not a
closed-form construction) — verified, reproducible, but not yet "vetted"; a
production version should also confirm robustness under variable dt.

## Literature

- Jackiewicz & Tracogna, *A general class of two-step Runge-Kutta methods*, SIAM
  J. Numer. Anal. 32 (1995) — the TSRK framework for the target direction.
- Sanches, Brandt, Kalinani, Ji, Schnetter, *Accelerating NR Simulations with New
  Multistep Fourth-Order RK Methods*, arXiv:2603.05763 — the in-tree RK4-MSRK.
- Peer methods (future option): Schmitt & Weiner, SINUM 42 (2004); Weiner,
  Biermann, Schmitt, Podhaisky, *Explicit two-step peer methods*, Comput. Math.
  Appl. (2008); Klinge, Weiner, Podhaisky, *Optimally zero-stable explicit peer
  methods with variable nodes*, BIT (2018); EPPEER software +
  `ivpepp.f90`/`man_epp.pdf` at
  https://www.mathematik.uni-marburg.de/~schmitt/peer/ (epp5f3 = order-5, FSAL).
