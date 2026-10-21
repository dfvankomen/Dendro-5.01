#!/usr/bin/env python3
"""Maximum stable CFL number for each ETS RK scheme, with Kreiss-Oliger
dissipation in the spatial operator.

stability_regions.py gives each scheme's region; modified_wavenumber.py gives
the derivative symbol w'(xi). This adds the KO symbol and finds the largest CFL
whose eigenvalue locus fits inside each region:

    d_t u = -c sum_d d_d u + sigma sum_d KO_d u
    lambda*dt = nu * L(xi),  L = -i*Wsum - r*Ssum,  nu = c dt/h,  r = sigma/c

dendrolib's ko_deriv21/42/64 are -delta^(2p)/(4^p h), so every one of them has
symbol -sin^2p(xi/2)/h -- max magnitude exactly 1/h, independent of order. The
prefactor is 1/h, NOT h^(2p-1).

Wsum has two forms, reported as a bracket: "sum" (sum_d w') and "iso"
(sqrt(sum_d w'^2)). Quote iso; sum is falsified by MSRK3 having run production.
KO is additive over directions either way, as rhs.cpp applies it.

Linear, constant-coefficient and scalar: no gauge terms, no nonlinearity, no
CAHD (which cannot enter -- it is nonlinear and carries an explicit 1/dt).
Ratios between schemes are solid, absolute nu is indicative; measured against
BSSN 2026-10-02, where it runs 1-8% optimistic. Full writeup in the vault:
Dendro-GR Solver / Experiments / RK CFL limits with KO.

Run:  python3 scripts/cfl_with_ko.py [--deriv E6] [--nko 3]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modified_wavenumber as MW  # noqa: E402
import stability_regions as S  # noqa: E402

ORDER = ["RK4", "RK4_RALSTON", "RK4_MSRK2_1", "RK4_MSRK2_2", "RK6_TSRK", "RK3",
         "RK4_MSRK3", "RK5", "RK6", "RK45_CASH_KARP", "RKF45", "RK5_NYSTROM"]


def wprime(xi, p):
    """Vectorized modified wavenumber (Lele form); matches MW's scalar fn."""
    a, b, c = p["a"], p["b"], p["c"]
    al, be = p["alpha"], p["beta"]
    num = a * np.sin(xi) + (b / 2.0) * np.sin(2 * xi) + (c / 3.0) * np.sin(3 * xi)
    den = 1.0 + 2.0 * al * np.cos(xi) + 2.0 * be * np.cos(2 * xi)
    return num / den


def ko_symbol(xi, nko):
    """|symbol| of dendrolib's KO operator: sin^{2p}(xi/2), max 1 at Nyquist."""
    return np.sin(xi / 2.0) ** (2 * nko)


def locus(deriv, nko, r, ndim, model, n):
    """L(xi) = -i*Wsum - r*Ssum over the mode grid, flattened."""
    xi = np.linspace(0.0, np.pi, n)
    w, s = wprime(xi, deriv), ko_symbol(xi, nko)
    if ndim == 1:
        W, Ssum = w, s
    else:
        W1, W2, W3 = np.meshgrid(w, w, w, indexing="ij")
        S1, S2, S3 = np.meshgrid(s, s, s, indexing="ij")
        Ssum = (S1 + S2 + S3).ravel()
        if model == "sum":
            W = (W1 + W2 + W3).ravel()
        else:
            W = np.sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2).ravel()
    return -1j * np.asarray(W).ravel() - r * np.asarray(Ssum).ravel()


def max_nu(m, L, hi=20.0, tol=1e-6):
    """Largest nu with rho(nu*L) <= 1 everywhere, by bisection on nu."""
    def ok(nu):
        return bool(np.all(S.rho(m, nu * L) <= 1.0 + 1e-8))
    if not ok(1e-9):
        return 0.0
    lo = 1e-9
    while ok(hi) and hi < 1e4:
        hi *= 2.0
    while hi - lo > tol * max(lo, 1e-9):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def validate(deriv, n=4001):
    """Pipeline checks: the vectorized symbol, and the sigma=0 1D closed form."""
    print("=== validation ===")
    xi = np.linspace(0.05, np.pi - 0.05, 17)
    err = max(abs(wprime(np.array([x]), deriv)[0] - MW.modified_wavenumber(x, deriv))
              for x in xi)
    print(f"  vectorized w' vs modified_wavenumber.py : max diff {err:.2e}"
          f"   {'PASS' if err < 1e-13 else 'FAIL'}")

    # With no KO the 1D locus is purely imaginary, so nu_max must be exactly
    # the scheme's imaginary-axis intercept divided by max|w'|.
    L0 = locus(deriv, 3, 0.0, 1, "sum", n)
    wmax = float(np.max(np.abs(wprime(np.linspace(0, np.pi, n), deriv))))
    print(f"  max|w'| for this derivative scheme      : {wmax:.6f}")
    bad = 0
    for nm in ["RK4", "RK3", "RK6", "RK4_MSRK3", "RK6_TSRK"]:
        got = max_nu(S.ALL[nm], L0)
        want = S.intercept(S.ALL[nm], "imag") / wmax
        rel = abs(got - want) / want
        if rel > 2e-3:
            bad += 1
        print(f"  sigma=0, 1D, {nm:<12} nu={got:.4f}  vs intercept/max|w'|"
              f"={want:.4f}  {'PASS' if rel <= 2e-3 else 'FAIL'}")
    print()
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv", default="E6", help="derivative scheme (E4/E6/JTT6)")
    ap.add_argument("--nko", type=int, default=3,
                    help="KO order p: delta^(2p). 6th-order BSSN at pw=3 -> 3")
    ap.add_argument("--n", type=int, default=33, help="modes per direction (3D)")
    ap.add_argument("--n1d", type=int, default=2001, help="modes (1D)")
    ap.add_argument("--sigmas", default="0.0,0.2,0.4")
    args = ap.parse_args()

    deriv = MW.SCHEME_COEFFS[args.deriv]
    sigmas = [float(x) for x in args.sigmas.split(",")]

    print(__doc__.split("Run:")[0].strip()[:0] or "", end="")
    print(f"CFL limits with KO -- derivative {args.deriv}, KO delta^{2*args.nko}"
          f" (p={args.nko}), c=1\n")
    bad = validate(deriv)

    for ndim, model, n in [(1, "sum", args.n1d), (3, "iso", args.n),
                           (3, "sum", args.n)]:
        tag = f"{ndim}D" + ("" if ndim == 1 else f" / {model}")
        print(f"=== max CFL number  nu = c*dt/h   [{tag}] ===")
        head = f"{'scheme':<16}" + "".join(f"{'s=' + str(s):>10}" for s in sigmas)
        print(head + f"{'KO effect':>12}{'vs RK4':>9}")
        base = {}
        rows = []
        for nm in ORDER:
            vals = [max_nu(S.ALL[nm], locus(deriv, args.nko, s, ndim, model, n))
                    for s in sigmas]
            rows.append((nm, vals))
            base[nm] = vals
        rk4_last = base["RK4"][-1]
        for nm, vals in rows:
            eff = (vals[-1] / vals[0] - 1.0) * 100.0 if vals[0] > 0 else float("nan")
            rel = vals[-1] / rk4_last if rk4_last > 0 else float("nan")
            print(f"{nm:<16}" + "".join(f"{v:10.4f}" for v in vals)
                  + f"{eff:+11.1f}%{rel:9.3f}")
        print()

    # grid convergence for the 3D scan
    print("=== 3D grid convergence (sum model, largest sigma) ===")
    s = sigmas[-1]
    for nm in ["RK4", "RK4_MSRK2_1", "RK6", "RK6_TSRK"]:
        a = max_nu(S.ALL[nm], locus(deriv, args.nko, s, 3, "sum", args.n))
        b = max_nu(S.ALL[nm], locus(deriv, args.nko, s, 3, "sum", args.n + 16))
        print(f"  {nm:<16} n={args.n}: {a:.4f}   n={args.n+16}: {b:.4f}"
              f"   drift {abs(b-a)/a*100:.2f}%")
    print()
    print("ALL CHECKS PASS" if bad == 0 else f"VALIDATION FAILURES: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
