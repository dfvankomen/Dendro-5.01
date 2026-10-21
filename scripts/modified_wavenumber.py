"""
Modified-wavenumber (spectral resolution) analysis for compact vs explicit
first-derivative schemes.

This is the theoretical "quality" figure for the methods paper: it shows how
faithfully each scheme reproduces the derivative of a Fourier mode e^{ikx} as a
function of the scaled wavenumber w = k*h. The exact derivative has modified
wavenumber w'(w) = w (the diagonal). A scheme that tracks the diagonal out to
larger w resolves shorter wavelengths per grid point -- i.e. it needs *fewer
points per wavelength* for a given fidelity, which is the basis of the
"compact lets us use smaller meshes" argument.

For a centered first-derivative scheme written in the general compact form

    beta f'_{i-2} + alpha f'_{i-1} + f'_i + alpha f'_{i+1} + beta f'_{i+2}
        = (1/h) [ a (f_{i+1}-f_{i-1})/2
                + b (f_{i+2}-f_{i-2})/4
                + c (f_{i+3}-f_{i-3})/6 ]

the modified wavenumber is (see Lele 1992, Eq. 2.4):

    w'(w) =  [ a sin(w) + (b/2) sin(2w) + (c/3) sin(3w) ]
            / [ 1 + 2 alpha cos(w) + 2 beta cos(2w) ]

Explicit schemes are the special case alpha = beta = 0 (P = I).

Only *interior* coefficients enter here -- the spectral resolution of a scheme
is an interior property; boundary closures do not appear in Fourier analysis.

The verified built-in coefficient sets below are the standard published values.
JTT6 is exactly the standard 6th-order tridiagonal Pade scheme (the paper's
reference compact scheme). To analyse the repo's custom schemes (JTP6, BL6,
BorisO6, BYU*), dump their interior diagonals from C++ via
MatrixDiagonalEntries::toStdOut() and add them to a TOML passed with
--coeffs (see SCHEME_COEFFS for the field layout), rather than transcribing
by hand.

Usage:
    python scripts/modified_wavenumber.py
    python scripts/modified_wavenumber.py --coeffs scripts/scheme_coeffs.toml \
        --scheme E4 --scheme E6 --scheme JTT6 \
        --out-dir scripts/spectral_plots
"""

import argparse
import math
import os
import sys


# Interior coefficients, first derivative, in the (alpha, beta, a, b, c) form
# above. All entries here are the standard published values.
#   verified: True  -> closed-form published coefficients, safe to cite
# Add the repo's custom schemes via --coeffs (TOML) once dumped from C++.
SCHEME_COEFFS = {
    # explicit (alpha = beta = 0)
    "E2":   dict(alpha=0.0,     beta=0.0, a=1.0,      b=0.0,      c=0.0,
                 order=2, compact=False, verified=True),
    "E4":   dict(alpha=0.0,     beta=0.0, a=4.0/3.0,  b=-1.0/3.0, c=0.0,
                 order=4, compact=False, verified=True),
    "E6":   dict(alpha=0.0,     beta=0.0, a=3.0/2.0,  b=-3.0/5.0, c=1.0/10.0,
                 order=6, compact=False, verified=True),
    # compact tridiagonal Pade (beta = 0)
    "JTT4": dict(alpha=1.0/4.0, beta=0.0, a=3.0/2.0,  b=0.0,      c=0.0,
                 order=4, compact=True,  verified=True),
    # JTT6 == standard 6th-order tridiagonal Pade (paper's reference scheme)
    "JTT6": dict(alpha=1.0/3.0, beta=0.0, a=14.0/9.0, b=1.0/9.0,  c=0.0,
                 order=6, compact=True,  verified=True),
}

# Schemes plotted by default (kept small: the paper's core comparison is
# compact-6 vs explicit-4 and explicit-6).
DEFAULT_SCHEMES = ["E4", "E6", "JTT6"]


def modified_wavenumber(w, p):
    """w'(w) for first derivative given coeff dict p. w may be a list."""
    a, b, c = p["a"], p["b"], p["c"]
    al, be = p["alpha"], p["beta"]
    num = a * math.sin(w) + (b / 2.0) * math.sin(2 * w) \
        + (c / 3.0) * math.sin(3 * w)
    den = 1.0 + 2.0 * al * math.cos(w) + 2.0 * be * math.cos(2 * w)
    return num / den


def resolving_efficiency(p, tol=0.01):
    """Largest w (as a fraction of pi) for which |w'(w) - w| / pi <= tol.

    This is the standard "resolving efficiency": the highest scaled wavenumber
    the scheme reproduces to within `tol`. Higher is better. Points-per-
    wavelength at that limit is 2*pi / w_max.
    """
    N = 2000
    w_max = 0.0
    for i in range(1, N + 1):
        w = math.pi * i / N
        if abs(modified_wavenumber(w, p) - w) / math.pi <= tol:
            w_max = w
        else:
            break
    return w_max


def load_coeffs(path):
    """Merge scheme coeffs from a TOML file into SCHEME_COEFFS (override)."""
    try:
        import tomllib  # py3.11+
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except ModuleNotFoundError:
        try:
            import tomli  # py3.8-3.10
            with open(path, "rb") as f:
                data = tomli.load(f)
        except ModuleNotFoundError:
            print("warning: no tomllib/tomli available; --coeffs ignored",
                  file=sys.stderr)
            return
    for name, d in data.get("scheme", {}).items():
        SCHEME_COEFFS[name] = dict(
            alpha=float(d.get("alpha", 0.0)), beta=float(d.get("beta", 0.0)),
            a=float(d["a"]), b=float(d.get("b", 0.0)), c=float(d.get("c", 0.0)),
            order=int(d.get("order", 0)), compact=bool(d.get("compact", True)),
            verified=bool(d.get("verified", False)))


def print_summary(schemes, tol):
    print(f"Resolving efficiency (|w'-w|/pi <= {tol}); higher w_max is better")
    print(f"{'Scheme':<10}{'order':>6}{'compact':>9}"
          f"{'w_max/pi':>10}{'pts/wave':>10}{'verified':>10}")
    print("-" * 55)
    for s in schemes:
        p = SCHEME_COEFFS[s]
        wmax = resolving_efficiency(p, tol)
        ppw = (2 * math.pi / wmax) if wmax > 0 else float("inf")
        print(f"{s:<10}{p['order']:>6}{str(p['compact']):>9}"
              f"{wmax/math.pi:>10.3f}{ppw:>10.2f}"
              f"{str(p['verified']):>10}")


def plot(schemes, out_dir, tol):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot", file=sys.stderr)
        return False

    N = 400
    ws = [math.pi * i / N for i in range(N + 1)]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    # ideal
    ax.plot(ws, ws, "k--", linewidth=1.0, label="exact (w'=w)")
    for s in schemes:
        p = SCHEME_COEFFS[s]
        wp = [modified_wavenumber(w, p) for w in ws]
        style = "-" if p["compact"] else ":"
        ax.plot(ws, wp, style, linewidth=1.6, label=s)
    ax.set_xlabel("w = k h")
    ax.set_ylabel("modified wavenumber  w'(w)")
    ax.set_title("Spectral resolution of first-derivative schemes")
    ax.set_xlim(0, math.pi)
    ax.set_ylim(0, math.pi)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "modified_wavenumber_d1.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", default=None,
                    help="TOML with extra [scheme.NAME] interior coefficients "
                         "(alpha, beta, a, b, c, order, compact, verified)")
    ap.add_argument("--scheme", action="append", default=None,
                    help="scheme name to include; repeatable. "
                         "default: E4 E6 JTT6")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__),
                                         "spectral_plots"))
    ap.add_argument("--tol", type=float, default=0.01,
                    help="resolving-efficiency tolerance (default 0.01)")
    args = ap.parse_args()

    if args.coeffs:
        load_coeffs(args.coeffs)

    schemes = args.scheme or DEFAULT_SCHEMES
    missing = [s for s in schemes if s not in SCHEME_COEFFS]
    if missing:
        print(f"unknown scheme(s): {missing}\n"
              f"known: {sorted(SCHEME_COEFFS)}", file=sys.stderr)
        return 1

    print_summary(schemes, args.tol)
    plot(schemes, args.out_dir, args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
