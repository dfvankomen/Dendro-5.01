#!/usr/bin/env python3
"""Plot absolute-stability regions for the ETS Runge-Kutta schemes.

Draws the region {z = lambda*h : the method is absolutely stable} in the complex
plane, in the style of the MSRK paper (arXiv:2603.05763). Three method families,
one uniform notion of stability (spectral radius of the one-step amplification /
companion map <= 1):

  * single-step RK  -- R(z) = 1 + sum_k (b^T A^k 1) z^{k+1}; region |R(z)| <= 1.
  * MSRK            -- companion map on [y, history derivatives]; rho(M(z)) <= 1.
  * TSRK (order 6)  -- amplification map on [y, G^(1), G^(2)]; rho(M(z)) <= 1.

Coefficients are transcribed from ODE/include/rkTableau.h (the C++ single source
of truth) with the same provenance. The MSRK/TSRK companion maps mirror
scripts/derive_{msrk,tsrk}_coefficients.py exactly.

Outputs (scripts/stability_plots/):
  stability_all.png            every scheme overlaid (colors + dash styles)
  stability_rk_singlestep.png  the classical single-step RK family
  stability_msrk_vs_rk4.png    RK4 reference (filled) + the 3 MSRK variants
  stability_tsrk_vs_ref.png    RK4/RK5/RK6 references + the order-6 TSRK

Also prints the imaginary- and real-axis intercepts (CFL / dissipation headroom).

Run:  python3 scripts/stability_regions.py
"""
import os

import numpy as np

np.seterr(all="ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "stability_plots")


# =====================================================================
# tableaux (from ODE/include/rkTableau.h)
# =====================================================================
def _rk(nodes, weights, aij):
    s = len(nodes)
    return dict(kind="rk", s=s, c=np.array(nodes, float),
                b=np.array(weights, float), A=np.array(aij, float).reshape(s, s))


RK = {}

RK["RK3"] = _rk(
    [0.0, 1.0, 0.5],
    [1 / 6, 1 / 6, 2 / 3],
    [0, 0, 0,
     1, 0, 0,
     1 / 4, 1 / 4, 0])

RK["RK4"] = _rk(
    [0.0, 0.5, 0.5, 1.0],
    [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    [0, 0, 0, 0,
     1 / 2, 0, 0, 0,
     0, 1 / 2, 0, 0,
     0, 0, 1, 0])

RK["RK5"] = _rk(  # Butcher, 6 stages
    [0.0, 1 / 4, 1 / 4, 1 / 2, 3 / 4, 1.0],
    [7 / 90, 0.0, 32 / 90, 12 / 90, 32 / 90, 7 / 90],
    [0, 0, 0, 0, 0, 0,
     1 / 4, 0, 0, 0, 0, 0,
     1 / 8, 1 / 8, 0, 0, 0, 0,
     0, -1 / 2, 1, 0, 0, 0,
     3 / 16, 0, 0, 9 / 16, 0, 0,
     -3 / 7, 2 / 7, 12 / 7, -12 / 7, 8 / 7, 0])

RK["RK4_RALSTON"] = _rk(
    [0.0, 0.4, 0.4557372542187894, 1.0],
    [0.17476028226269036, -0.551480662878733, 1.2055355993965235,
     0.17118478121951902],
    [0, 0, 0, 0,
     0.4, 0, 0, 0,
     0.2969776092477536, 0.15875964497103584, 0, 0,
     0.21810038822592046, -3.050965148692931, 3.8328647604670105, 0])

RK["RK5_NYSTROM"] = _rk(
    [0.0, 1 / 3, 2 / 5, 1.0, 2 / 3, 4 / 5],
    [23 / 192, 0.0, 125 / 192, 0.0, -27 / 64, 125 / 192],
    [0, 0, 0, 0, 0, 0,
     1 / 3, 0, 0, 0, 0, 0,
     4 / 25, 6 / 25, 0, 0, 0, 0,
     1 / 4, -3, 15 / 4, 0, 0, 0,
     2 / 27, 10 / 9, -50 / 81, 8 / 81, 0, 0,
     2 / 25, 12 / 25, 2 / 15, 8 / 75, 0, 0])

RK["RK45_CASH_KARP"] = _rk(  # 5th-order weights of the embedded pair
    [0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8],
    [37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771],
    [0, 0, 0, 0, 0, 0,
     1 / 5, 0, 0, 0, 0, 0,
     3 / 40, 9 / 40, 0, 0, 0, 0,
     3 / 10, -9 / 10, 6 / 5, 0, 0, 0,
     -11 / 54, 5 / 2, -70 / 27, 35 / 27, 0, 0,
     1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096, 0])

RK["RKF45"] = _rk(  # 5th-order weights of Runge-Kutta-Fehlberg
    [0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2],
    [16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
    [0, 0, 0, 0, 0, 0,
     1 / 4, 0, 0, 0, 0, 0,
     3 / 32, 9 / 32, 0, 0, 0, 0,
     1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0,
     439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0,
     -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0])

RK["RK6"] = _rk(  # Luther 1968, 7 stages
    [0, 1, 0.5, 0.66666666666666663, 0.17267316464601143,
     0.82732683535398854, 1],
    [0.050000000000000003, 0, 0.35555555555555557, 0, 0.2722222222222222,
     0.2722222222222222, 0.050000000000000003],
    [0, 0, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0,
     0.375, 0.125, 0, 0, 0, 0, 0,
     0.29629629629629628, 0.07407407407407407, 0.29629629629629628, 0, 0, 0, 0,
     0.051640768506639186, -0.049335189898860411, 0.29601113939316243,
     -0.1256435533549298, 0, 0, 0,
     -1.1854881643947648, -0.23637909581542529, -0.74817562366625956,
     0.88085458023927043, 2.1165151389911681, 0, 0,
     4.50650248872424, 0.66666666666666663, 6.0173399699313057,
     -4.111704479703632, -7.0189140975801996, 0.94010945196161799, 0])


# --------------------------------------------------------------------
# MSRK (arXiv:2603.05763, Table 1) -- 4 logical stages, first H reused
# --------------------------------------------------------------------
def _msrk(H, aij, b, c):
    A = np.zeros((4, 4))
    for (i, j), val in aij.items():
        A[i, j] = val
    return dict(kind="msrk", H=H, S=4, A=A, b=np.array(b, float),
                c=np.array(c, float))


MSRK = {}
MSRK["RK4_MSRK2_1"] = _msrk(
    1,
    {(2, 0): -49 / 1250, (2, 1): 399 / 1250, (3, 0): 7033 / 960000,
     (3, 1): -217633 / 210000, (3, 2): 5473 / 10752},
    [-643 / 1536, -4237 / 1092, 38125 / 10752, 4375 / 2496],
    [0.0, 0.0, 7 / 25, -13 / 25])
MSRK["RK4_MSRK2_2"] = _msrk(
    1,
    {(2, 0): 1309 / 15500, (2, 1): -31999 / 15500, (3, 0): -241289 / 5880000,
     (3, 1): 22846301 / 16170000, (3, 2): -936169 / 2587200},
    [-191 / 882, 48241 / 59994, 193750 / 4351347, 100000 / 271791],
    [0.0, 0.0, -99 / 50, 101 / 100])
MSRK["RK4_MSRK3"] = _msrk(
    2,
    {(3, 0): 2511 / 62500, (3, 1): -2268 / 15625, (3, 2): 29061 / 62500},
    [-85 / 1416, 131 / 408, -29 / 24, 15625 / 8024],
    [0.0, 0.0, 0.0, 9 / 25])


# --------------------------------------------------------------------
# TSRK order 6 (get_tsrk_o6_tableau in rkTableau.h) -- s=4, depth=2
# --------------------------------------------------------------------
def _tsrk():
    A1 = np.array([
        0.013593190529617938, 0.21175157807916295, -0.12595036277141974, 0.4757104110436477,
        -0.0708335997874584, 0.17706373925409918, 0.15598039225995208, -0.04203496245036641,
        -0.07471584482081398, 0.2589845732144846, 0.12873016139377028, 0.21921271021279182,
        -0.2957467086460748, 0.4341609248017752, 0.678325363931038, 0.07602500907276638]).reshape(4, 4)
    A2 = np.array([
        -0.03771812922029248, 0.14116512647628252, -0.14028789950350323, -0.016175324253200424,
        -0.05424333545301849, 0.18847606951984108, -0.17007719286131595, 0.0037751342424668377,
        -0.04246130841437379, 0.10736428370341408, -0.02311686313880127, -0.13501933772931757,
        -0.1919639769736794, 0.43391220601192226, -0.027219055049993328, -0.48287151555336155]).reshape(4, 4)
    r = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.55715570041205, 0.0, 0.0, 0.0,
        0.03143043938004049, 0.48676496750677734, 0.0, 0.0,
        -0.016155720486475872, -0.38659089230841814, 1.1050673544673897, 0.0]).reshape(4, 4)
    b = np.array([0.1860879824774312, 0.29389561663531305, 0.11922369721530686,
                  -0.003920034931639108])
    v1 = np.array([0.07838759529108924, 0.1778168831343859, 0.061607402210060086,
                   0.23596604652835984])
    v2 = np.array([-0.047952779499139855, 0.138819433426463, -0.06754081105372743,
                   -0.17239103143390233])
    return dict(kind="tsrk", s=4, depth=2, A=[A1, A2], r=r, b=b, v=[v1, v2])


TSRK = {"RK6_TSRK": _tsrk()}


# =====================================================================
# spectral radius of the one-step map, vectorized over a grid of z
# =====================================================================
def rho_rk(m, Z):
    """|R(z)| for a single-step RK method (polynomial stability function)."""
    A, b, s = m["A"], m["b"], m["s"]
    v = np.ones(s)
    coeffs = [1.0]
    Ak1 = v.copy()
    for _ in range(s):
        coeffs.append(b @ Ak1)
        Ak1 = A @ Ak1
    R = np.zeros_like(Z)
    for k, ck in enumerate(coeffs):
        R = R + ck * Z ** k
    return np.abs(R)


def _specrad_stack(build_cols, dim, Z):
    """rho(M(z)) over grid Z; build_cols(e, Z) returns the e-th column, shape Z."""
    shp = Z.shape
    M = np.zeros(shp + (dim, dim), complex)
    for col in range(dim):
        e = np.zeros(dim)
        e[col] = 1.0
        M[..., :, col] = np.moveaxis(build_cols(e, Z), 0, -1)
    return np.max(np.abs(np.linalg.eigvals(M)), axis=-1)


def rho_msrk(m, Z):
    H, S, A, b = m["H"], m["S"], m["A"], m["b"]
    dim = H + 1

    def build_cols(s, Z):
        y = s[0]
        g = [None] * S
        for hh in range(H):
            g[hh] = np.full(Z.shape, s[1 + hh], complex)
        for i in range(H, S):
            Yi = y + sum(A[i, j] * g[j] for j in range(i))
            g[i] = Z * Yi
        yn = y + sum(b[i] * g[i] for i in range(S))
        out = [yn] + [np.full(Z.shape, s[k], complex) for k in range(2, dim)] \
            + [g[H]]
        return np.array(out)

    return _specrad_stack(build_cols, dim, Z)


def rho_tsrk(m, Z):
    s, depth, A, r, b, v = m["s"], m["depth"], m["A"], m["r"], m["b"], m["v"]
    dim = 1 + depth * s

    def build_cols(st, Z):
        y = np.full(Z.shape, st[0], complex)
        G = [np.array([np.full(Z.shape, st[1 + mm * s + j], complex)
                       for j in range(s)]) for mm in range(depth)]
        Y = [None] * s
        for i in range(s):
            val = y.copy()
            for mm in range(depth):
                val = val + sum(A[mm][i, j] * G[mm][j] for j in range(s))
            val = val + Z * sum(r[i, j] * Y[j] for j in range(i))
            Y[i] = val
        Gnew = [Z * Y[j] for j in range(s)]
        yn = y + Z * sum(b[i] * Y[i] for i in range(s)) \
            + sum(v[mm][j] * G[mm][j] for mm in range(depth) for j in range(s))
        out = [yn]
        out += Gnew                          # newest history
        for mm in range(1, depth):           # shift older history back
            out += [G[mm - 1][j] for j in range(s)]
        return np.array(out)

    return _specrad_stack(build_cols, dim, Z)


def rho(m, Z):
    return {"rk": rho_rk, "msrk": rho_msrk, "tsrk": rho_tsrk}[m["kind"]](m, Z)


# =====================================================================
# axis intercepts (imag = CFL for waves; real = dissipation headroom)
# =====================================================================
def intercept(m, axis, hi=6.0):
    zof = (lambda x: np.array([[complex(0, x)]])) if axis == "imag" \
        else (lambda x: np.array([[complex(-x, 0)]]))
    lo = 0.0
    x, step = 1e-3, 1e-3
    found = 0.0
    xx = step
    while xx < hi:
        if rho(m, zof(xx))[0, 0] > 1 + 1e-8:
            lo, found = xx - step, xx
            break
        xx += step
    if found == 0.0:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + found)
        if rho(m, zof(mid))[0, 0] <= 1 + 1e-8:
            lo = mid
        else:
            found = mid
    return 0.5 * (lo + found)


# =====================================================================
# plotting
# =====================================================================
# name -> (color, linestyle, linewidth)
STYLE = {
    "RK3":            ("#4c72b0", "-",  1.4),
    "RK4":            ("#000000", "-",  2.2),
    "RK5":            ("#dd8452", "--", 1.6),
    "RK5_NYSTROM":    ("#55a868", "-.", 1.6),
    "RK45_CASH_KARP": ("#c44e52", ":",  1.8),
    "RKF45":          ("#8172b3", "--", 1.6),
    "RK4_RALSTON":    ("#937860", "-.", 1.6),
    "RK6":            ("#da8bc3", "-",  2.0),
    "RK4_MSRK2_1":    ("#2ca02c", "--", 1.8),
    "RK4_MSRK2_2":    ("#1f77b4", "-.", 1.8),
    "RK4_MSRK3":      ("#ff7f0e", ":",  2.0),
    "RK6_TSRK":       ("#d62728", "-",  2.6),
}
ALL = {**RK, **MSRK, **TSRK}


def _grid(re=(-7.0, 1.0), im=(-4.5, 4.5), n=600):
    x = np.linspace(re[0], re[1], n)
    y = np.linspace(im[0], im[1], n)
    RE, IM = np.meshgrid(x, y)
    return RE, IM, RE + 1j * IM


def _frame(ax, re, im, title):
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.axvline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlim(re)
    ax.set_ylim(im)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re($\lambda h$)")
    ax.set_ylabel(r"Im($\lambda h$)")
    ax.set_title(title)
    ax.grid(True, lw=0.3, color="0.85")


def plot_outlines(names, fname, title, re=(-7.0, 1.0), im=(-4.5, 4.5),
                  n=600, fill_ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RE, IM, Z = _grid(re, im, n)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    if fill_ref is not None:
        Rf = rho(ALL[fill_ref], Z)
        ax.contourf(RE, IM, Rf, levels=[0.0, 1.0],
                    colors=[STYLE[fill_ref][0]], alpha=0.12, zorder=1)

    handles = []
    for nm in names:
        col, ls, lw = STYLE[nm]
        R = rho(ALL[nm], Z)
        cs = ax.contour(RE, IM, R, levels=[1.0], colors=[col],
                        linestyles=[ls], linewidths=lw, zorder=3)
        # proxy artist for a clean legend
        h, = ax.plot([], [], color=col, ls=ls, lw=lw,
                     label=_label(nm))
        handles.append(h)

    _frame(ax, re, im, title)
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _label(nm):
    im = intercept(ALL[nm], "imag")
    re = intercept(ALL[nm], "real")
    return f"{nm}  (i {im:.2f}, r {re:.2f})"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # intercept table -----------------------------------------------------
    print(f"{'scheme':<16}{'family':<7}{'imag-axis':>11}{'real-axis':>11}")
    print("-" * 45)
    for nm, m in ALL.items():
        print(f"{nm:<16}{m['kind']:<7}{intercept(m, 'imag'):>11.3f}"
              f"{intercept(m, 'real'):>11.3f}")
    print()

    # 1) everything overlaid ---------------------------------------------
    plot_outlines(
        list(ALL.keys()), "stability_all.png",
        "Absolute-stability regions -- all ETS schemes",
        re=(-7.5, 1.0), im=(-4.5, 4.5))

    # 2) classical single-step RK family ---------------------------------
    plot_outlines(
        ["RK3", "RK4", "RK5", "RK5_NYSTROM", "RK45_CASH_KARP", "RKF45",
         "RK4_RALSTON", "RK6"],
        "stability_rk_singlestep.png",
        "Absolute-stability regions -- single-step RK schemes",
        re=(-7.5, 1.0), im=(-4.5, 4.5))

    # 3) MSRK variants vs the RK4 reference (paper-style comparison) ------
    plot_outlines(
        ["RK4", "RK4_MSRK2_1", "RK4_MSRK2_2", "RK4_MSRK3"],
        "stability_msrk_vs_rk4.png",
        "MSRK variants vs RK4 reference",
        re=(-3.5, 0.6), im=(-3.2, 3.2), fill_ref="RK4")

    # 4) order-6 TSRK vs RK4/RK5/RK6 references --------------------------
    plot_outlines(
        ["RK4", "RK5", "RK6", "RK6_TSRK"],
        "stability_tsrk_vs_ref.png",
        "Order-6 TSRK vs single-step RK references",
        re=(-6.0, 0.8), im=(-4.0, 4.0), fill_ref="RK6_TSRK")


if __name__ == "__main__":
    main()
