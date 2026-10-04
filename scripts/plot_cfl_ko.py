#!/usr/bin/env python3
"""Figures for the RK stability / CFL / error-budget story.

  fig1_mechanism.png  WHY dissipation reverses the scheme ranking. Stability
                      regions for RK4 / RK6 / MSRK2_1 with the semi-discrete
                      eigenvalue cloud overlaid at ONE fixed timestep, with and
                      without KO. At nu = 0.6, turning KO on brings RK6 inside
                      its region and pushes MSRK2_1 outside its own.
  fig2_maxcfl.png     WHAT it costs: max BSSN_CFL_FACTOR per scheme against the
                      0.25 every shipped par uses.
  fig3_error_budget.png  WHERE the error is: measured temporal error vs dt for
                      all twelve schemes, against the spatial-error floor of the
                      grid they were measured on. Below that floor the time
                      integrator's accuracy is invisible.

Run:  python3 scripts/plot_cfl_ko.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfl_with_ko as C  # noqa: E402
import modified_wavenumber as MW  # noqa: E402
import stability_regions as S  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfl_ko_plots")
DERIV, NKO, CSPEED, SIGMA = MW.SCHEME_COEFFS["E6"], 3, np.sqrt(2.0), 0.4
R = SIGMA / CSPEED
CFL_OP = 0.25

# max CFL number, 3D/iso, r = sigma/c, from cfl_with_ko.py.
# Not stability_regions._label(): its imaginary intercepts for RK6, Cash-Karp,
# RKF45 and Nystrom are tolerance artifacts -- those intervals are zero.
NAME = {"RK4": "RK4", "RK4_RALSTON": "RK4 Ralston",
        "RK45_CASH_KARP": "RK45 Cash-Karp", "RK5": "RK5 (Butcher)",
        "RKF45": "RKF45", "RK4_MSRK2_2": "RK4-MSRK2(2)",
        "RK5_NYSTROM": "RK5 Nystrom", "RK6": "RK6 (Luther)",
        "RK3": "RK3", "RK4_MSRK2_1": "RK4-MSRK2(1)",
        "RK6_TSRK": "RK6-TSRK", "RK4_MSRK3": "RK4-MSRK3"}

NU = {"RK4": 1.0695, "RK4_RALSTON": 1.0695, "RK45_CASH_KARP": 1.0037,
      "RK5": 0.9327, "RKF45": 0.8908, "RK4_MSRK2_2": 0.8204,
      "RK5_NYSTROM": 0.8185, "RK6": 0.7816, "RK3": 0.7579,
      "RK4_MSRK2_1": 0.6642, "RK6_TSRK": 0.5479, "RK4_MSRK3": 0.4569}

# Measured by testMeshTimeIntegrator's adv arm; spatial floor of that grid 1e-3.
DT0 = 5.0e-2
ERRS = {
    "RK3":            [3.11e-03, 3.96e-04, 5.00e-05, 6.27e-06],
    "RK4":            [2.06e-04, 1.25e-05, 7.63e-07, 4.71e-08],
    "RK4_RALSTON":    [2.06e-04, 1.25e-05, 7.63e-07, 4.71e-08],
    "RK4_MSRK2_1":    [5.34e-04, 3.23e-05, 2.00e-06, 1.24e-07],
    "RK4_MSRK2_2":    [5.00e-04, 3.08e-05, 1.90e-06, 1.18e-07],
    "RK4_MSRK3":      [1.68e-03, 1.25e-04, 8.14e-06, 5.13e-07],
    "RK5":            [1.90e-06, 4.56e-08, 1.50e-09, 4.69e-11],
    "RK45_CASH_KARP": [1.44e-06, 4.02e-08, 1.21e-09, 3.73e-11],
    "RKF45":          [7.66e-06, 2.49e-07, 7.83e-09, 2.44e-10],
    "RK5_NYSTROM":    [1.15e-05, 3.80e-07, 1.20e-08, 3.73e-10],
    "RK6":            [2.06e-06, 3.41e-08, 5.71e-10, 9.10e-12],
    "RK6_TSRK":       [7.48e-07, 8.43e-09, 1.21e-10, 1.73e-12],
}
SPATIAL_FLOOR = 1.0e-3
FAMILY = {"RK4_MSRK2_1": "msrk", "RK4_MSRK2_2": "msrk", "RK4_MSRK3": "msrk",
          "RK6_TSRK": "tsrk"}
FCOL = {"single": "#4C72B0", "msrk": "#C44E52", "tsrk": "#55A868"}


def fam(nm):
    return FAMILY.get(nm, "single")


def fig1_mechanism():
    """Same dt, KO off vs on: RK6 comes in, MSRK2_1 goes out."""
    nu = 0.72   # between MSRK2_1 (0.664) and RK6 (0.782), so the claim holds
    shown = ["RK4", "RK6", "RK4_MSRK2_1"]
    col = {"RK4": "#333333", "RK6": "#4C72B0", "RK4_MSRK2_1": "#C44E52"}
    lab = NAME
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), sharex=True, sharey=True)
    re, im, n = (-3.3, 0.7), (-3.0, 3.0), 520
    RE, IM = np.meshgrid(np.linspace(*re, n), np.linspace(*im, n))
    Z = RE + 1j * IM

    for ax, r, ttl in [(axes[0], 0.0, "KO off  ($\\sigma = 0$)"),
                       (axes[1], R, f"KO on  ($\\sigma = {SIGMA}$)")]:
        for nm in shown:
            Rr = S.rho(S.ALL[nm], Z)
            ax.contourf(RE, IM, Rr, levels=[0.0, 1.0], colors=[col[nm]],
                        alpha=0.13, zorder=2)
            ax.contour(RE, IM, Rr, levels=[1.0], colors=[col[nm]],
                       linewidths=2.0, zorder=3)
        L = C.locus(DERIV, NKO, r, 3, "iso", 21)
        pts = nu * L
        for sgn in (1, -1):
            ax.scatter(pts.real, sgn * pts.imag, s=2.0, c="#E8A33D",
                       alpha=0.7, zorder=5, rasterized=True,
                       edgecolors="none")
        ax.axhline(0, color="#999999", lw=0.6, zorder=1)
        ax.axvline(0, color="#999999", lw=0.6, zorder=1)
        ax.set_xlim(*re)
        ax.set_ylim(*im)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel(r"$\Re(\lambda\,\Delta t)$")
    axes[0].set_ylabel(r"$\Im(\lambda\,\Delta t)$")

    hs = [plt.Line2D([], [], color=col[nm], lw=2.0, label=lab[nm])
          for nm in shown]
    hs.append(plt.Line2D([], [], marker="o", ls="", color="#E8A33D", ms=5,
                         label=f"spectrum at $\\nu={nu}$"))
    axes[1].legend(handles=hs, loc="upper left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_mechanism.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"  wrote {p}")


def fig2_maxcfl():
    order = sorted(NU, key=lambda k: NU[k])
    vals = [NU[k] / CSPEED for k in order]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.barh(range(len(order)), vals,
            color=[FCOL[fam(k)] for k in order], alpha=0.88, zorder=3)
    ax.axvline(CFL_OP, color="#333333", ls="--", lw=1.6, zorder=4)
    ax.text(CFL_OP + 0.012, -0.55, "shipped pars: 0.25", fontsize=9,
            rotation=90, va="bottom", color="#333333")
    for i, (k, v) in enumerate(zip(order, vals)):
        ax.text(v + 0.012, i, f"{v:.3f}   ({v/CFL_OP:.2f}×)",
                va="center", fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[k] for k in order], fontsize=9)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("max BSSN_CFL_FACTOR   "
                  r"($\nu_{max}/c$,  $c=\sqrt{2}$,  KO $\sigma=0.4$)")
    hs = [plt.Line2D([], [], color=FCOL[f], lw=7, label=lab) for f, lab in
          [("single", "single-step RK"), ("msrk", "multistep (MSRK)"),
           ("tsrk", "TSRK")]]
    ax.legend(handles=hs, loc="lower right", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_maxcfl.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"  wrote {p}")


def fig3_error_budget():
    dts = np.array([DT0 / (2 ** k) for k in range(4)])
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.axhspan(SPATIAL_FLOOR, 1e0, color="#BBBBBB", alpha=0.3, zorder=0)
    ax.axhline(SPATIAL_FLOOR, color="#555555", lw=1.4, ls="--", zorder=2)
    ax.text(dts[0] * 0.99, SPATIAL_FLOOR * 1.7,
            "BL6 spatial error, this grid",
            fontsize=8.8, color="#333333", ha="left")
    styles = ["-", "--", ":", "-."]
    for i, (nm, e) in enumerate(sorted(ERRS.items(), key=lambda kv: -kv[1][0])):
        ax.loglog(dts, e, "o", ls=styles[i % 4], ms=3.4, lw=1.3,
                  color=FCOL[fam(nm)], alpha=0.9, zorder=3, label=NAME[nm])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              framealpha=0.95, title="coarsest error, high to low",
              title_fontsize=8)
    ax.set_xlabel(r"$\Delta t$   (shared across schemes)")
    ax.set_ylabel(r"temporal error  (vs fine-$\Delta t$ reference, same mesh)")
    ax.invert_xaxis()
    ax.grid(alpha=0.25, which="both", zorder=0)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_error_budget.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("writing figures:")
    fig1_mechanism()
    fig2_maxcfl()
    fig3_error_budget()
