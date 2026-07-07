// Order + stability test for the ETS time integrators. Pulls the actual
// tableaux from ts::get_rk_tableau / ts::get_msrk_tableau (single source of
// truth), so any coefficient regression fails the test.
//
// Single-step RK: tableau consistency, temporal order (fit over a dt sweep on
// two known IVPs), and linear stability boundaries (|R(z)| <= 1). Multistep RK:
// order via the real evolve_msrk recurrence, and stability via the companion-
// matrix spectral radius (checked against the arXiv:2603.05763 intercepts).
//
// Embedded pairs (RKF45, Cash-Karp) use their 5th-order rows, so expected 5.
// Run: ./testTimeIntegratorOrder   (exit 0 = all pass, 1 = failure)

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "rkTableau.h"

using ts::ETSType;

using RhsFn   = std::function<void(double, const std::vector<double>&,
                                 std::vector<double>&)>;
using ExactFn = std::function<std::vector<double>(double)>;

struct SchemeSpec {
    ETSType type;
    const char* name;
    int order;  // theoretical order of the weights used
};

struct Problem {
    const char* name;
    RhsFn rhs;
    ExactFn exact;
};

// one explicit RK step using an arbitrary Butcher tableau
static std::vector<double> rk_step(const RhsFn& f, unsigned int ns,
                                   const DendroScalar* c,
                                   const DendroScalar* b,
                                   const DendroScalar* a, double t,
                                   const std::vector<double>& y, double h) {
    const std::size_t m = y.size();
    std::vector<std::vector<double>> k(ns, std::vector<double>(m));
    std::vector<double> yi(m), dy(m);
    for (unsigned int i = 0; i < ns; i++) {
        for (std::size_t d = 0; d < m; d++) {
            double acc = 0.0;
            for (unsigned int j = 0; j < i; j++) acc += a[i * ns + j] * k[j][d];
            yi[d] = y[d] + h * acc;
        }
        f(t + c[i] * h, yi, dy);
        k[i] = dy;
    }
    std::vector<double> yn(m);
    for (std::size_t d = 0; d < m; d++) {
        double acc = 0.0;
        for (unsigned int i = 0; i < ns; i++) acc += b[i] * k[i][d];
        yn[d] = y[d] + h * acc;
    }
    return yn;
}

static double integrate_err(const Problem& p, unsigned int ns,
                            const DendroScalar* c, const DendroScalar* b,
                            const DendroScalar* a, double T, int N) {
    std::vector<double> y = p.exact(0.0);
    const double h        = T / N;
    double t              = 0.0;
    for (int s = 0; s < N; s++) {
        y = rk_step(p.rhs, ns, c, b, a, t, y, h);
        t += h;
    }
    std::vector<double> ye = p.exact(T);
    double e               = 0.0;
    for (std::size_t d = 0; d < y.size(); d++)
        e = std::max(e, std::fabs(y[d] - ye[d]));
    return e;
}

// shared h-refinement sweep: T fixed, N doubling each entry (h halves).
static const double SWEEP_T          = 1.0;
static const std::vector<int> NSWEEP = {5, 10, 20, 40, 80};

static std::vector<double> error_sweep(const Problem& p, unsigned int ns,
                                       const DendroScalar* c,
                                       const DendroScalar* b,
                                       const DendroScalar* a) {
    std::vector<double> errs;
    for (int N : NSWEEP)
        errs.push_back(integrate_err(p, ns, c, b, a, SWEEP_T, N));
    return errs;
}

// max pairwise convergence rate over an h-halving sweep, restricted to the
// asymptotic band (errors above the roundoff floor, below O(1)).
static double observed_order(const Problem& p, unsigned int ns,
                             const DendroScalar* c, const DendroScalar* b,
                             const DendroScalar* a) {
    std::vector<double> errs = error_sweep(p, ns, c, b, a);

    double best = 0.0;
    for (std::size_t i = 0; i + 1 < errs.size(); i++) {
        // N doubles each step -> h halves; rate = log2(e_i / e_{i+1})
        if (errs[i] < 1e-11 || errs[i] > 1e-1) continue;
        if (errs[i + 1] < 1e-13) continue;
        double rate = std::log2(errs[i] / errs[i + 1]);
        if (rate > best) best = rate;
    }
    return best;
}

// ---- linear stability: apply the scheme to y' = z*y (one step, y0=1). The
// result is the amplification factor R(z); the method is stable where |R|<=1.
using cplx = std::complex<double>;
static cplx amp_factor(unsigned int ns, const DendroScalar* b,
                       const DendroScalar* a, cplx z) {
    std::vector<cplx> k(ns);
    for (unsigned int i = 0; i < ns; i++) {
        cplx stage_val(1.0, 0.0);  // y0 + sum a_ij*kappa_j, y0 = 1
        for (unsigned int j = 0; j < i; j++)
            stage_val += a[i * ns + j] * k[j];
        k[i] = z * stage_val;
    }
    cplx R(1.0, 0.0);
    for (unsigned int i = 0; i < ns; i++) R += b[i] * k[i];
    return R;
}

// Largest interval [0, beta] on the negative-real (axis=0) or positive-imag
// (axis=1) axis over which |R| <= 1, found by scanning to the first crossing
// then bisecting. Returns 0 if the scheme is unstable arbitrarily close to 0.
static double stability_boundary(unsigned int ns, const DendroScalar* b,
                                 const DendroScalar* a, int axis) {
    auto zof = [axis](double x) {
        return axis == 0 ? cplx(-x, 0.0) : cplx(0.0, x);
    };
    const double step = 1e-3;
    double lo = 0.0, hi = 0.0;
    for (double x = step; x < 16.0; x += step) {
        if (std::abs(amp_factor(ns, b, a, zof(x))) > 1.0 + 1e-12) {
            lo = x - step;
            hi = x;
            break;
        }
    }
    if (hi == 0.0) return 0.0;
    for (int it = 0; it < 60; it++) {
        double mid = 0.5 * (lo + hi);
        if (std::abs(amp_factor(ns, b, a, zof(mid))) <= 1.0)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

static bool check_consistency(const char* name, unsigned int ns,
                              const DendroScalar* c, const DendroScalar* b,
                              const DendroScalar* a) {
    bool ok         = true;
    const double tol = 1e-12;
    for (unsigned int i = 0; i < ns; i++) {
        double rs = 0.0;
        for (unsigned int j = 0; j < ns; j++) rs += a[i * ns + j];
        if (std::fabs(rs - c[i]) > tol) {
            std::printf("    [%s] row-sum FAIL stage %u: sum(a)=%.15g c=%.15g\n",
                        name, i, rs, c[i]);
            ok = false;
        }
    }
    double bs = 0.0;
    for (unsigned int i = 0; i < ns; i++) bs += b[i];
    if (std::fabs(bs - 1.0) > tol) {
        std::printf("    [%s] weight-sum FAIL: sum(b)=%.15g\n", name, bs);
        ok = false;
    }
    return ok;
}

// ---- Multistep RK (MSRK) coverage ----
// MSRK reuses past-step evals, so we can't use the single-step helpers above.
// These replicate the exact evolve_msrk() recurrence (RK4 bootstrap + history
// aging) with coefficients from ts::get_msrk_tableau().

struct MsrkSpec {
    ETSType type;
    const char* name;
    int order;
    double imag_ref;  // published imag-axis stability intercept (arXiv:2603.05763)
};

// evaluate the RHS once: f(t, y).
static std::vector<double> f_eval(const RhsFn& f, double t,
                                  const std::vector<double>& y) {
    std::vector<double> dy(y.size());
    f(t, y, dy);
    return dy;
}

// one standard RK4 step (bootstrap); also returns the base eval f(t_n, y_n).
static std::vector<double> rk4_boot_step(const RhsFn& f, double t,
                                         const std::vector<double>& y, double h,
                                         std::vector<double>& base_out) {
    static const double rc[4]  = {0.0, 0.5, 0.5, 1.0};
    static const double rb[4]  = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    static const double ra[16] = {0, 0,   0, 0, 0.5, 0, 0, 0,
                                  0, 0.5, 0, 0, 0,   0, 1.0, 0};
    base_out = f_eval(f, t, y);  // base = f(t_n, y_n)
    return rk_step(f, 4, rc, rb, ra, t, y, h);
}

// integrate an IVP with an MSRK variant: RK4 bootstrap for the first H steps
// (saving base evals into history), then the multistep recurrence.
static double msrk_integrate(const Problem& p, const DendroScalar* a,
                             const DendroScalar* b, const DendroScalar* c,
                             unsigned int F, unsigned int H, double T, int N) {
    std::vector<double> y = p.exact(0.0);
    const double h        = T / N;
    double t              = 0.0;
    std::vector<std::vector<double>> hist(H);

    for (unsigned int n = 0; n < H && (int)n < N; n++) {
        std::vector<double> base;
        y = rk4_boot_step(p.rhs, t, y, h, base);
        hist[H == 1 ? 0 : n] = base;  // H==1 -> slot0; H==2 -> slot n
        t += h;
    }

    for (int n = (int)H; n < N; n++) {
        std::vector<std::vector<double>> k(4);
        for (unsigned int s = 0; s < F; s++) k[s] = hist[s];
        for (unsigned int stage = F; stage < 4; stage++) {
            std::vector<double> tmp(y.size());
            for (std::size_t d = 0; d < y.size(); d++) {
                double acc = 0.0;
                for (unsigned int pp = 0; pp < stage; pp++)
                    acc += a[stage * 4 + pp] * k[pp][d];
                tmp[d] = y[d] + h * acc;
            }
            k[stage] = f_eval(p.rhs, t + c[stage] * h, tmp);
        }
        for (std::size_t d = 0; d < y.size(); d++) {
            double acc = 0.0;
            for (unsigned int i = 0; i < 4; i++) acc += b[i] * k[i][d];
            y[d] += h * acc;
        }
        std::vector<double> base = k[F];  // f(t_n, y_n)
        if (H == 1)
            hist[0] = base;
        else {
            hist[0] = hist[1];
            hist[1] = base;
        }
        t += h;
    }
    std::vector<double> ye = p.exact(T);
    double e               = 0.0;
    for (std::size_t d = 0; d < y.size(); d++)
        e = std::max(e, std::fabs(y[d] - ye[d]));
    return e;
}

// ---- MSRK linear stability: apply to y' = (z/h) y. One step is a linear map
// on the state [y_n, g_0..g_{H-1}] (g = h*k, the aged history evals). The method
// is stable at z where the spectral radius of that (H+1)-dim map is <= 1.
static std::vector<cplx> msrk_lin_step(const DendroScalar* a,
                                       const DendroScalar* b, unsigned int F,
                                       unsigned int H, cplx z,
                                       const std::vector<cplx>& s) {
    cplx y = s[0];
    std::vector<cplx> g(4, cplx(0.0, 0.0));
    for (unsigned int i = 0; i < F; i++) g[i] = s[1 + i];  // history stages
    for (unsigned int stage = F; stage < 4; stage++) {
        cplx Y = y;
        for (unsigned int pp = 0; pp < stage; pp++)
            Y += a[stage * 4 + pp] * g[pp];
        g[stage] = z * Y;
    }
    cplx yn = y;
    for (unsigned int i = 0; i < 4; i++) yn += b[i] * g[i];
    std::vector<cplx> ns(H + 1);
    ns[0] = yn;
    if (H == 1)
        ns[1] = g[F];
    else {
        ns[1] = s[2];   // hist[0] <- old hist[1]
        ns[2] = g[F];   // hist[1] <- fresh base eval f(t_n, y_n)
    }
    return ns;
}

// max |root| of the monic cubic  x^3 + A x^2 + B x + C  (complex, via Cardano).
static double cubic_maxmod(cplx A, cplx B, cplx C) {
    const cplx p    = B - A * A / 3.0;
    const cplx q    = 2.0 * A * A * A / 27.0 - A * B / 3.0 + C;
    const cplx half = q * 0.5;
    const cplx D    = std::sqrt(half * half + p * p * p / 27.0);
    cplx u3         = -half + D;
    if (std::abs(u3) < 1e-300) u3 = -half - D;
    cplx u             = std::pow(u3, cplx(1.0 / 3.0, 0.0));
    const cplx w1(-0.5, std::sqrt(3.0) / 2.0), w2(-0.5, -std::sqrt(3.0) / 2.0);
    cplx t[3];
    if (std::abs(u) < 1e-300) {
        t[0] = t[1] = t[2] = cplx(0.0, 0.0);  // triple root at the shift
    } else {
        const cplx us[3] = {u, u * w1, u * w2};
        for (int k = 0; k < 3; k++) t[k] = us[k] - p / (3.0 * us[k]);
    }
    double m = 0.0;
    for (int k = 0; k < 3; k++) m = std::max(m, std::abs(t[k] - A / 3.0));
    return m;
}

// exact spectral radius of the (H+1)-dim MSRK amplification map at z.
static double msrk_specrad(const DendroScalar* a, const DendroScalar* b,
                           unsigned int F, unsigned int H, cplx z) {
    const unsigned int dim = H + 1;
    std::vector<std::vector<cplx>> M(dim, std::vector<cplx>(dim));
    for (unsigned int j = 0; j < dim; j++) {
        std::vector<cplx> e(dim, cplx(0.0, 0.0));
        e[j]                   = cplx(1.0, 0.0);
        std::vector<cplx> col  = msrk_lin_step(a, b, F, H, z, e);
        for (unsigned int i = 0; i < dim; i++) M[i][j] = col[i];
    }
    if (dim == 2) {
        cplx tr   = M[0][0] + M[1][1];
        cplx det  = M[0][0] * M[1][1] - M[0][1] * M[1][0];
        cplx disc = std::sqrt(tr * tr - 4.0 * det);
        return std::max(std::abs((tr + disc) * 0.5),
                        std::abs((tr - disc) * 0.5));
    }
    // dim == 3: char poly  x^3 - c2 x^2 + c1 x - c0
    cplx c2 = M[0][0] + M[1][1] + M[2][2];
    cplx c1 = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) +
              (M[0][0] * M[2][2] - M[0][2] * M[2][0]) +
              (M[1][1] * M[2][2] - M[1][2] * M[2][1]);
    cplx c0 = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
              M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
              M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    return cubic_maxmod(-c2, c1, -c0);
}

static double msrk_stability_boundary(const DendroScalar* a,
                                      const DendroScalar* b, unsigned int F,
                                      unsigned int H, int axis) {
    auto zof = [axis](double x) {
        return axis == 0 ? cplx(-x, 0.0) : cplx(0.0, x);
    };
    const double step = 2e-3;
    double lo = 0.0, hi = 0.0;
    for (double x = step; x < 6.0; x += step) {
        if (msrk_specrad(a, b, F, H, zof(x)) > 1.0 + 1e-9) {
            lo = x - step;
            hi = x;
            break;
        }
    }
    if (hi == 0.0) return 0.0;
    for (int it = 0; it < 45; it++) {
        double mid = 0.5 * (lo + hi);
        if (msrk_specrad(a, b, F, H, zof(mid)) <= 1.0 + 1e-9)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

int main() {
    const std::vector<SchemeSpec> schemes = {
        {ETSType::RK3, "RK3", 3},
        {ETSType::RK4, "RK4", 4},
        {ETSType::RK4_RALSTON, "RK4_RALSTON", 4},
        {ETSType::RK5, "RK5", 5},
        {ETSType::RK5_NYSTROM, "RK5_NYSTROM", 5},
        {ETSType::RK45_CASH_KARP, "RK45_CASH_KARP", 5},
        {ETSType::RKF45, "RKF45", 5},
        {ETSType::RK6, "RK6", 6},
    };

    const std::vector<Problem> problems = {
        {"nonlinear y'=-y^2",
         [](double, const std::vector<double>& y, std::vector<double>& dy) {
             dy[0] = -y[0] * y[0];
         },
         [](double t) { return std::vector<double>{1.0 / (1.0 + t)}; }},
        {"oscillator y1'=y2,y2'=-y1",
         [](double, const std::vector<double>& y, std::vector<double>& dy) {
             dy[0] = y[1];
             dy[1] = -y[0];
         },
         [](double t) {
             return std::vector<double>{std::cos(t), -std::sin(t)};
         }},
    };

    // ---- raw per-step error tables (error at final time vs dt) ----
    std::printf("\n=== raw error tables (max |y - y_exact| at t=%.1f) ===\n",
                SWEEP_T);
    for (const auto& s : schemes) {
        unsigned int ns = 0;
        const DendroScalar *c = nullptr, *b = nullptr, *a = nullptr;
        if (ts::get_rk_tableau(s.type, ns, c, b, a) != 0) continue;
        std::printf("\n%s  (%u stages, theory order %d)\n", s.name, ns,
                    s.order);
        for (std::size_t pi = 0; pi < problems.size(); pi++) {
            std::vector<double> e = error_sweep(problems[pi], ns, c, b, a);
            std::printf("  %-26s %8s %12s %8s\n", problems[pi].name, "dt",
                        "err", "rate");
            for (std::size_t i = 0; i < NSWEEP.size(); i++) {
                double dt = SWEEP_T / NSWEEP[i];
                if (i == 0)
                    std::printf("  %-26s %8.5f %12.4e %8s\n", "", dt, e[i], "-");
                else
                    std::printf("  %-26s %8.5f %12.4e %8.2f\n", "", dt, e[i],
                                std::log2(e[i - 1] / e[i]));
            }
        }
    }

    std::printf(
        "\n=== ETS Runge-Kutta temporal convergence test ===\n"
        "%-16s %8s %14s %14s %8s\n",
        "scheme", "theory", "order(P1)", "order(P2)", "result");

    int failures = 0;
    for (const auto& s : schemes) {
        unsigned int ns = 0;
        const DendroScalar *c = nullptr, *b = nullptr, *a = nullptr;
        if (ts::get_rk_tableau(s.type, ns, c, b, a) != 0) {
            std::printf("%-16s   get_rk_tableau FAILED\n", s.name);
            failures++;
            continue;
        }

        bool cons = check_consistency(s.name, ns, c, b, a);

        double o1  = observed_order(problems[0], ns, c, b, a);
        double o2  = observed_order(problems[1], ns, c, b, a);
        double obs = std::max(o1, o2);

        // order tolerance: observed must reach within 0.4 of theory
        bool order_ok = obs >= (s.order - 0.4);
        bool pass     = cons && order_ok;
        if (!pass) failures++;

        std::printf("%-16s %8d %14.3f %14.3f %8s\n", s.name, s.order, o1, o2,
                    pass ? "PASS" : "FAIL");
    }

    // ---- linear stability regions ----
    // real-axis boundary -> max stable dt for dissipative/real-eigenvalue terms
    // imag-axis boundary -> CFL limit for advection/wave (pure-imaginary) terms
    std::printf(
        "\n=== linear stability boundaries (|R(z)| <= 1) ===\n"
        "%-16s %8s %12s %12s\n",
        "scheme", "stages", "real-axis", "imag-axis");
    for (const auto& s : schemes) {
        unsigned int ns = 0;
        const DendroScalar *c = nullptr, *b = nullptr, *a = nullptr;
        if (ts::get_rk_tableau(s.type, ns, c, b, a) != 0) continue;
        double rb = stability_boundary(ns, b, a, 0);
        double ib = stability_boundary(ns, b, a, 1);
        std::printf("%-16s %8u %12.4f %12.4f\n", s.name, ns, rb, ib);

        // correctness anchors: classical RK3/RK4 have textbook boundaries.
        // (all p-stage order-p methods share R(z), so RK4 == RK4_RALSTON.)
        auto near = [](double x, double t) { return std::fabs(x - t) < 0.02; };
        if (s.type == ETSType::RK3 && !(near(rb, 2.5127) && near(ib, 1.7321))) {
            std::printf("    [RK3] stability boundary mismatch vs textbook\n");
            failures++;
        }
        if (s.type == ETSType::RK4 && !(near(rb, 2.7853) && near(ib, 2.8284))) {
            std::printf("    [RK4] stability boundary mismatch vs textbook\n");
            failures++;
        }
    }
    std::printf(
        "  note: imag-axis reach is the CFL-relevant number for hyperbolic /\n"
        "  wave systems; the high-order schemes are near-zero there and need\n"
        "  dissipation (e.g. Kreiss-Oliger) to be usable on advective terms.\n");

    // ================= Multistep RK (MSRK) variants =================
    const std::vector<MsrkSpec> msrk = {
        {ETSType::RK4_MSRK2_1, "RK4_MSRK2_1", 4, 2.54},
        {ETSType::RK4_MSRK2_2, "RK4_MSRK2_2", 4, 2.46},
        {ETSType::RK4_MSRK3, "RK4_MSRK3", 4, 1.31},
    };
    // fresh RHS evals per step = 4 - first_fresh_stage (the comm-saving metric).
    std::printf(
        "\n=== multistep RK (MSRK) — order + stability ===\n"
        "reuses history to cut fresh RHS evals/step (=> fewer ghost syncs)\n"
        "%-14s %6s %8s %10s %10s %10s %10s\n",
        "variant", "theory", "order", "real-axis", "imag-axis", "imag-ref",
        "fresh/step");
    for (const auto& m : msrk) {
        DendroScalar a[16], b[4], c[4];
        unsigned int F = 0, H = 0;
        if (ts::get_msrk_tableau(m.type, a, b, c, F, H) != 0) {
            std::printf("%-14s   get_msrk_tableau FAILED\n", m.name);
            failures++;
            continue;
        }

        // observed order = best asymptotic pairwise rate over the sweep
        double obs = 0.0;
        for (const auto& prob : problems) {
            std::vector<double> e;
            for (int N : NSWEEP)
                e.push_back(msrk_integrate(prob, a, b, c, F, H, SWEEP_T, N));
            for (std::size_t i = 0; i + 1 < e.size(); i++) {
                if (e[i] < 1e-11 || e[i] > 1e-1 || e[i + 1] < 1e-13) continue;
                obs = std::max(obs, std::log2(e[i] / e[i + 1]));
            }
        }
        double rb = msrk_stability_boundary(a, b, F, H, 0);
        double ib = msrk_stability_boundary(a, b, F, H, 1);

        bool order_ok = obs >= (m.order - 0.4);
        bool stab_ok  = std::fabs(ib - m.imag_ref) < 0.02;  // vs published value
        if (!order_ok || !stab_ok) failures++;

        std::printf("%-14s %6d %8.3f %10.4f %10.4f %10.2f %10u   %s\n", m.name,
                    m.order, obs, rb, ib, m.imag_ref, 4 - F,
                    (order_ok && stab_ok) ? "PASS" : "FAIL");
    }
    std::printf(
        "  imag-axis intercepts validated against arXiv:2603.05763; note they\n"
        "  far exceed the single-step high-order schemes above (RK6=0.079),\n"
        "  which is why 4th-order MSRK is the practical comm-reduction path.\n");

    std::printf("\n%s (%d check%s failed)\n\n",
                failures == 0 ? "ALL CHECKS PASS" : "FAILURES DETECTED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
