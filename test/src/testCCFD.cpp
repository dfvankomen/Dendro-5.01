/**
 * testCCFD — correctness gate for the combined compact finite difference operators.
 *
 * Unlike testDerivConvergence (a reporting tool that always exits 0), this is
 * an assertion test: it exits non-zero on failure.
 *
 * Three checks:
 *
 *  1. Oracle match. scripts/ccfd_operators.py builds the same operators by an
 *     independent route (and cross-checks itself against the Sengupta N x N
 *     Schur form), so its matrices are a real reference rather than a snapshot
 *     of our own output. Its embed also differs structurally from ours: it
 *     places a standalone solve into the padded block AFTER solving, while the
 *     C++ embeds identity 2x2 blocks into the coupled system BEFORE solving.
 *     Agreement therefore tests the assembly rather than restating it.
 *     All four bflag variants are compared, since a bad 2x2-block embed only
 *     shows up when bflag != 0.
 *
 *  2. Convergence at bflag = all six bits, so the closure rows are exercised.
 *
 *  3. A deliberately degenerate closure pair is REJECTED. This guards the guard:
 *     a singular CCFD system returns plausible garbage rather than failing, so
 *     the rcond check in the builder is load-bearing and must itself be tested.
 *
 * Run:
 *     cmake --build build_avx2 --target testCCFD -j && ./build_avx2/testCCFD
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ccfd_golden.h"
#include "derivatives.h"
#include "derivatives/derivs_ccfd.h"
#include "derivatives/derivs_factory.h"
#include "derivatives/impl_ccfd.h"

using namespace dendroderivs;

// matches the golden header (pw = ele_order / 2)
static constexpr unsigned int ELEORDER = 6;
static constexpr unsigned int PW       = ELEORDER / 2;

// cond(A) ~ 1e6 for a healthy CCFD6 system, so ~1e-10 is the honest floor for
// an independent rebuild; 1e-9 leaves a little room without being permissive.
static constexpr double ORACLE_TOL     = 1.0e-9;

static int g_failures                  = 0;

static void report(const std::string &what, bool ok, const std::string &detail) {
    std::cout << (ok ? "  [ OK ] " : "  [FAIL] ") << what;
    if (!detail.empty()) std::cout << "  " << detail;
    std::cout << "\n";
    if (!ok) g_failures++;
}

// f(x) = sin(2*pi*x): one full period across the active block. This is
// deliberately the same function testDerivConvergence uses, and it matters --
// a smoother probe (e.g. exp(sin(3x+0.5))) reaches asymptopia much earlier and
// reports ~6.7 for f'' where this reports ~5.0, i.e. it would make this gate
// weaker than the project's own convergence test and hide a closure whose
// error constant is too large.
static double f_test(double x) { return std::sin(2.0 * M_PI * x); }
static double df_test(double x) {
    return 2.0 * M_PI * std::cos(2.0 * M_PI * x);
}
static double d2f_test(double x) {
    return -(2.0 * M_PI) * (2.0 * M_PI) * std::sin(2.0 * M_PI * x);
}

static std::unique_ptr<Derivs> make_ccfd6(unsigned int order) {
    auto &reg = (order == 1) ? get_first_order_registry()
                             : get_second_order_registry();
    auto it   = reg.find("CCFD6");
    if (it == reg.end()) return nullptr;
    return it->second(ELEORDER, "none", {}, {}, 0);
}

// A minimal *parameterized* CCFD creator, used only to exercise the
// coeffs-threading path (make_ccfd_coeffs / GenericCCFDDerivsWithCoeffs). c[0]
// is an additive knob on the A1 interior off-diagonals; at c[0] = 0 it is CCFD6
// exactly, so the plumbing is checked without registering a whole new scheme.
// A real tunable scheme's generated header would reference D_coeffs[k] across
// its coefficient section instead of this one perturbation.
static CCFDDiagonalEntries *ccfd6_param(const std::vector<double> &c) {
    CCFDDiagonalEntries *e = createCCFD6Diagonals();
    const double d         = c.empty() ? 0.0 : c[0];
    e->A1Interior.front() += d;  // symmetric off-diagonals, so parity holds
    e->A1Interior.back() += d;
    return e;
}

// ---------------------------------------------------------------------------
// 1. oracle match
// ---------------------------------------------------------------------------

static const std::vector<double> *variant_of(DerivMatrixStorage *st,
                                             const std::string &name) {
    if (name == "original") return &st->D_original;
    if (name == "left") return &st->D_left;
    if (name == "right") return &st->D_right;
    return &st->D_leftright;
}

static void check_oracle() {
    std::cout << "\n[1] oracle match vs scripts/ccfd_operators.py "
              << "(all four bflag variants)\n";

    for (unsigned int order = 1; order <= 2; order++) {
        auto d = make_ccfd6(order);
        if (!d) {
            report("CCFD6 registry lookup", false,
                   "not registered for order " + std::to_string(order));
            return;
        }
        for (unsigned int e = 0; e < CCFD_GOLDEN_COUNT; e++) {
            const auto &g = CCFD_GOLDEN[e];
            if (static_cast<unsigned int>(g.deriv) != order) continue;

            DerivMatrixStorage *st = nullptr;
            if (order == 1) {
                auto *mm = dynamic_cast<MatrixCompactDerivs<1> *>(d.get());
                if (!mm) {
                    report("dynamic_cast to MatrixCompactDerivs<1>", false, "");
                    return;
                }
                st = mm->get_storage_for_size(g.n);
            } else {
                auto *mm = dynamic_cast<MatrixCompactDerivs<2> *>(d.get());
                if (!mm) {
                    report("dynamic_cast to MatrixCompactDerivs<2>", false, "");
                    return;
                }
                st = mm->get_storage_for_size(g.n);
            }

            const std::vector<double> *D = variant_of(st, g.variant);
            const std::string label      = "D" + std::to_string(g.deriv) + " " +
                                      g.variant + " n=" + std::to_string(g.n);

            // init() deliberately skips the LEFTRIGHT variant for the smallest
            // block (n = 2*eleorder+1), on the grounds that a single smallest
            // block is never left-right bounded. It comes back all zeros, so
            // there is nothing to compare — flag it rather than pretend.
            const bool all_zero =
                std::all_of(D->begin(), D->end(),
                            [](double v) { return v == 0.0; });
            if (all_zero && std::string(g.variant) == "leftright") {
                std::cout << "  [skip] " << label
                          << "  (not built: init() skips LEFTRIGHT at the "
                             "smallest block size)\n";
                continue;
            }

            double max_diff = 0.0;
            for (unsigned int i = 0; i < g.n * g.n; i++) {
                max_diff = std::max(max_diff, std::fabs((*D)[i] - g.data[i]));
            }
            char buf[128];
            std::snprintf(buf, sizeof(buf), "max|C++ - python| = %.3e",
                          max_diff);
            report(label, max_diff <= ORACLE_TOL, buf);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. convergence
// ---------------------------------------------------------------------------

static double fit_order(const std::vector<double> &h,
                        const std::vector<double> &rmse, unsigned int n_fit) {
    const unsigned int n_pts = h.size();
    if (n_pts < 2 || n_fit < 2 || n_fit > n_pts) return std::nan("");
    const unsigned int start = n_pts - n_fit;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    unsigned int valid = 0;
    for (unsigned int i = start; i < n_pts; i++) {
        if (rmse[i] <= 0.0 || !std::isfinite(rmse[i])) continue;
        const double lx = std::log(h[i]);
        const double ly = std::log(rmse[i]);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
        valid++;
    }
    if (valid < 2) return std::nan("");
    const double denom = valid * sxx - sx * sx;
    if (denom == 0.0) return std::nan("");
    return (valid * sxy - sx * sy) / denom;
}

static void check_convergence() {
    std::cout << "\n[2] convergence, bflag = all six bits "
              << "(closures exercised), RMSE over the active region\n";

    const std::vector<unsigned int> sweep = {17, 21, 25, 33, 49};
    const unsigned int bflag =
        (1u << OCT_DIR_LEFT) | (1u << OCT_DIR_RIGHT) | (1u << OCT_DIR_DOWN) |
        (1u << OCT_DIR_UP) | (1u << OCT_DIR_BACK) | (1u << OCT_DIR_FRONT);

    for (unsigned int order = 1; order <= 2; order++) {
        auto d = make_ccfd6(order);
        if (!d) {
            report("CCFD6 registry lookup", false, "");
            return;
        }

        std::vector<double> hs, rmses;
        std::cout << "      order " << order << ":  ";
        for (unsigned int n : sweep) {
            // same geometry as testDerivConvergence: the ACTIVE region spans
            // [0,1], so h keys off n_active and x is measured from the first
            // active point.
            const unsigned int n_active = n - 2 * PW;
            const double h = 1.0 / static_cast<double>(n_active - 1);
            const unsigned int sz[3] = {n, n, n};
            const size_t tot         = static_cast<size_t>(n) * n * n;

            std::vector<double> u(tot), du(tot, 0.0);
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++)
                        u[i + n * (j + n * k)] =
                            f_test((static_cast<int>(i) - static_cast<int>(PW)) *
                                   h);

            d->set_maximum_block_size(tot);
            d->pre_create_for_size(n);
            d->do_grad_x(du.data(), u.data(), h, sz, bflag);

            // "interior" in testDerivConvergence's sense: active cells more
            // than PW from either active face. Because D is dense, a closure
            // with a large error constant still shows up here — which is the
            // whole point of measuring it.
            double s          = 0.0;
            unsigned long cnt = 0;
            for (unsigned int k = 2 * PW; k < n - 2 * PW; k++)
                for (unsigned int j = 2 * PW; j < n - 2 * PW; j++)
                    for (unsigned int i = 2 * PW; i < n - 2 * PW; i++) {
                        const double x =
                            (static_cast<int>(i) - static_cast<int>(PW)) * h;
                        const double t = (order == 1) ? df_test(x) : d2f_test(x);
                        const double e = du[i + n * (j + n * k)] - t;
                        s += e * e;
                        cnt++;
                    }
            const double rmse = std::sqrt(s / static_cast<double>(cnt));
            hs.push_back(h);
            rmses.push_back(rmse);
            std::printf("%8.2e ", rmse);
        }
        std::cout << "\n";

        const double fitted = fit_order(hs, rmses, 4);
        const bool monotone =
            std::is_sorted(rmses.rbegin(), rmses.rend()) &&
            rmses.front() > rmses.back();
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "fitted order = %.2f (formal 6), monotone = %s", fitted,
                      monotone ? "yes" : "no");
        // one-sided: overshoot is fine (preasymptotic, and CCFD's coupling
        // genuinely lifts f' above its row's formal order); falling short isn't.
        report("D" + std::to_string(order) + " convergence",
               std::isfinite(fitted) && (6.0 - fitted) <= 0.5 && monotone, buf);
    }
}

// ---------------------------------------------------------------------------
// 3. the guard itself
// ---------------------------------------------------------------------------

static void check_degenerate_rejected() {
    std::cout << "\n[3] a degenerate closure pair must be refused, not solved\n";

    // Take CCFD6 and make the two boundary rows linearly dependent on purpose,
    // by replacing eq1's closure with a scaled copy of eq2's. This is exactly
    // the failure the derivation falls into if both closures are derived from
    // the same term set at maximal order: eq2 comes out as 49/6 times eq1.
    // Scaling by 6/49 keeps eq1's g_0 normalized to 1, so the rows look
    // perfectly reasonable in isolation.
    std::unique_ptr<CCFDDiagonalEntries> bad(createCCFD6Diagonals());
    const double k = 6.0 / 49.0;
    bad->A1Boundary = bad->A2Boundary;
    bad->B1Boundary = bad->B2Boundary;
    bad->C1Boundary = bad->C2Boundary;
    for (auto &row : bad->A1Boundary)
        for (auto &v : row) v *= k;
    for (auto &row : bad->B1Boundary)
        for (auto &v : row) v *= k;
    for (auto &row : bad->C1Boundary)
        for (auto &v : row) v *= k;
    bad->A1BoundaryLower = bad->A1Boundary;
    bad->B1BoundaryLower = bad->B1Boundary;
    bad->C1BoundaryLower = bad->C1Boundary;

    bool threw = false;
    std::string msg;
    try {
        auto st = createCCFDMatrixSystemForSingleSize<1>(PW, 17, bad.get(),
                                                        false);
    } catch (const std::exception &e) {
        threw = true;
        msg   = e.what();
    }
    report("degenerate closure pair throws", threw,
           threw ? "(caught, as intended)"
                 : "NO THROW — the singular-system guard is not working, so a "
                   "bad scheme would silently produce garbage operators");

    // and the healthy scheme must still build, i.e. the guard isn't just
    // rejecting everything
    bool ok = true;
    try {
        std::unique_ptr<CCFDDiagonalEntries> good(createCCFD6Diagonals());
        auto st = createCCFDMatrixSystemForSingleSize<1>(PW, 17, good.get(),
                                                        false);
    } catch (const std::exception &e) {
        ok  = false;
        msg = e.what();
    }
    report("healthy CCFD6 still builds", ok, ok ? "" : msg);
}

// ---------------------------------------------------------------------------
// 4. the parameterized path
// ---------------------------------------------------------------------------

static double max_abs_diff(const std::vector<double> &a,
                           const std::vector<double> &b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static void check_parameterized() {
    std::cout << "\n[4] parameterized path "
              << "(make_ccfd_coeffs / GenericCCFDDerivsWithCoeffs)\n";
    const unsigned int N = 17;

    // fixed CCFD6 operator, for reference.
    auto fixed = make_ccfd6(1);
    auto *fx   = dynamic_cast<MatrixCompactDerivs<1> *>(fixed.get());
    if (!fx) {
        report("dynamic_cast fixed CCFD6", false, "");
        return;
    }
    const std::vector<double> &Dfix = fx->get_storage_for_size(N)->D_original;

    // baseline: coeffs = {0} must reproduce the fixed operator exactly, which
    // proves the coefficient function runs and its result flows into the build.
    GenericCCFDDerivsWithCoeffs<1> base(ccfd6_param, DerivType::D_CCFD6,
                                        "CCFD6_param", 1, ELEORDER, "none", {},
                                        {0.0});
    const std::vector<double> &Dbase = base.get_storage_for_size(N)->D_original;
    const double d_base              = max_abs_diff(Dfix, Dbase);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "max|param(0) - fixed| = %.3e", d_base);
    report("coeffs={0} reproduces fixed CCFD6", d_base <= ORACLE_TOL, buf);

    // and a nonzero coefficient must actually change the operator, i.e. the
    // vector is genuinely threaded through, not silently dropped.
    GenericCCFDDerivsWithCoeffs<1> tuned(ccfd6_param, DerivType::D_CCFD6,
                                         "CCFD6_param", 1, ELEORDER, "none", {},
                                         {0.05});
    const std::vector<double> &Dtuned = tuned.get_storage_for_size(N)->D_original;
    const double d_tuned              = max_abs_diff(Dfix, Dtuned);
    std::snprintf(buf, sizeof(buf), "max|param(0.05) - fixed| = %.3e", d_tuned);
    report("coeffs={0.05} changes the operator", d_tuned > 1.0e-6, buf);
}

int main(int argc, char **argv) {
    std::cout << "================================================\n"
              << "testCCFD — combined compact finite difference operators\n"
              << "  ele_order = " << ELEORDER << ", pw = " << PW << "\n"
              << "================================================\n";

    check_oracle();
    check_convergence();
    check_degenerate_rejected();
    check_parameterized();

    std::cout << "\n------------------------------------------------\n";
    if (g_failures == 0) {
        std::cout << "PASS — all CCFD checks green\n";
    } else {
        std::cout << "FAIL — " << g_failures << " check(s) failed\n";
    }
    return g_failures > 0 ? 1 : 0;
}
