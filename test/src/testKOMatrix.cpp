// Gate for the matrix-form KO dissipation filters (KO{2,4,6,8}Matrix): for
// every order the padding width supports, every bflag variant and block sizes
// n = 13 / 17 / 21, the matrix path must agree with the explicit stencil path
// to roundoff (both the scalar-coefficient and the per-point CAKO entry) and a
// clone must reproduce its source exactly. Timings are printed for information
// only — a shared or noisy node must not turn an accuracy gate red.
// Exit code 0 = pass.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "derivatives.h"

using namespace dendroderivs;

static const double REL_TOL = 1e-13;  // roundoff: same products, GEMM order

template <typename Fn>
static double time_ns(Fn &&fn, unsigned int it) {
    for (unsigned int i = 0; i < 100; i++) fn();
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < it; i++) fn();
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - t0)
               .count() /
           it;
}

int main() {
    const unsigned int L = 1u << OCT_DIR_LEFT, R = 1u << OCT_DIR_RIGHT,
                       D = 1u << OCT_DIR_DOWN, U = 1u << OCT_DIR_UP,
                       B = 1u << OCT_DIR_BACK, F = 1u << OCT_DIR_FRONT;
    const std::vector<unsigned int> bflags = {0, L, R, L | R, D, U, D | U, B, F,
                                             B | F, L | D | B,
                                             L | R | D | U | B | F};
    unsigned long checked = 0, bad = 0;

    for (const std::string order : {"KO2", "KO4", "KO6", "KO8"}) {
        for (unsigned int eo : {6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            const double dx = 0.05, dy = 0.07, dz = 0.11;

            std::unique_ptr<DendroDerivatives> ds, dm;
            try {
                ds = std::make_unique<DendroDerivatives>(
                    "JTT6", "JTT6", eo, std::vector<double>(),
                    std::vector<double>(), 0u, 0u, "none", "none",
                    std::vector<double>(), std::vector<double>(), order);
                dm = std::make_unique<DendroDerivatives>(
                    "JTT6", "JTT6", eo, std::vector<double>(),
                    std::vector<double>(), 0u, 0u, "none", "none",
                    std::vector<double>(), std::vector<double>(),
                    order + "Matrix");
            } catch (const std::exception &e) {
                std::printf("  [skip] %s at eleorder %u: %s\n", order.c_str(),
                            eo, e.what());
                continue;
            }
            ds->set_maximum_block_size(tot);
            dm->set_maximum_block_size(tot);
            DendroDerivatives dc(*dm);  // clone, as the per-thread pool does
            dc.set_maximum_block_size(tot);

            std::vector<double> u(tot), coeff(tot), base(tot), rs(tot), rm(tot),
                rc(tot), wx(tot), wy(tot), wz(tot);
            for (size_t i = 0; i < tot; i++) {
                u[i]     = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
                coeff[i] = 0.1 + 0.05 * std::sin(0.2 * i);
                base[i]  = std::cos(0.1 * i);
            }

            double worst = 0.0;
            for (unsigned int bf : bflags) {
                // per-point (CAKO) entry
                rs = base; rm = base; rc = base;
                ds->filter_cako(u.data(), rs.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                dm->filter_cako(u.data(), rm.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                dc.filter_cako(u.data(), rc.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                double md = 0.0, scale = 0.0, mdc = 0.0;
                for (unsigned int k = pw; k < n - pw; k++)
                    for (unsigned int j = pw; j < n - pw; j++)
                        for (unsigned int i = pw; i < n - pw; i++) {
                            const size_t p = i + n * (j + n * k);
                            md    = std::max(md, std::fabs(rm[p] - rs[p]));
                            scale = std::max(scale, std::fabs(rs[p] - base[p]));
                            mdc   = std::max(mdc, std::fabs(rc[p] - rm[p]));
                        }
                checked += 2;
                const double rel = scale > 0 ? md / scale : md;
                worst = std::max(worst, rel);
                if (rel > REL_TOL) { bad++; std::printf("  MISMATCH %sMatrix n=%u bflag=%u cako rel=%g\n", order.c_str(), n, bf, rel); }
                if (mdc != 0.0) { bad++; std::printf("  MISMATCH %sMatrix n=%u bflag=%u clone differs by %g\n", order.c_str(), n, bf, mdc); }
                // uniform coefficient: ko_accumulate (three beta = 1 GEMMs into rhs)
                // against the stencil's uniform-sigma filter and the matrix filter_cako
                std::vector<double> ru(tot, 0.1), ra(base), rsu(base), rmu(base);
                ds->filter(u.data(), rsu.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 0.1, sz, bf);
                dm->filter_cako(u.data(), rmu.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, ru.data(), sz, bf);
                const bool did = dm->ko_accumulate(ra.data(), u.data(), 0.1, dx, dy, dz, sz, bf);
                double mda = 0.0, mdm = 0.0, sca = 0.0;
                for (unsigned int k = pw; k < n - pw; k++)
                    for (unsigned int j = pw; j < n - pw; j++)
                        for (unsigned int i = pw; i < n - pw; i++) {
                            const size_t p = i + n * (j + n * k);
                            mda = std::max(mda, std::fabs(ra[p] - rsu[p]));
                            mdm = std::max(mdm, std::fabs(ra[p] - rmu[p]));
                            sca = std::max(sca, std::fabs(rsu[p] - base[p]));
                        }
                checked += 2;
                const double rela = sca > 0 ? mda / sca : mda, relm = sca > 0 ? mdm / sca : mdm;
                worst = std::max(worst, rela);
                if (!did) { bad++; std::printf("  MISMATCH %sMatrix n=%u bflag=%u ko_accumulate declined\n", order.c_str(), n, bf); }
                if (rela > REL_TOL) { bad++; std::printf("  MISMATCH %sMatrix n=%u bflag=%u accumulate vs stencil rel=%g\n", order.c_str(), n, bf, rela); }
                if (relm > REL_TOL) { bad++; std::printf("  MISMATCH %sMatrix n=%u bflag=%u accumulate vs matrix rel=%g\n", order.c_str(), n, bf, relm); }
            }
            const double t_s = time_ns([&]() { ds->filter_cako(u.data(), rs.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, 0); }, 3000);
            const double t_m = time_ns([&]() { dm->filter_cako(u.data(), rm.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, 0); }, 3000);
            if (t_m > t_s) std::printf("  note: %sMatrix n=%u measured slower this run (%.0f vs %.0f ns) — timing only, not a failure\n", order.c_str(), n, t_m, t_s);
            std::printf("  %-4s n=%2u  worst rel diff %.2e   stencil %6.0f ns  matrix %6.0f ns  (%.2fx)\n",
                        order.c_str(), n, worst, t_s, t_m, t_s / t_m);
        }
    }
    std::printf("KO matrix filters: %lu checks, %lu failures -> %s\n", checked,
                bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
