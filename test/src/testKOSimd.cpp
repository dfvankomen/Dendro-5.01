// Gate for the span-vectorized KO filters (KO{2,4,6,8}Simd): for every order
// the padding width supports, every bflag variant and block sizes
// n = 13 / 17 / 21, the simd path must agree with the explicit stencil path
// (uniform filter(), per-point filter_cako(), and ko_accumulate()) — bit for
// bit on boundary blocks (delegated) and to roundoff on interior blocks — and
// a clone must reproduce its source exactly. Whether the interior is also
// bit-identical is reported. Timings (stencil / simd / matrix) are printed
// for information only. Exit code 0 = pass.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "derivatives.h"

using namespace dendroderivs;

static const double REL_TOL = 1e-13;

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

static std::unique_ptr<DendroDerivatives> make(const std::string &filter,
                                               unsigned int eo) {
    return std::make_unique<DendroDerivatives>(
        "E6", "E6", eo, std::vector<double>(), std::vector<double>(), 0u, 0u,
        "none", "none", std::vector<double>(), std::vector<double>(), filter);
}

int main() {
    const unsigned int L = 1u << OCT_DIR_LEFT, R = 1u << OCT_DIR_RIGHT,
                       D = 1u << OCT_DIR_DOWN, U = 1u << OCT_DIR_UP,
                       B = 1u << OCT_DIR_BACK, F = 1u << OCT_DIR_FRONT;
    const std::vector<unsigned int> bflags = {0, L, R, L | R, D, U, D | U, B, F,
                                             B | F, L | D | B,
                                             L | R | D | U | B | F};
    unsigned long checked = 0, bad = 0;
    bool all_bitwise = true;

    for (const std::string order : {"KO2", "KO4", "KO6", "KO8"}) {
        for (unsigned int eo : {6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            const double dx = 0.05, dy = 0.07, dz = 0.11;

            std::unique_ptr<DendroDerivatives> ds, dv, dm;
            try {
                ds = make(order, eo);
                dv = make(order + "Simd", eo);
                dm = make(order + "Matrix", eo);
            } catch (const std::exception &e) {
                std::printf("  [skip] %s at eleorder %u: %s\n", order.c_str(), eo, e.what());
                continue;
            }
            ds->set_maximum_block_size(tot);
            dv->set_maximum_block_size(tot);
            dm->set_maximum_block_size(tot);
            DendroDerivatives dc(*dv);  // clone, as the per-thread pool does
            dc.set_maximum_block_size(tot);

            std::vector<double> u(tot), coeff(tot), base(tot), wx(tot), wy(tot), wz(tot);
            for (size_t i = 0; i < tot; i++) {
                u[i]     = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
                coeff[i] = 0.1 + 0.05 * std::sin(0.2 * i);
                base[i]  = std::cos(0.1 * i);
            }

            double worst = 0.0, worst_interior_abs = 0.0;
            for (unsigned int bf : bflags) {
                // three entries: uniform filter, per-point cako, accumulate
                std::vector<double> rs[3] = {base, base, base}, rv[3] = {base, base, base}, rc(base);
                ds->filter(u.data(), rs[0].data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 0.1, sz, bf);
                dv->filter(u.data(), rv[0].data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 0.1, sz, bf);
                ds->filter_cako(u.data(), rs[1].data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                dv->filter_cako(u.data(), rv[1].data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                rs[2] = rs[0];  // stencil has no accumulate: uniform filter is the reference
                const bool did = dv->ko_accumulate(rv[2].data(), u.data(), 0.1, dx, dy, dz, sz, bf);
                dc.filter_cako(u.data(), rc.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                if (!did) { bad++; std::printf("  MISMATCH %sSimd n=%u bflag=%u ko_accumulate declined\n", order.c_str(), n, bf); }
                static const char *what[3] = {"filter", "cako", "accumulate"};
                for (int e = 0; e < 3; e++) {
                    double md = 0.0, scale = 0.0;
                    for (unsigned int k = pw; k < n - pw; k++)
                        for (unsigned int j = pw; j < n - pw; j++)
                            for (unsigned int i = pw; i < n - pw; i++) {
                                const size_t p = i + n * (j + n * k);
                                md    = std::max(md, std::fabs(rv[e][p] - rs[e][p]));
                                scale = std::max(scale, std::fabs(rs[e][p] - base[p]));
                            }
                    checked++;
                    const double rel = scale > 0 ? md / scale : md;
                    worst = std::max(worst, rel);
                    if (bf == 0) worst_interior_abs = std::max(worst_interior_abs, md);
                    const bool ok = bf ? (md == 0.0) : (rel <= REL_TOL);
                    if (!ok) { bad++; std::printf("  MISMATCH %sSimd n=%u bflag=%u %s rel=%g abs=%g\n", order.c_str(), n, bf, what[e], rel, md); }
                }
                double mdc = 0.0;
                for (size_t p = 0; p < tot; p++) mdc = std::max(mdc, std::fabs(rc[p] - rv[1][p]));
                checked++;
                if (mdc != 0.0) { bad++; std::printf("  MISMATCH %sSimd n=%u bflag=%u clone differs by %g\n", order.c_str(), n, bf, mdc); }
            }
            if (worst_interior_abs != 0.0) all_bitwise = false;

            std::vector<double> o(tot);
            const double t_s = time_ns([&]() { ds->filter(u.data(), o.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 0.1, sz, 0); }, 3000);
            const double t_v = time_ns([&]() { dv->filter(u.data(), o.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 0.1, sz, 0); }, 3000);
            const double t_m = time_ns([&]() { dm->filter(u.data(), o.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 0.1, sz, 0); }, 3000);
            const double t_sc = time_ns([&]() { ds->filter_cako(u.data(), o.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, 0); }, 3000);
            const double t_vc = time_ns([&]() { dv->filter_cako(u.data(), o.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, coeff.data(), sz, 0); }, 3000);
            const double t_ma = time_ns([&]() { dm->ko_accumulate(o.data(), u.data(), 0.1, dx, dy, dz, sz, 0); }, 3000);
            std::printf("  %-4s n=%2u  worst rel %.1e  interior %s   filter: stencil %6.0f  simd %6.0f (%.2fx)  matrix %6.0f (%.2fx)  ko_acc %6.0f (%.2fx) | cako: stencil %6.0f  simd %6.0f (%.2fx) ns\n",
                        order.c_str(), n, worst, worst_interior_abs == 0.0 ? "bitwise " : "roundoff",
                        t_s, t_v, t_s / t_v, t_m, t_s / t_m, t_ma, t_s / t_ma, t_sc, t_vc, t_sc / t_vc);
        }
    }
    std::printf("KO simd filters: %lu checks, %lu failures, interior %s -> %s\n", checked, bad,
                all_bitwise ? "bit-identical to the stencil" : "within roundoff of the stencil",
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
