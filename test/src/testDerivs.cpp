// The derivative and KO unit gates, one section per former executable (the
// section keeps that name, so `gate: testKOSign` comments still apply). No
// MPI, no mesh. Every section runs even after an earlier one fails; an
// exception counts as that section's failure. None of them asserts or exits.
// Usage: testDerivs [section ...]   (no argument runs them all)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "derivatives.h"
#include "derivatives/derivs_factory.h"

using namespace dendroderivs;

// Gate for engine copies meeting a block size they have never seen: a remesh
// can introduce one after the per-thread clones were made, and the clone must
// build its own D storage (filter included) and match the original bit for bit.
namespace testCloneNewSize {
int run() {
    struct Cfg {
        std::string d1, d2, filt;
    };
    const std::vector<Cfg> cfgs = {{"JTT6", "JTT6", "none"},
                                   {"JTP6", "JTP6", "none"},
                                   {"JTT6", "JTT6", "KIM"},
                                   {"E6", "E6", "none"},
                                   {"E6Simd", "E6Simd", "none"}};
    const unsigned int eo = 6, pw = 3;
    unsigned long checked = 0, bad = 0;
    for (const Cfg &c : cfgs) {
        std::unique_ptr<DendroDerivatives> orig;
        try {
            // KIM in-matrix needs {sigma, kc_factor}
            const std::vector<double> fc = c.filt == "KIM"
                                               ? std::vector<double>{1.0, 0.5}
                                               : std::vector<double>();
            orig                         = std::make_unique<DendroDerivatives>(
                c.d1, c.d2, eo, std::vector<double>(), std::vector<double>(), 0,
                0, c.filt, c.filt, fc, fc);
        } catch (const std::exception &e) {
            std::printf("  [skip] %s/%s filter %s: %s\n", c.d1.c_str(),
                        c.d2.c_str(), c.filt.c_str(), e.what());
            continue;
        }
        orig->set_maximum_block_size(13 * 13 * 13);
        orig->pre_create_for_size(13);
        DendroDerivatives lazy(*orig);  // used on the new size with no warm-up
        DendroDerivatives warmed(
            *orig);  // pre_create_for_size first, as the solver does
        DendroDerivatives twice(lazy);  // copy of a copy
        // init() builds n = 13..55 eagerly (DDERIVS_MAX_BLOCKS_INIT); these are
        // not in the map
        for (unsigned int n : {61u, 67u}) {
            const size_t tot         = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            std::vector<double> u(tot), ws(tot);
            for (size_t i = 0; i < tot; i++)
                u[i] = std::sin(0.31 * i) + 0.2 * std::cos(0.007 * i * i);
            for (DendroDerivatives *d : {orig.get(), &lazy, &warmed, &twice})
                d->set_maximum_block_size(tot);
            warmed.pre_create_for_size(n);
            auto run = [&](DendroDerivatives &d,
                           std::vector<std::vector<double>> &r) {
                DendroDerivatives::DerivSet out;
                out.x  = r[0].data();
                out.y  = r[1].data();
                out.z  = r[2].data();
                out.xx = r[3].data();
                out.yy = r[4].data();
                out.zz = r[5].data();
                out.xy = r[6].data();
                out.xz = r[7].data();
                out.yz = r[8].data();
                d.grad_set(out, u.data(), DendroDerivatives::DM_ALL, 0.05, 0.07,
                           0.11, sz, 0, ws.data());
            };
            std::vector<std::vector<double>> ro(9, std::vector<double>(tot)),
                rc(9, std::vector<double>(tot));
            run(*orig, ro);
            for (DendroDerivatives *d : {&lazy, &warmed, &twice}) {
                for (auto &v : rc) std::fill(v.begin(), v.end(), 0.0);
                run(*d, rc);
                for (int b = 0; b < 9; b++) {
                    bool same = true;
                    for (unsigned int k = pw; k < n - pw && same; k++)
                        for (unsigned int j = pw; j < n - pw && same; j++)
                            for (unsigned int i = pw; i < n - pw; i++) {
                                const size_t p = i + n * (j + n * k);
                                if (ro[b][p] != rc[b][p]) {
                                    same = false;
                                    break;
                                }
                            }
                    checked++;
                    if (!same) {
                        bad++;
                        std::printf("  MISMATCH %s/%s filter %s n=%u out=%d\n",
                                    c.d1.c_str(), c.d2.c_str(), c.filt.c_str(),
                                    n, b);
                    }
                }
            }
        }
        std::printf(
            "  %s/%s filter %-6s: copies agree on unseen sizes 61, 67\n",
            c.d1.c_str(), c.d2.c_str(), c.filt.c_str());
    }
    std::printf("clone-then-new-size: %lu checks, %lu failures -> %s\n",
                checked, bad, (bad == 0 && checked > 0) ? "PASS" : "FAIL");
    return (bad == 0 && checked > 0) ? 0 : 1;
}
}  // namespace testCloneNewSize

// Gate for DendroDerivatives::grad_set — the planned per-variable derivative
// entry point. For every one of the 512 masks, every bflag variant and the
// three engine families (matrix compact, explicit, CCFD), each requested
// output must match the corresponding individual grad_* call bit-for-bit on
// the active region. Also checks the null-output guard and the batch form.
//
// Exit code 0 = all identical, 1 = a mismatch (printed).
namespace testDerivSet {
static double maxdiff_active(const double *a, const double *b, unsigned int n,
                             unsigned int pw) {
    double m = 0.0;
    for (unsigned int k = pw; k < n - pw; k++)
        for (unsigned int j = pw; j < n - pw; j++)
            for (unsigned int i = pw; i < n - pw; i++) {
                const unsigned int idx = i + n * (j + n * k);
                m = std::max(m, std::fabs(a[idx] - b[idx]));
            }
    return m;
}

static double maxabs_active(const double *a, unsigned int n, unsigned int pw) {
    double m = 0.0;
    for (unsigned int k = pw; k < n - pw; k++)
        for (unsigned int j = pw; j < n - pw; j++)
            for (unsigned int i = pw; i < n - pw; i++)
                m = std::max(m, std::fabs(a[i + n * (j + n * k)]));
    return m;
}

// grad_set batches the same arithmetic in a different order, so it agrees to
// roundoff rather than bit-for-bit. Scale by the derivative magnitude: 2nd
// derivatives carry 1/h^2, so an absolute bound would not travel across axes.
static bool agrees(const double *a, const double *b, unsigned int n,
                   unsigned int pw, double &md) {
    const double rtol = 1e-12, atol = 1e-14;
    md = maxdiff_active(a, b, n, pw);
    return md <= atol + rtol * maxabs_active(b, n, pw);
}

int run() {
    const unsigned int eo = 6, n = 2 * eo + 1, pw = eo / 2;
    const size_t tot         = (size_t)n * n * n;
    const unsigned int sz[3] = {n, n, n};
    const double dx = 0.05, dy = 0.07, dz = 0.11;
    const unsigned int L = 1u << OCT_DIR_LEFT, R = 1u << OCT_DIR_RIGHT,
                       D = 1u << OCT_DIR_DOWN, U = 1u << OCT_DIR_UP,
                       B = 1u << OCT_DIR_BACK, F = 1u << OCT_DIR_FRONT;
    const std::vector<unsigned int> bflags = {0, L | D | B, R | U | F,
                                              L | R | D | U | B | F};
    const std::vector<std::pair<std::string, std::string>> engines = {
        {"JTT6", "JTT6"}, {"E6", "E6"}, {"CCFD6", "CCFD6"}};

    std::vector<double> u(tot);
    for (size_t i = 0; i < tot; i++)
        u[i] = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);

    unsigned long checked = 0, bad = 0;
    for (const auto &e : engines) {
        DendroDerivatives dd(e.first, e.second, eo);
        dd.set_maximum_block_size(tot);
        std::vector<double> ws(tot);
        std::vector<std::vector<double>> ref(9, std::vector<double>(tot)),
            got(9, std::vector<double>(tot));

        for (unsigned int bf : bflags) {
            // reference: the individual calls
            dd.grad_x(ref[0].data(), u.data(), dx, sz, bf);
            dd.grad_y(ref[1].data(), u.data(), dy, sz, bf);
            dd.grad_z(ref[2].data(), u.data(), dz, sz, bf);
            dd.grad_xx(ref[3].data(), u.data(), dx, sz, bf);
            dd.grad_yy(ref[4].data(), u.data(), dy, sz, bf);
            dd.grad_zz(ref[5].data(), u.data(), dz, sz, bf);
            dd.grad_xy(ref[6].data(), u.data(), ws.data(), dx, dy, sz, bf);
            dd.grad_xz(ref[7].data(), u.data(), ws.data(), dx, dz, sz, bf);
            dd.grad_yz(ref[8].data(), u.data(), ws.data(), dy, dz, sz, bf);

            for (unsigned int mask = 0; mask <= DendroDerivatives::DM_ALL;
                 mask++) {
                for (auto &g : got) std::fill(g.begin(), g.end(), 0.0);
                DendroDerivatives::DerivSet out;
                double *slots[9] = {nullptr};
                for (int b = 0; b < 9; b++)
                    if (mask & (1u << b)) slots[b] = got[b].data();
                out.x  = slots[0];
                out.y  = slots[1];
                out.z  = slots[2];
                out.xx = slots[3];
                out.yy = slots[4];
                out.zz = slots[5];
                out.xy = slots[6];
                out.xz = slots[7];
                out.yz = slots[8];
                dd.grad_set(out, u.data(), mask, dx, dy, dz, sz, bf, ws.data());
                static const char *names[9] = {"x",  "y",  "z",  "xx", "yy",
                                               "zz", "xy", "xz", "yz"};
                for (int b = 0; b < 9; b++) {
                    if (!(mask & (1u << b))) continue;
                    checked++;
                    double md = 0.0;
                    if (!agrees(got[b].data(), ref[b].data(), n, pw, md)) {
                        bad++;
                        if (bad <= 20)
                            std::printf(
                                "  MISMATCH %s/%s bflag=%u mask=%u %s "
                                "maxdiff=%g\n",
                                e.first.c_str(), e.second.c_str(), bf, mask,
                                names[b], md);
                    }
                }
            }

            // batch form: two variables, same mask
            {
                std::vector<double> u2(tot);
                for (size_t i = 0; i < tot; i++) u2[i] = 0.5 * u[i] + 0.1;
                std::vector<double> o1(tot), o2(tot), r2(tot);
                DendroDerivatives::DerivSet outs[2];
                outs[0].x           = o1.data();
                outs[1].x           = o2.data();
                const double *us[2] = {u.data(), u2.data()};
                dd.grad_set_batch(outs, us, 2, DendroDerivatives::DM_X, dx, dy,
                                  dz, sz, bf);
                dd.grad_x_last(r2.data(), u2.data(), dx, sz, bf);
                checked += 2;
                double mdb1 = 0.0, mdb2 = 0.0;
                if (!agrees(o1.data(), ref[0].data(), n, pw, mdb1) ||
                    !agrees(o2.data(), r2.data(), n, pw, mdb2)) {
                    bad++;
                    std::printf("  MISMATCH batch %s bflag=%u\n",
                                e.first.c_str(), bf);
                }
            }
        }

        // null-output guard
        {
            bool threw = false;
            try {
                DendroDerivatives::DerivSet out;  // all null
                dd.grad_set(out, u.data(), DendroDerivatives::DM_X, dx, dy, dz,
                            sz, 0);
            } catch (const std::invalid_argument &) {
                threw = true;
            }
            checked++;
            if (!threw) {
                bad++;
                std::printf("  MISMATCH %s: null output did not throw\n",
                            e.first.c_str());
            }
        }
        std::printf("engine %s/%s: done\n", e.first.c_str(), e.second.c_str());
    }

    std::printf("grad_set: %lu outputs checked, %lu mismatches -> %s\n",
                checked, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
}  // namespace testDerivSet

// Gate for the simd explicit engines (E4/E6/E8Simd): every grad_set mask
// against the loop engine of the same order, all bflag variants, block sizes
// the padding allows. Interior blocks must agree to roundoff; boundary blocks
// (delegated to the loop engine) must agree exactly. Timing is informational.
namespace testExplicitSimd {
static const double REL_TOL = 1e-13;

static double maxdiff_active(const double *a, const double *b, unsigned int n,
                             unsigned int pw, double &scale) {
    double m = 0.0;
    for (unsigned int k = pw; k < n - pw; k++)
        for (unsigned int j = pw; j < n - pw; j++)
            for (unsigned int i = pw; i < n - pw; i++) {
                const size_t p = i + n * (j + n * k);
                m              = std::max(m, std::fabs(a[p] - b[p]));
                scale          = std::max(scale, std::fabs(b[p]));
            }
    return m;
}

int run() {
    const unsigned int L = 1u << OCT_DIR_LEFT, R = 1u << OCT_DIR_RIGHT,
                       D = 1u << OCT_DIR_DOWN, U = 1u << OCT_DIR_UP,
                       B = 1u << OCT_DIR_BACK, F = 1u << OCT_DIR_FRONT;
    const std::vector<unsigned int> bflags = {0, L, R | U, D | B,
                                              L | R | D | U | B | F};
    unsigned long checked = 0, bad = 0;
    for (const std::string order : {"E4", "E6", "E8"}) {
        for (unsigned int eo : {4u, 6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot         = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            const double dx = 0.05, dy = 0.07, dz = 0.11;
            std::unique_ptr<DendroDerivatives> dl, ds;
            try {
                dl = std::make_unique<DendroDerivatives>(order, order, eo);
                ds = std::make_unique<DendroDerivatives>(order + "Simd",
                                                         order + "Simd", eo);
            } catch (const std::exception &e) {
                std::printf("  [skip] %sSimd at eleorder %u: %s\n",
                            order.c_str(), eo, e.what());
                continue;
            }
            dl->set_maximum_block_size(tot);
            ds->set_maximum_block_size(tot);
            DendroDerivatives dc(*ds);
            std::vector<double> u(tot), ws(tot);
            for (size_t i = 0; i < tot; i++)
                u[i] = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
            std::vector<std::vector<double>> rl(9, std::vector<double>(tot)),
                rs(9, std::vector<double>(tot));
            double worst = 0.0;
            for (unsigned int bf : bflags) {
                for (unsigned int mask :
                     {(unsigned)DendroDerivatives::DM_ALL,
                      (unsigned)DendroDerivatives::DM_FIRST,
                      (unsigned)DendroDerivatives::DM_MIXED,
                      (unsigned)DendroDerivatives::DM_SECOND}) {
                    auto fill = [&](DendroDerivatives &d,
                                    std::vector<std::vector<double>> &r) {
                        for (auto &v : r) std::fill(v.begin(), v.end(), 0.0);
                        DendroDerivatives::DerivSet out;
                        double *s[9] = {r[0].data(), r[1].data(), r[2].data(),
                                        r[3].data(), r[4].data(), r[5].data(),
                                        r[6].data(), r[7].data(), r[8].data()};
                        out.x        = s[0];
                        out.y        = s[1];
                        out.z        = s[2];
                        out.xx       = s[3];
                        out.yy       = s[4];
                        out.zz       = s[5];
                        out.xy       = s[6];
                        out.xz       = s[7];
                        out.yz       = s[8];
                        d.grad_set(out, u.data(), mask, dx, dy, dz, sz, bf,
                                   ws.data());
                    };
                    fill(*dl, rl);
                    fill(dc, rs);
                    for (int b = 0; b < 9; b++) {
                        if (!(mask & (1u << b))) continue;
                        double scale    = 0.0;
                        const double md = maxdiff_active(
                            rs[b].data(), rl[b].data(), n, pw, scale);
                        const double rel = scale > 0 ? md / scale : md;
                        checked++;
                        const bool ok = bf ? (md == 0.0) : (rel <= REL_TOL);
                        worst         = std::max(worst, rel);
                        if (!ok) {
                            bad++;
                            std::printf(
                                "  MISMATCH %sSimd n=%u bflag=%u mask=%u "
                                "out=%d rel=%g\n",
                                order.c_str(), n, bf, mask, b, rel);
                        }
                    }
                }
            }
            auto tm = [&](DendroDerivatives &d) {
                std::vector<double> o(tot);
                for (int i = 0; i < 50; i++)
                    d.grad_x_last(o.data(), u.data(), dx, sz, 0);
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 2000; i++) {
                    d.grad_x_last(o.data(), u.data(), dx, sz, 0);
                    d.grad_y_last(o.data(), u.data(), dy, sz, 0);
                    d.grad_z(o.data(), u.data(), dz, sz, 0);
                }
                return std::chrono::duration<double, std::nano>(
                           std::chrono::steady_clock::now() - t0)
                           .count() /
                       2000;
            };
            std::printf(
                "  %sSimd n=%2u  worst rel diff %.2e   x+y+z: loop %6.0f ns  "
                "simd %6.0f ns  (%.2fx)\n",
                order.c_str(), n, worst, tm(*dl), tm(*ds), tm(*dl) / tm(*ds));
        }
    }
    std::printf("simd explicit engines: %lu checks, %lu failures -> %s\n",
                checked, bad, (bad == 0 && checked > 0) ? "PASS" : "FAIL");
    return (bad == 0 && checked > 0) ? 0 : 1;
}
}  // namespace testExplicitSimd

// Gate for the matrix-form KO dissipation filters (KO{2,4,6,8}Matrix): for
// every order the padding width supports, every bflag variant and block sizes
// n = 13 / 17 / 21, the matrix path must agree with the explicit stencil path
// to roundoff (both the scalar-coefficient and the per-point CAKO entry) and a
// clone must reproduce its source exactly. Timings are printed for information
// only — a shared or noisy node must not turn an accuracy gate red.
// Exit code 0 = pass.
namespace testKOMatrix {
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

int run() {
    const unsigned int L = 1u << OCT_DIR_LEFT, R = 1u << OCT_DIR_RIGHT,
                       D = 1u << OCT_DIR_DOWN, U = 1u << OCT_DIR_UP,
                       B = 1u << OCT_DIR_BACK, F = 1u << OCT_DIR_FRONT;
    const std::vector<unsigned int> bflags = {
        0,     L, R, L | R, D,         U,
        D | U, B, F, B | F, L | D | B, L | R | D | U | B | F};
    unsigned long checked = 0, bad = 0;

    for (const std::string order : {"KO2", "KO4", "KO6", "KO8"}) {
        for (unsigned int eo : {4u, 6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot         = (size_t)n * n * n;
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
                rs = base;
                rm = base;
                rc = base;
                ds->filter_cako(u.data(), rs.data(), wx.data(), wy.data(),
                                wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                dm->filter_cako(u.data(), rm.data(), wx.data(), wy.data(),
                                wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                dc.filter_cako(u.data(), rc.data(), wx.data(), wy.data(),
                               wz.data(), dx, dy, dz, coeff.data(), sz, bf);
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
                worst            = std::max(worst, rel);
                if (rel > REL_TOL) {
                    bad++;
                    std::printf(
                        "  MISMATCH %sMatrix n=%u bflag=%u cako rel=%g\n",
                        order.c_str(), n, bf, rel);
                }
                if (mdc != 0.0) {
                    bad++;
                    std::printf(
                        "  MISMATCH %sMatrix n=%u bflag=%u clone differs by "
                        "%g\n",
                        order.c_str(), n, bf, mdc);
                }
                // uniform coefficient: ko_accumulate (three beta = 1 GEMMs into
                // rhs) against the stencil's uniform-sigma filter and the
                // matrix filter_cako
                std::vector<double> ru(tot, 0.1), ra(base), rsu(base),
                    rmu(base);
                ds->filter(u.data(), rsu.data(), wx.data(), wy.data(),
                           wz.data(), dx, dy, dz, 0.1, sz, bf);
                dm->filter_cako(u.data(), rmu.data(), wx.data(), wy.data(),
                                wz.data(), dx, dy, dz, ru.data(), sz, bf);
                const bool did = dm->ko_accumulate(ra.data(), u.data(), 0.1, dx,
                                                   dy, dz, sz, bf);
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
                const double rela = sca > 0 ? mda / sca : mda,
                             relm = sca > 0 ? mdm / sca : mdm;
                worst             = std::max(worst, rela);
                if (!did) {
                    bad++;
                    std::printf(
                        "  MISMATCH %sMatrix n=%u bflag=%u ko_accumulate "
                        "declined\n",
                        order.c_str(), n, bf);
                }
                if (rela > REL_TOL) {
                    bad++;
                    std::printf(
                        "  MISMATCH %sMatrix n=%u bflag=%u accumulate vs "
                        "stencil rel=%g\n",
                        order.c_str(), n, bf, rela);
                }
                if (relm > REL_TOL) {
                    bad++;
                    std::printf(
                        "  MISMATCH %sMatrix n=%u bflag=%u accumulate vs "
                        "matrix rel=%g\n",
                        order.c_str(), n, bf, relm);
                }
            }
            const double t_s = time_ns(
                [&]() {
                    ds->filter_cako(u.data(), rs.data(), wx.data(), wy.data(),
                                    wz.data(), dx, dy, dz, coeff.data(), sz, 0);
                },
                3000);
            const double t_m = time_ns(
                [&]() {
                    dm->filter_cako(u.data(), rm.data(), wx.data(), wy.data(),
                                    wz.data(), dx, dy, dz, coeff.data(), sz, 0);
                },
                3000);
            if (t_m > t_s)
                std::printf(
                    "  note: %sMatrix n=%u measured slower this run (%.0f vs "
                    "%.0f ns) — timing only, not a failure\n",
                    order.c_str(), n, t_m, t_s);
            std::printf(
                "  %-4s n=%2u  worst rel diff %.2e   stencil %6.0f ns  matrix "
                "%6.0f ns  (%.2fx)\n",
                order.c_str(), n, worst, t_s, t_m, t_s / t_m);
        }
    }
    std::printf("KO matrix filters: %lu checks, %lu failures -> %s\n", checked,
                bad, (bad == 0 && checked > 0) ? "PASS" : "FAIL");
    return (bad == 0 && checked > 0) ? 0 : 1;
}
}  // namespace testKOMatrix

// Sign gate for the Kreiss-Oliger filters: applied to the Nyquist mode
// u = (-1)^(i+j+k) with coeff = 1 on a block with no boundary flags, every
// KO order must return output += -(1/dx + 1/dy + 1/dz) * u on the active
// region — the same damping rate for every order by the KO normalization,
// and negative, i.e. dissipative. Found because the explicit KO8 stencil grew
// a seeded mode at exactly +0.3/step in EM4 (2026-08-29). With all faces
// closed the active output must not depend on padding and must annihilate a
// constant. Also reports, for information, whether an in-matrix filter
// changes the JTT6 operator at all.
// Exit code 0 = pass.
namespace testKOSign {
extern "C" void dgeev_(const char *, const char *, const int *, double *,
                       const int *, double *, double *, double *, const int *,
                       double *, const int *, double *, const int *, int *);

int run() {
    unsigned long bad = 0, checked = 0;
    for (const std::string order :
         {"KO2", "KO4", "KO6", "KO8", "KO2Matrix", "KO4Matrix", "KO6Matrix",
          "KO8Matrix", "KO2Simd", "KO4Simd", "KO6Simd", "KO8Simd"}) {
        for (unsigned int eo : {4u, 6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot         = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            const double dx = 0.5, dy = 0.25, dz = 0.125;
            std::unique_ptr<DendroDerivatives> d;
            try {
                d = std::make_unique<DendroDerivatives>(
                    "E6", "E6", eo, std::vector<double>(),
                    std::vector<double>(), 0u, 0u, "none", "none",
                    std::vector<double>(), std::vector<double>(), order);
            } catch (const std::exception &) {
                continue;
            }
            d->set_maximum_block_size(tot);
            std::vector<double> u(tot), out(tot, 0.0), wx(tot), wy(tot),
                wz(tot);
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++)
                        u[i + n * (j + n * k)] = ((i + j + k) & 1) ? -1.0 : 1.0;
            d->filter(u.data(), out.data(), wx.data(), wy.data(), wz.data(), dx,
                      dy, dz, 1.0, sz, 0);
            const double expect = -(1.0 / dx + 1.0 / dy + 1.0 / dz);
            double worst        = 0.0;
            for (unsigned int k = pw; k < n - pw; k++)
                for (unsigned int j = pw; j < n - pw; j++)
                    for (unsigned int i = pw; i < n - pw; i++) {
                        const size_t p = i + n * (j + n * k);
                        worst =
                            std::max(worst, std::fabs(out[p] / u[p] - expect));
                    }
            // boundary blocks: the one-sided layers have their own rates, so
            // gate on the sign alone — every active point must be damped
            std::vector<double> ob(tot, 0.0);
            const unsigned int all =
                (1u << OCT_DIR_LEFT) | (1u << OCT_DIR_RIGHT) |
                (1u << OCT_DIR_DOWN) | (1u << OCT_DIR_UP) |
                (1u << OCT_DIR_BACK) | (1u << OCT_DIR_FRONT);
            d->filter(u.data(), ob.data(), wx.data(), wy.data(), wz.data(), dx,
                      dy, dz, 1.0, sz, all);
            unsigned long wrong = 0, zero = 0, pos = 0;
            double rmin = 1e300, rmax = -1e300;
            unsigned int imin = n, imax = 0;
            for (unsigned int k = pw; k < n - pw; k++)
                for (unsigned int j = pw; j < n - pw; j++)
                    for (unsigned int i = pw; i < n - pw; i++) {
                        const size_t p = i + n * (j + n * k);
                        const double r = ob[p] / u[p];
                        rmin           = std::min(rmin, r);
                        rmax           = std::max(rmax, r);
                        if (r >= 0.0) {
                            wrong++;
                            if (r == 0.0)
                                zero++;
                            else
                                pos++;
                            imin = std::min(imin, i);
                            imax = std::max(imax, i);
                        }
                    }
            checked++;
            if (wrong) {
                bad++;
                std::printf(
                    "  %-10s n=%2u  bflag=ALL: %lu active points NOT damped "
                    "(%lu zero, %lu positive; out/u in [%+.3f, %+.3f]; i in "
                    "[%u,%u])\n",
                    order.c_str(), n, wrong, zero, pos, rmin, rmax, imin, imax);
            }
            // all faces closed: the active output must not read padding, and
            // a constant must be annihilated on every active row
            {
                auto active = [&](unsigned int i, unsigned int j,
                                  unsigned int k) {
                    return i >= pw && i < n - pw && j >= pw && j < n - pw &&
                           k >= pw && k < n - pw;
                };
                std::vector<double> ug(u), uc(tot), og(tot, 0.0), oc(tot, 0.0);
                for (unsigned int k = 0; k < n; k++)
                    for (unsigned int j = 0; j < n; j++)
                        for (unsigned int i = 0; i < n; i++) {
                            const size_t p = i + n * (j + n * k);
                            const bool a   = active(i, j, k);
                            if (!a) ug[p] = 1e3 * std::sin(0.7 * p);
                            uc[p] = a ? 1.0 : 1e3 * std::cos(0.3 * p);
                        }
                d->filter(ug.data(), og.data(), wx.data(), wy.data(), wz.data(),
                          dx, dy, dz, 1.0, sz, all);
                d->filter(uc.data(), oc.data(), wx.data(), wy.data(), wz.data(),
                          dx, dy, dz, 1.0, sz, all);
                unsigned long reads = 0;
                double cmax         = 0.0;
                for (unsigned int k = pw; k < n - pw; k++)
                    for (unsigned int j = pw; j < n - pw; j++)
                        for (unsigned int i = pw; i < n - pw; i++) {
                            const size_t p = i + n * (j + n * k);
                            if (og[p] != ob[p]) reads++;
                            cmax = std::max(cmax, std::isfinite(oc[p])
                                                      ? std::fabs(oc[p])
                                                      : HUGE_VAL);
                        }
                checked += 2;
                if (reads) {
                    bad++;
                    std::printf(
                        "  %-10s n=%2u  bflag=ALL: %lu active points depend on "
                        "padding\n",
                        order.c_str(), n, reads);
                }
                if (cmax > 1e-10 / dz) {
                    bad++;
                    std::printf(
                        "  %-10s n=%2u  bflag=ALL: constant not annihilated, "
                        "max |out| = %.3e\n",
                        order.c_str(), n, cmax);
                }
            }
            checked++;
            const bool ok = worst < 1e-10 * std::fabs(expect);
            if (!ok) bad++;
            std::printf(
                "  %-10s n=%2u  out/u on Nyquist = %+.6f  expected %+.6f  -> "
                "%s\n",
                order.c_str(), n,
                out[(pw) + n * (pw + n * pw)] / u[(pw) + n * (pw + n * pw)],
                expect, ok ? "ok" : "WRONG SIGN / RATE");
        }
    }
    // KIMF: gate on the energy form + spectrum and report the transfer
    // (its Nyquist rate is 1 - G(pi), not the KO normalization)
    for (unsigned int eo : {4u, 6u, 10u}) {
        const unsigned int n = 2 * eo + 1, pw = eo / 2;
        const size_t tot         = (size_t)n * n * n;
        const unsigned int sz[3] = {n, n, n};
        std::unique_ptr<DendroDerivatives> d;
        try {
            d = std::make_unique<DendroDerivatives>(
                "E6", "E6", eo, std::vector<double>(), std::vector<double>(),
                0u, 0u, "none", "none", std::vector<double>(),
                std::vector<double>(), "KIMF");
        } catch (const std::exception &e) {
            std::printf("  [skip] KIMF at eleorder %u: %s\n", eo, e.what());
            continue;
        }
        d->set_maximum_block_size(tot);
        std::vector<double> u(tot), out(tot), wx(tot), wy(tot), wz(tot);
        const double dx = 1.0;
        auto response   = [&](int period) {  // out/u with coeff 1, dx=dy=dz=1
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++)
                        u[i + n * (j + (size_t)n * k)] =
                            std::cos(M_PI * 2.0 * (double)(i + j + k) / period);
            std::fill(out.begin(), out.end(), 0.0);
            d->filter(u.data(), out.data(), wx.data(), wy.data(), wz.data(), dx,
                      dx, dx, 1.0, sz, 0);
            const size_t c = (n / 2) + n * ((n / 2) + (size_t)n * (n / 2));
            return out[c] / u[c];
        };
        const double rN = response(2), r4 = response(4);
        // energy gate <u, Du> <= 0: R^-1 spreads the boundary rows, so an NSD
        // operator may still respond positively at single points -- pointwise
        // sign is the wrong test for an implicit filter
        unsigned long wrong = 0;
        double worst_pos    = 0.0;
        unsigned int wpi    = 0;
        bool wpb            = false;
        for (unsigned int bfsel = 0; bfsel < 2; bfsel++) {
            const unsigned int bf =
                bfsel ? ((1u << OCT_DIR_LEFT) | (1u << OCT_DIR_RIGHT) |
                         (1u << OCT_DIR_DOWN) | (1u << OCT_DIR_UP) |
                         (1u << OCT_DIR_BACK) | (1u << OCT_DIR_FRONT))
                      : 0u;
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++)
                        u[i + n * (j + (size_t)n * k)] =
                            ((i + j + k) & 1) ? -1.0 : 1.0;
            std::fill(out.begin(), out.end(), 0.0);
            d->filter(u.data(), out.data(), wx.data(), wy.data(), wz.data(), dx,
                      dx, dx, 1.0, sz, bf);
            double energy = 0.0;
            for (unsigned int k = pw; k < n - pw; k++)
                for (unsigned int j = pw; j < n - pw; j++)
                    for (unsigned int i = pw; i < n - pw; i++) {
                        const size_t p = i + n * (j + (size_t)n * k);
                        energy += u[p] * out[p];
                        const double r = out[p] / u[p];
                        if (r > worst_pos) {
                            worst_pos = r;
                            wpi       = i;
                            wpb       = bfsel;
                        }
                    }
            checked++;
            if (energy > 0.0) {
                wrong++;
                bad++;
                std::printf(
                    "  KIMF n=%u bflag=%s: <u, Du> = %+.3e > 0 on the Nyquist "
                    "mode\n",
                    n, bfsel ? "ALL" : "0", energy);
            }
        }
        if (worst_pos > 0.0)
            std::printf(
                "  KIMF n=%2u  info: largest positive pointwise response %+.3e "
                "at i=%u (%s block) -- NSD operator, see spectrum\n",
                n, worst_pos, wpi, wpb ? "all-faces" : "interior");
        std::printf(
            "  KIMF n=%2u  response per axis (coeff 1, h 1): Nyquist %+.4f "
            "(KO: -1.0000)  4dx %+.6f (KO4: -0.1250)  ratio %.0fx gentler\n",
            n, rN / 3.0, r4 / 3.0,
            (0.125) / std::max(1e-12, std::fabs(r4 / 3.0)));
        {
            // speed: must match matrix KO (same three GEMMs)
            auto dm = std::make_unique<DendroDerivatives>(
                "E6", "E6", eo, std::vector<double>(), std::vector<double>(),
                0u, 0u, "none", "none", std::vector<double>(),
                std::vector<double>(), "KO4Matrix");
            dm->set_maximum_block_size(tot);
            std::vector<double> rhs(tot, 0.0);
            auto tns = [&](DendroDerivatives &dd) {
                for (int i = 0; i < 100; i++)
                    dd.ko_accumulate(rhs.data(), u.data(), 0.4, 0.1, 0.1, 0.1,
                                     sz, 0);
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 2000; i++)
                    dd.ko_accumulate(rhs.data(), u.data(), 0.4, 0.1, 0.1, 0.1,
                                     sz, 0);
                return std::chrono::duration<double, std::nano>(
                           std::chrono::steady_clock::now() - t0)
                           .count() /
                       2000;
            };
            std::printf(
                "  KIMF n=%2u  ko_accumulate: %6.0f ns  (KO4Matrix: %6.0f "
                "ns)\n",
                n, tns(*d), tns(*dm));
        }
    }

    // spectral gate: with both faces of an axis physical, the 1-D operator on
    // the active points is square (no padding is read); probe it column by
    // column with unit vectors constant along the other axes (their stencils
    // vanish exactly) and require every eigenvalue to have Re <= 0
    for (const std::string order : {"KO2", "KO4", "KO6", "KO8", "KIMF"}) {
        for (unsigned int eo : {4u, 6u, 8u, 10u, 12u, 16u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2, na = n - 2 * pw;
            const size_t tot         = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            std::unique_ptr<DendroDerivatives> d;
            try {
                d = std::make_unique<DendroDerivatives>(
                    "E6", "E6", eo, std::vector<double>(),
                    std::vector<double>(), 0u, 0u, "none", "none",
                    std::vector<double>(), std::vector<double>(), order);
            } catch (const std::exception &) {
                continue;
            }
            d->set_maximum_block_size(tot);
            const unsigned int lo[3] = {1u << OCT_DIR_LEFT, 1u << OCT_DIR_DOWN,
                                        1u << OCT_DIR_BACK};
            const unsigned int hi[3] = {1u << OCT_DIR_RIGHT, 1u << OCT_DIR_UP,
                                        1u << OCT_DIR_FRONT};
            for (int axis = 0; axis < 3; axis++) {
                std::vector<double> e(tot), out(tot), wx(tot), wy(tot), wz(tot),
                    A((size_t)na * na, 0.0);
                auto idx = [&](unsigned int a, unsigned int b, unsigned int c) {
                    unsigned int q[3];
                    q[axis]           = a;
                    q[(axis + 1) % 3] = b;
                    q[(axis + 2) % 3] = c;
                    return (size_t)q[0] + n * (q[1] + (size_t)n * q[2]);
                };
                for (unsigned int c = 0; c < na; c++) {
                    std::fill(e.begin(), e.end(), 0.0);
                    std::fill(out.begin(), out.end(), 0.0);
                    for (unsigned int b = 0; b < n; b++)
                        for (unsigned int cc = 0; cc < n; cc++)
                            e[idx(pw + c, b, cc)] = 1.0;
                    d->filter(e.data(), out.data(), wx.data(), wy.data(),
                              wz.data(), 1.0, 1.0, 1.0, 1.0, sz,
                              lo[axis] | hi[axis]);
                    for (unsigned int r = 0; r < na; r++)
                        A[r + (size_t)na * c] =
                            out[idx(pw + r, pw, pw)];  // column-major
                }
                int N = na, lwork = 8 * N, info = 0, one = 1;
                std::vector<double> wr(N), wi(N), work(lwork);
                dgeev_("N", "N", &N, A.data(), &N, wr.data(), wi.data(),
                       nullptr, &one, nullptr, &one, work.data(), &lwork,
                       &info);
                double remax = -1e300, amax = 0.0;
                for (int i = 0; i < N; i++) {
                    remax = std::max(remax, wr[i]);
                    amax  = std::max(amax, std::hypot(wr[i], wi[i]));
                }
                checked++;
                const bool ok = info == 0 && remax <= 1e-10 * amax;
                if (!ok) {
                    bad++;
                    std::printf(
                        "  %-4s n=%2u axis %d both faces closed: max Re(eig) = "
                        "%+.3e (|eig|max %.3e) -> NOT dissipative\n",
                        order.c_str(), n, axis, remax, amax);
                } else if (axis == 0)
                    std::printf(
                        "  %-4s n=%2u  both faces closed: max Re(eig) = %+.2e "
                        "of |eig|max %.2e -> dissipative\n",
                        order.c_str(), n, remax, amax);
            }
        }
    }

    // KIM in-matrix: one coefficient leaves kc = pi (identity filter), so it
    // must throw; {1, 0.5} must change the JTT6 operator
    {
        const unsigned int eo = 10, n = 21;
        const size_t tot         = (size_t)n * n * n;
        const unsigned int sz[3] = {n, n, n};
        auto make                = [&](const std::vector<double> &c) {
            return std::make_unique<DendroDerivatives>(
                "JTT6", "E6", eo, std::vector<double>(), std::vector<double>(),
                0u, 0u, c.empty() ? "none" : "KIM", "none", c,
                std::vector<double>(), "KO4");
        };
        bool refused = false;
        try {
            make({0.5});
        } catch (const std::invalid_argument &) {
            refused = true;
        }
        checked++;
        if (!refused) {
            bad++;
            std::printf(
                "  KIM in-matrix with one coefficient was accepted (kc = pi, "
                "identity filter)\n");
        }
        auto a = make({}), b = make({1.0, 0.5});
        a->set_maximum_block_size(tot);
        b->set_maximum_block_size(tot);
        std::vector<double> u(tot), oa(tot), ob(tot);
        for (size_t i = 0; i < tot; i++)
            u[i] = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
        a->grad_x(oa.data(), u.data(), 0.1, sz, 0);
        b->grad_x(ob.data(), u.data(), 0.1, sz, 0);
        double md = 0.0, sc = 0.0;
        for (size_t i = 0; i < tot; i++) {
            md = std::max(md, std::fabs(oa[i] - ob[i]));
            sc = std::max(sc, std::fabs(oa[i]));
        }
        checked++;
        std::printf(
            "  KIM in-matrix {1, 0.5}: JTT6 grad_x max rel diff vs none %.3e\n",
            md / sc);
        if (!(md / sc > 1e-3)) {
            bad++;
            std::printf(
                "  KIM in-matrix {1, 0.5} left the operator unchanged\n");
        }
    }

    // info: does an in-matrix filter change the first-derivative operator?
    for (const std::string f : {"BYUT6"}) {
        const unsigned int eo = 10, n = 21;
        const size_t tot         = (size_t)n * n * n;
        const unsigned int sz[3] = {n, n, n};
        try {
            DendroDerivatives a("JTT6", "E6", eo, std::vector<double>(),
                                std::vector<double>(), 0u, 0u, "none", "none",
                                std::vector<double>(), std::vector<double>(),
                                "KO4");
            DendroDerivatives b("JTT6", "E6", eo, std::vector<double>(),
                                std::vector<double>(), 0u, 0u, f, "none",
                                std::vector<double>{0.5}, std::vector<double>(),
                                "KO4");
            a.set_maximum_block_size(tot);
            b.set_maximum_block_size(tot);
            std::vector<double> u(tot), oa(tot), ob(tot);
            for (size_t i = 0; i < tot; i++)
                u[i] = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
            a.grad_x(oa.data(), u.data(), 0.1, sz, 0);
            b.grad_x(ob.data(), u.data(), 0.1, sz, 0);
            double md = 0.0, sc = 0.0;
            for (size_t i = 0; i < tot; i++) {
                md = std::max(md, std::fabs(oa[i] - ob[i]));
                sc = std::max(sc, std::fabs(oa[i]));
            }
            std::printf(
                "  info: JTT6 grad_x with in-matrix %s (coeff 0.5) vs none: "
                "max rel diff %.3e  [%s]\n",
                f.c_str(), md / sc, b.toString().c_str());
        } catch (const std::exception &e) {
            std::printf("  info: in-matrix %s: %s\n", f.c_str(), e.what());
        }
    }
    std::printf("KO sign gate: %lu checks, %lu failures -> %s\n", checked, bad,
                (bad == 0 && checked > 0) ? "PASS" : "FAIL");
    return (bad == 0 && checked > 0) ? 0 : 1;
}
}  // namespace testKOSign

// Gate for the span-vectorized KO filters (KO{2,4,6,8}Simd): for every order
// the padding width supports, every bflag variant and block sizes
// n = 13 / 17 / 21, the simd path must agree with the explicit stencil path
// (uniform filter(), per-point filter_cako(), and ko_accumulate()) — bit for
// bit on boundary blocks (delegated) and to roundoff on interior blocks — and
// a clone must reproduce its source exactly. Whether the interior is also
// bit-identical is reported. Timings (stencil / simd / matrix) are printed
// for information only. Exit code 0 = pass.
namespace testKOSimd {
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

int run() {
    const unsigned int L = 1u << OCT_DIR_LEFT, R = 1u << OCT_DIR_RIGHT,
                       D = 1u << OCT_DIR_DOWN, U = 1u << OCT_DIR_UP,
                       B = 1u << OCT_DIR_BACK, F = 1u << OCT_DIR_FRONT;
    const std::vector<unsigned int> bflags = {
        0,     L, R, L | R, D,         U,
        D | U, B, F, B | F, L | D | B, L | R | D | U | B | F};
    unsigned long checked = 0, bad = 0;
    bool all_bitwise = true;

    for (const std::string order : {"KO2", "KO4", "KO6", "KO8"}) {
        for (unsigned int eo : {4u, 6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot         = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            const double dx = 0.05, dy = 0.07, dz = 0.11;

            std::unique_ptr<DendroDerivatives> ds, dv, dm;
            try {
                ds = make(order, eo);
                dv = make(order + "Simd", eo);
                dm = make(order + "Matrix", eo);
            } catch (const std::exception &e) {
                std::printf("  [skip] %s at eleorder %u: %s\n", order.c_str(),
                            eo, e.what());
                continue;
            }
            ds->set_maximum_block_size(tot);
            dv->set_maximum_block_size(tot);
            dm->set_maximum_block_size(tot);
            DendroDerivatives dc(*dv);  // clone, as the per-thread pool does
            dc.set_maximum_block_size(tot);

            std::vector<double> u(tot), coeff(tot), base(tot), wx(tot), wy(tot),
                wz(tot);
            for (size_t i = 0; i < tot; i++) {
                u[i]     = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
                coeff[i] = 0.1 + 0.05 * std::sin(0.2 * i);
                base[i]  = std::cos(0.1 * i);
            }

            double worst = 0.0, worst_interior_abs = 0.0;
            for (unsigned int bf : bflags) {
                // three entries: uniform filter, per-point cako, accumulate
                std::vector<double> rs[3] = {base, base, base},
                                    rv[3] = {base, base, base}, rc(base);
                ds->filter(u.data(), rs[0].data(), wx.data(), wy.data(),
                           wz.data(), dx, dy, dz, 0.1, sz, bf);
                dv->filter(u.data(), rv[0].data(), wx.data(), wy.data(),
                           wz.data(), dx, dy, dz, 0.1, sz, bf);
                ds->filter_cako(u.data(), rs[1].data(), wx.data(), wy.data(),
                                wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                dv->filter_cako(u.data(), rv[1].data(), wx.data(), wy.data(),
                                wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                rs[2] = rs[0];  // stencil has no accumulate: uniform filter is
                                // the reference
                const bool did = dv->ko_accumulate(rv[2].data(), u.data(), 0.1,
                                                   dx, dy, dz, sz, bf);
                dc.filter_cako(u.data(), rc.data(), wx.data(), wy.data(),
                               wz.data(), dx, dy, dz, coeff.data(), sz, bf);
                if (!did) {
                    bad++;
                    std::printf(
                        "  MISMATCH %sSimd n=%u bflag=%u ko_accumulate "
                        "declined\n",
                        order.c_str(), n, bf);
                }
                static const char *what[3] = {"filter", "cako", "accumulate"};
                for (int e = 0; e < 3; e++) {
                    double md = 0.0, scale = 0.0;
                    for (unsigned int k = pw; k < n - pw; k++)
                        for (unsigned int j = pw; j < n - pw; j++)
                            for (unsigned int i = pw; i < n - pw; i++) {
                                const size_t p = i + n * (j + n * k);
                                md = std::max(md,
                                              std::fabs(rv[e][p] - rs[e][p]));
                                scale = std::max(scale,
                                                 std::fabs(rs[e][p] - base[p]));
                            }
                    checked++;
                    const double rel = scale > 0 ? md / scale : md;
                    worst            = std::max(worst, rel);
                    if (bf == 0)
                        worst_interior_abs = std::max(worst_interior_abs, md);
                    const bool ok = bf ? (md == 0.0) : (rel <= REL_TOL);
                    if (!ok) {
                        bad++;
                        std::printf(
                            "  MISMATCH %sSimd n=%u bflag=%u %s rel=%g "
                            "abs=%g\n",
                            order.c_str(), n, bf, what[e], rel, md);
                    }
                }
                double mdc = 0.0;
                for (size_t p = 0; p < tot; p++)
                    mdc = std::max(mdc, std::fabs(rc[p] - rv[1][p]));
                checked++;
                if (mdc != 0.0) {
                    bad++;
                    std::printf(
                        "  MISMATCH %sSimd n=%u bflag=%u clone differs by %g\n",
                        order.c_str(), n, bf, mdc);
                }
            }
            if (worst_interior_abs != 0.0) all_bitwise = false;

            std::vector<double> o(tot);
            const double t_s = time_ns(
                [&]() {
                    ds->filter(u.data(), o.data(), wx.data(), wy.data(),
                               wz.data(), dx, dy, dz, 0.1, sz, 0);
                },
                3000);
            const double t_v = time_ns(
                [&]() {
                    dv->filter(u.data(), o.data(), wx.data(), wy.data(),
                               wz.data(), dx, dy, dz, 0.1, sz, 0);
                },
                3000);
            const double t_m = time_ns(
                [&]() {
                    dm->filter(u.data(), o.data(), wx.data(), wy.data(),
                               wz.data(), dx, dy, dz, 0.1, sz, 0);
                },
                3000);
            const double t_sc = time_ns(
                [&]() {
                    ds->filter_cako(u.data(), o.data(), wx.data(), wy.data(),
                                    wz.data(), dx, dy, dz, coeff.data(), sz, 0);
                },
                3000);
            const double t_vc = time_ns(
                [&]() {
                    dv->filter_cako(u.data(), o.data(), wx.data(), wy.data(),
                                    wz.data(), dx, dy, dz, coeff.data(), sz, 0);
                },
                3000);
            const double t_ma = time_ns(
                [&]() {
                    dm->ko_accumulate(o.data(), u.data(), 0.1, dx, dy, dz, sz,
                                      0);
                },
                3000);
            std::printf(
                "  %-4s n=%2u  worst rel %.1e  interior %s   filter: stencil "
                "%6.0f  simd %6.0f (%.2fx)  matrix %6.0f (%.2fx)  ko_acc %6.0f "
                "(%.2fx) | cako: stencil %6.0f  simd %6.0f (%.2fx) ns\n",
                order.c_str(), n, worst,
                worst_interior_abs == 0.0 ? "bitwise " : "roundoff", t_s, t_v,
                t_s / t_v, t_m, t_s / t_m, t_ma, t_s / t_ma, t_sc, t_vc,
                t_sc / t_vc);
        }
    }
    std::printf(
        "KO simd filters: %lu checks, %lu failures, interior %s -> %s\n",
        checked, bad,
        all_bitwise ? "bit-identical to the stencil"
                    : "within roundoff of the stencil",
        (bad == 0 && checked > 0) ? "PASS" : "FAIL");
    return (bad == 0 && checked > 0) ? 0 : 1;
}
}  // namespace testKOSimd

// Gate for DendroDerivatives::grad_mixed_set: the trimmed feeders (E*Simd) must
// match the grad_x/grad_y/grad_y/grad_z/grad_z chain byte for byte on every
// point the mixed chains and the fused kernel read. Timing is informational.
namespace testMixedSetTrim {
int run() {
    const unsigned int eo = 6, pw = 3, H = 3;
    const double dx = 0.05, dy = 0.07, dz = 0.11;
    unsigned long checked = 0, bad = 0;
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (const std::string eng : {"E6Simd", "E6"}) {
        for (unsigned int n : {13u, 19u, 25u, 31u, 15u, 37u}) {
            const size_t tot         = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            DendroDerivatives d(eng, eng, eo);
            d.set_maximum_block_size(tot);
            std::vector<double> u(tot);
            for (auto &v : u) v = dist(rng);
            const double nan = std::numeric_limits<double>::quiet_NaN();
            std::vector<std::vector<double>> r(5,
                                               std::vector<double>(tot, 7.0)),
                t(5, std::vector<double>(tot, nan));
            d.grad_x(r[3].data(), u.data(), dx, sz, 0);
            d.grad_y(r[4].data(), u.data(), dy, sz, 0);
            d.grad_y(r[0].data(), r[3].data(), dy, sz, 0);
            d.grad_z(r[1].data(), r[3].data(), dz, sz, 0);
            d.grad_z(r[2].data(), r[4].data(), dz, sz, 0);
            d.grad_mixed_set(t[0].data(), t[1].data(), t[2].data(), t[3].data(),
                             t[4].data(), u.data(), dx, dy, dz, sz, 0);
            // consumed box per output: {j0, j1, k0, k1}, x always active
            const unsigned int P = pw, a = pw - H, E = n - pw;
            struct Box {
                int out;
                unsigned int j0, j1, k0, k1;
            };
            const Box boxes[] = {{0, P, E, P, E},     {1, P, E, P, E},
                                 {2, P, E, P, E},     {3, a, n - a, P, E},
                                 {3, P, E, a, n - a}, {4, P, E, a, n - a}};
            unsigned long pts = 0, nbad = 0;
            for (const Box &b : boxes)
                for (unsigned int k = b.k0; k < b.k1; k++)
                    for (unsigned int j = b.j0; j < b.j1; j++)
                        for (unsigned int i = P; i < E; i++) {
                            const size_t p = i + n * (j + n * (size_t)k);
                            pts++;
                            if (std::memcmp(&r[b.out][p], &t[b.out][p],
                                            sizeof(double)))
                                nbad++;
                        }
            checked += pts;
            bad += nbad;
            auto once = [&](auto &&f) {
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 1000; i++) f();
                return std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - t0)
                           .count() /
                       1000;
            };
            auto chain = [&] {
                d.grad_x(r[3].data(), u.data(), dx, sz, 0);
                d.grad_y(r[4].data(), u.data(), dy, sz, 0);
                d.grad_y(r[0].data(), r[3].data(), dy, sz, 0);
                d.grad_z(r[1].data(), r[3].data(), dz, sz, 0);
                d.grad_z(r[2].data(), r[4].data(), dz, sz, 0);
            };
            auto trim = [&] {
                d.grad_mixed_set(t[0].data(), t[1].data(), t[2].data(),
                                 t[3].data(), t[4].data(), u.data(), dx, dy, dz,
                                 sz, 0);
            };
            double tc = 1e30, tt = 1e30;
            for (int rep = 0; rep < 15; rep++) {
                tc = std::min(tc, once(chain));
                tt = std::min(tt, once(trim));
            }
            std::printf(
                "  %-6s n=%2u  %6lu pts  %s  chain %.3f us  trimmed %.3f us  "
                "(%.2fx)\n",
                eng.c_str(), n, pts, nbad ? "MISMATCH" : "identical", tc, tt,
                tc / tt);
        }
    }
    std::printf("%lu points checked, %lu differ\n", checked, bad);
    return bad ? 1 : 0;
}
}  // namespace testMixedSetTrim

// KIMBYU_17 must throw on selection (its coefficients were a copy of
// KIMBYU_3); neighboring schemes must still build.
namespace testKimByu17 {
static bool builds(const char *name) {
    try {
        return DerivsFactory::create_first_order(name, 8, {}, 0, "none", {}) !=
               nullptr;
    } catch (const std::exception &e) {
        std::printf("%s: %s\n", name, e.what());
        return false;
    }
}

int run() {
    int bad = 0;
    if (builds("KIMBYU_17")) {
        std::printf("FAIL: KIMBYU_17 built\n");
        bad++;
    }
    for (const char *n : {"KIMBYU_3", "KIMBYU_16", "KIMBYU_18"})
        if (!builds(n)) {
            std::printf("FAIL: %s did not build\n", n);
            bad++;
        }
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
}  // namespace testKimByu17

namespace {
struct Section {
    const char *name;
    int (*run)();
};
const Section kSections[] = {
    {"testCloneNewSize", testCloneNewSize::run},
    {"testDerivSet", testDerivSet::run},
    {"testExplicitSimd", testExplicitSimd::run},
    {"testKOMatrix", testKOMatrix::run},
    {"testKOSign", testKOSign::run},
    {"testKOSimd", testKOSimd::run},
    {"testMixedSetTrim", testMixedSetTrim::run},
    {"testKimByu17", testKimByu17::run},
};
}  // namespace

int main(int argc, char **argv) {
    unsigned int ran = 0, failed = 0;
    for (const Section &s : kSections) {
        bool wanted = argc < 2;
        for (int a = 1; a < argc; a++)
            if (std::string(argv[a]) == s.name) wanted = true;
        if (!wanted) continue;
        int rc = 1;
        try {
            rc = s.run();
        } catch (const std::exception &e) {
            std::printf("%s: uncaught exception: %s\n", s.name, e.what());
        } catch (...) {
            std::printf("%s: uncaught exception\n", s.name);
        }
        std::printf("== section %s: %s\n", s.name, rc ? "FAIL" : "PASS");
        std::fflush(stdout);
        ran++;
        if (rc) failed++;
    }
    // a misspelled section name must not pass by running nothing
    std::printf("testDerivs: %u of %u sections failed, %s\n", failed, ran,
                (ran && !failed) ? "PASS" : "FAIL");
    return (ran && !failed) ? 0 : 1;
}
