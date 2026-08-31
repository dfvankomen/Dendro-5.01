// Sign gate for the Kreiss-Oliger filters: applied to the Nyquist mode
// u = (-1)^(i+j+k) with coeff = 1 on a block with no boundary flags, every
// KO order must return output += -(1/dx + 1/dy + 1/dz) * u on the active
// region — the same damping rate for every order by the KO normalization,
// and negative, i.e. dissipative. Found because the explicit KO8 stencil grew
// a seeded mode at exactly +0.3/step in EM4 (2026-08-29). Also reports, for
// information, whether an in-matrix filter changes the JTT6 operator at all.
// Exit code 0 = pass.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "derivatives.h"

extern "C" void dgeev_(const char *, const char *, const int *, double *, const int *, double *, double *, double *, const int *, double *, const int *, double *, const int *, int *);

using namespace dendroderivs;

int main() {
    unsigned long bad = 0, checked = 0;
    for (const std::string order : {"KO2", "KO4", "KO6", "KO8", "KO2Matrix", "KO4Matrix", "KO6Matrix", "KO8Matrix", "KO2Simd", "KO4Simd", "KO6Simd", "KO8Simd"}) {
        for (unsigned int eo : {6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            const double dx = 0.5, dy = 0.25, dz = 0.125;
            std::unique_ptr<DendroDerivatives> d;
            try {
                d = std::make_unique<DendroDerivatives>("E6", "E6", eo, std::vector<double>(), std::vector<double>(), 0u, 0u, "none", "none", std::vector<double>(), std::vector<double>(), order);
            } catch (const std::exception &) { continue; }
            d->set_maximum_block_size(tot);
            std::vector<double> u(tot), out(tot, 0.0), wx(tot), wy(tot), wz(tot);
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++) u[i + n * (j + n * k)] = ((i + j + k) & 1) ? -1.0 : 1.0;
            d->filter(u.data(), out.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 1.0, sz, 0);
            const double expect = -(1.0 / dx + 1.0 / dy + 1.0 / dz);
            double worst = 0.0;
            for (unsigned int k = pw; k < n - pw; k++)
                for (unsigned int j = pw; j < n - pw; j++)
                    for (unsigned int i = pw; i < n - pw; i++) {
                        const size_t p = i + n * (j + n * k);
                        worst = std::max(worst, std::fabs(out[p] / u[p] - expect));
                    }
            // boundary blocks: the one-sided layers have their own rates, so
            // gate on the sign alone — every active point must be damped
            std::vector<double> ob(tot, 0.0);
            const unsigned int all = (1u << OCT_DIR_LEFT) | (1u << OCT_DIR_RIGHT) | (1u << OCT_DIR_DOWN) | (1u << OCT_DIR_UP) | (1u << OCT_DIR_BACK) | (1u << OCT_DIR_FRONT);
            d->filter(u.data(), ob.data(), wx.data(), wy.data(), wz.data(), dx, dy, dz, 1.0, sz, all);
            unsigned long wrong = 0, zero = 0, pos = 0; double rmin = 1e300, rmax = -1e300; unsigned int imin = n, imax = 0;
            for (unsigned int k = pw; k < n - pw; k++)
                for (unsigned int j = pw; j < n - pw; j++)
                    for (unsigned int i = pw; i < n - pw; i++) {
                        const size_t p = i + n * (j + n * k);
                        const double r = ob[p] / u[p];
                        rmin = std::min(rmin, r); rmax = std::max(rmax, r);
                        if (r >= 0.0) { wrong++; if (r == 0.0) zero++; else pos++; imin = std::min(imin, i); imax = std::max(imax, i); }
                    }
            checked++;
            if (wrong) { bad++; std::printf("  %-10s n=%2u  bflag=ALL: %lu active points NOT damped (%lu zero, %lu positive; out/u in [%+.3f, %+.3f]; i in [%u,%u])\n", order.c_str(), n, wrong, zero, pos, rmin, rmax, imin, imax); }
            checked++;
            const bool ok = worst < 1e-10 * std::fabs(expect);
            if (!ok) bad++;
            std::printf("  %-10s n=%2u  out/u on Nyquist = %+.6f  expected %+.6f  -> %s\n", order.c_str(), n, out[(pw) + n * (pw + n * pw)] / u[(pw) + n * (pw + n * pw)], expect, ok ? "ok" : "WRONG SIGN / RATE");
        }
    }
    // KIMF: gate on the energy form + spectrum and report the transfer
    // (its Nyquist rate is 1 - G(pi), not the KO normalization)
    for (unsigned int eo : {6u, 10u}) {
        const unsigned int n = 2 * eo + 1, pw = eo / 2;
        const size_t tot = (size_t)n * n * n;
        const unsigned int sz[3] = {n, n, n};
        std::unique_ptr<DendroDerivatives> d;
        try {
            d = std::make_unique<DendroDerivatives>("E6", "E6", eo, std::vector<double>(), std::vector<double>(), 0u, 0u, "none", "none", std::vector<double>(), std::vector<double>(), "KIMF");
        } catch (const std::exception &e) {
            std::printf("  [skip] KIMF at eleorder %u: %s\n", eo, e.what()); continue;
        }
        d->set_maximum_block_size(tot);
        std::vector<double> u(tot), out(tot), wx(tot), wy(tot), wz(tot);
        const double dx = 1.0;
        auto response = [&](int period) {  // out/u with coeff 1, dx=dy=dz=1
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++)
                        u[i + n * (j + (size_t)n * k)] = std::cos(M_PI * 2.0 * (double)(i + j + k) / period);
            std::fill(out.begin(), out.end(), 0.0);
            d->filter(u.data(), out.data(), wx.data(), wy.data(), wz.data(), dx, dx, dx, 1.0, sz, 0);
            const size_t c = (n / 2) + n * ((n / 2) + (size_t)n * (n / 2));
            return out[c] / u[c];
        };
        const double rN = response(2), r4 = response(4);
        // energy gate <u, Du> <= 0: R^-1 spreads the boundary rows, so an NSD
        // operator may still respond positively at single points -- pointwise
        // sign is the wrong test for an implicit filter
        unsigned long wrong = 0; double worst_pos = 0.0; unsigned int wpi = 0; bool wpb = false;
        for (unsigned int bfsel = 0; bfsel < 2; bfsel++) {
            const unsigned int bf = bfsel ? ((1u << OCT_DIR_LEFT) | (1u << OCT_DIR_RIGHT) | (1u << OCT_DIR_DOWN) | (1u << OCT_DIR_UP) | (1u << OCT_DIR_BACK) | (1u << OCT_DIR_FRONT)) : 0u;
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++)
                        u[i + n * (j + (size_t)n * k)] = ((i + j + k) & 1) ? -1.0 : 1.0;
            std::fill(out.begin(), out.end(), 0.0);
            d->filter(u.data(), out.data(), wx.data(), wy.data(), wz.data(), dx, dx, dx, 1.0, sz, bf);
            double energy = 0.0;
            for (unsigned int k = pw; k < n - pw; k++)
                for (unsigned int j = pw; j < n - pw; j++)
                    for (unsigned int i = pw; i < n - pw; i++) {
                        const size_t p = i + n * (j + (size_t)n * k);
                        energy += u[p] * out[p];
                        const double r = out[p] / u[p];
                        if (r > worst_pos) { worst_pos = r; wpi = i; wpb = bfsel; }
                    }
            checked++;
            if (energy > 0.0) { wrong++; bad++; std::printf("  KIMF n=%u bflag=%s: <u, Du> = %+.3e > 0 on the Nyquist mode\n", n, bfsel ? "ALL" : "0", energy); }
        }
        if (worst_pos > 0.0)
            std::printf("  KIMF n=%2u  info: largest positive pointwise response %+.3e at i=%u (%s block) -- NSD operator, see spectrum\n",
                        n, worst_pos, wpi, wpb ? "all-faces" : "interior");
        std::printf("  KIMF n=%2u  response per axis (coeff 1, h 1): Nyquist %+.4f (KO: -1.0000)  4dx %+.6f (KO4: -0.1250)  ratio %.0fx gentler\n",
                    n, rN / 3.0, r4 / 3.0, (0.125) / std::max(1e-12, std::fabs(r4 / 3.0)));
        {
            // speed: must match matrix KO (same three GEMMs)
            auto dm = std::make_unique<DendroDerivatives>("E6", "E6", eo, std::vector<double>(), std::vector<double>(), 0u, 0u, "none", "none", std::vector<double>(), std::vector<double>(), "KO4Matrix");
            dm->set_maximum_block_size(tot);
            std::vector<double> rhs(tot, 0.0);
            auto tns = [&](DendroDerivatives &dd) {
                for (int i = 0; i < 100; i++) dd.ko_accumulate(rhs.data(), u.data(), 0.4, 0.1, 0.1, 0.1, sz, 0);
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 2000; i++) dd.ko_accumulate(rhs.data(), u.data(), 0.4, 0.1, 0.1, 0.1, sz, 0);
                return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 2000;
            };
            std::printf("  KIMF n=%2u  ko_accumulate: %6.0f ns  (KO4Matrix: %6.0f ns)\n", n, tns(*d), tns(*dm));
        }
    }

    // spectral gate: with both faces of an axis physical, the 1-D operator on
    // the active points is square (no padding is read); probe it column by
    // column with unit vectors constant along the other axes (their stencils
    // vanish exactly) and require every eigenvalue to have Re <= 0
    for (const std::string order : {"KO2", "KO4", "KO6", "KO8", "KIMF"}) {
        for (unsigned int eo : {6u, 8u, 10u, 12u, 16u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2, na = n - 2 * pw;
            const size_t tot = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            std::unique_ptr<DendroDerivatives> d;
            try {
                d = std::make_unique<DendroDerivatives>("E6", "E6", eo, std::vector<double>(), std::vector<double>(), 0u, 0u, "none", "none", std::vector<double>(), std::vector<double>(), order);
            } catch (const std::exception &) { continue; }
            d->set_maximum_block_size(tot);
            const unsigned int lo[3] = {1u << OCT_DIR_LEFT, 1u << OCT_DIR_DOWN, 1u << OCT_DIR_BACK};
            const unsigned int hi[3] = {1u << OCT_DIR_RIGHT, 1u << OCT_DIR_UP, 1u << OCT_DIR_FRONT};
            for (int axis = 0; axis < 3; axis++) {
                std::vector<double> e(tot), out(tot), wx(tot), wy(tot), wz(tot), A((size_t)na * na, 0.0);
                auto idx = [&](unsigned int a, unsigned int b, unsigned int c) { unsigned int q[3]; q[axis] = a; q[(axis + 1) % 3] = b; q[(axis + 2) % 3] = c; return (size_t)q[0] + n * (q[1] + (size_t)n * q[2]); };
                for (unsigned int c = 0; c < na; c++) {
                    std::fill(e.begin(), e.end(), 0.0); std::fill(out.begin(), out.end(), 0.0);
                    for (unsigned int b = 0; b < n; b++) for (unsigned int cc = 0; cc < n; cc++) e[idx(pw + c, b, cc)] = 1.0;
                    d->filter(e.data(), out.data(), wx.data(), wy.data(), wz.data(), 1.0, 1.0, 1.0, 1.0, sz, lo[axis] | hi[axis]);
                    for (unsigned int r = 0; r < na; r++) A[r + (size_t)na * c] = out[idx(pw + r, pw, pw)];  // column-major
                }
                int N = na, lwork = 8 * N, info = 0, one = 1;
                std::vector<double> wr(N), wi(N), work(lwork);
                dgeev_("N", "N", &N, A.data(), &N, wr.data(), wi.data(), nullptr, &one, nullptr, &one, work.data(), &lwork, &info);
                double remax = -1e300, amax = 0.0;
                for (int i = 0; i < N; i++) { remax = std::max(remax, wr[i]); amax = std::max(amax, std::hypot(wr[i], wi[i])); }
                checked++;
                const bool ok = info == 0 && remax <= 1e-10 * amax;
                if (!ok) { bad++; std::printf("  %-4s n=%2u axis %d both faces closed: max Re(eig) = %+.3e (|eig|max %.3e) -> NOT dissipative\n", order.c_str(), n, axis, remax, amax); }
                else if (axis == 0) std::printf("  %-4s n=%2u  both faces closed: max Re(eig) = %+.2e of |eig|max %.2e -> dissipative\n", order.c_str(), n, remax, amax);
            }
        }
    }

    // info: does an in-matrix filter change the first-derivative operator?
    for (const std::string f : {"KIM", "BYUT6"}) {
        const unsigned int eo = 10, n = 21; const size_t tot = (size_t)n * n * n; const unsigned int sz[3] = {n, n, n};
        try {
            DendroDerivatives a("JTT6", "E6", eo, std::vector<double>(), std::vector<double>(), 0u, 0u, "none", "none", std::vector<double>(), std::vector<double>(), "KO4");
            DendroDerivatives b("JTT6", "E6", eo, std::vector<double>(), std::vector<double>(), 0u, 0u, f, "none", std::vector<double>{0.5}, std::vector<double>(), "KO4");
            a.set_maximum_block_size(tot); b.set_maximum_block_size(tot);
            std::vector<double> u(tot), oa(tot), ob(tot);
            for (size_t i = 0; i < tot; i++) u[i] = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
            a.grad_x(oa.data(), u.data(), 0.1, sz, 0); b.grad_x(ob.data(), u.data(), 0.1, sz, 0);
            double md = 0.0, sc = 0.0;
            for (size_t i = 0; i < tot; i++) { md = std::max(md, std::fabs(oa[i] - ob[i])); sc = std::max(sc, std::fabs(oa[i])); }
            std::printf("  info: JTT6 grad_x with in-matrix %s (coeff 0.5) vs none: max rel diff %.3e  [%s]\n", f.c_str(), md / sc, b.toString().c_str());
        } catch (const std::exception &e) { std::printf("  info: in-matrix %s: %s\n", f.c_str(), e.what()); }
    }
    std::printf("KO sign gate: %lu checks, %lu failures -> %s\n", checked, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
