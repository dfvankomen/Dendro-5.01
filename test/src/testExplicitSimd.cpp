// Gate for the simd explicit engines (E4/E6/E8Simd): every grad_set mask
// against the loop engine of the same order, all bflag variants, block sizes
// the padding allows. Interior blocks must agree to roundoff; boundary blocks
// (delegated to the loop engine) must agree exactly. Timing is informational.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "derivatives.h"

using namespace dendroderivs;
static const double REL_TOL = 1e-13;

static double maxdiff_active(const double *a, const double *b, unsigned int n, unsigned int pw, double &scale) {
    double m = 0.0;
    for (unsigned int k = pw; k < n - pw; k++)
        for (unsigned int j = pw; j < n - pw; j++)
            for (unsigned int i = pw; i < n - pw; i++) {
                const size_t p = i + n * (j + n * k);
                m = std::max(m, std::fabs(a[p] - b[p])); scale = std::max(scale, std::fabs(b[p]));
            }
    return m;
}

int main() {
    const unsigned int L = 1u << OCT_DIR_LEFT, R = 1u << OCT_DIR_RIGHT, D = 1u << OCT_DIR_DOWN,
                       U = 1u << OCT_DIR_UP, B = 1u << OCT_DIR_BACK, F = 1u << OCT_DIR_FRONT;
    const std::vector<unsigned int> bflags = {0, L, R | U, D | B, L | R | D | U | B | F};
    unsigned long checked = 0, bad = 0;
    for (const std::string order : {"E4", "E6", "E8"}) {
        for (unsigned int eo : {4u, 6u, 8u, 10u}) {
            const unsigned int n = 2 * eo + 1, pw = eo / 2;
            const size_t tot = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            const double dx = 0.05, dy = 0.07, dz = 0.11;
            std::unique_ptr<DendroDerivatives> dl, ds;
            try {
                dl = std::make_unique<DendroDerivatives>(order, order, eo);
                ds = std::make_unique<DendroDerivatives>(order + "Simd", order + "Simd", eo);
            } catch (const std::exception &e) {
                std::printf("  [skip] %sSimd at eleorder %u: %s\n", order.c_str(), eo, e.what());
                continue;
            }
            dl->set_maximum_block_size(tot); ds->set_maximum_block_size(tot);
            DendroDerivatives dc(*ds);
            std::vector<double> u(tot), ws(tot);
            for (size_t i = 0; i < tot; i++) u[i] = std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i);
            std::vector<std::vector<double>> rl(9, std::vector<double>(tot)), rs(9, std::vector<double>(tot));
            double worst = 0.0;
            for (unsigned int bf : bflags) {
                for (unsigned int mask : {(unsigned)DendroDerivatives::DM_ALL, (unsigned)DendroDerivatives::DM_FIRST, (unsigned)DendroDerivatives::DM_MIXED, (unsigned)DendroDerivatives::DM_SECOND}) {
                    auto fill = [&](DendroDerivatives &d, std::vector<std::vector<double>> &r) {
                        for (auto &v : r) std::fill(v.begin(), v.end(), 0.0);
                        DendroDerivatives::DerivSet out;
                        double *s[9] = {r[0].data(), r[1].data(), r[2].data(), r[3].data(), r[4].data(), r[5].data(), r[6].data(), r[7].data(), r[8].data()};
                        out.x = s[0]; out.y = s[1]; out.z = s[2]; out.xx = s[3]; out.yy = s[4]; out.zz = s[5]; out.xy = s[6]; out.xz = s[7]; out.yz = s[8];
                        d.grad_set(out, u.data(), mask, dx, dy, dz, sz, bf, ws.data());
                    };
                    fill(*dl, rl); fill(dc, rs);
                    for (int b = 0; b < 9; b++) {
                        if (!(mask & (1u << b))) continue;
                        double scale = 0.0;
                        const double md = maxdiff_active(rs[b].data(), rl[b].data(), n, pw, scale);
                        const double rel = scale > 0 ? md / scale : md;
                        checked++;
                        const bool ok = bf ? (md == 0.0) : (rel <= REL_TOL);
                        worst = std::max(worst, rel);
                        if (!ok) { bad++; std::printf("  MISMATCH %sSimd n=%u bflag=%u mask=%u out=%d rel=%g\n", order.c_str(), n, bf, mask, b, rel); }
                    }
                }
            }
            auto tm = [&](DendroDerivatives &d) {
                std::vector<double> o(tot);
                for (int i = 0; i < 50; i++) d.grad_x_last(o.data(), u.data(), dx, sz, 0);
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 2000; i++) { d.grad_x_last(o.data(), u.data(), dx, sz, 0); d.grad_y_last(o.data(), u.data(), dy, sz, 0); d.grad_z(o.data(), u.data(), dz, sz, 0); }
                return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 2000;
            };
            std::printf("  %sSimd n=%2u  worst rel diff %.2e   x+y+z: loop %6.0f ns  simd %6.0f ns  (%.2fx)\n",
                        order.c_str(), n, worst, tm(*dl), tm(*ds), tm(*dl) / tm(*ds));
        }
    }
    std::printf("simd explicit engines: %lu checks, %lu failures -> %s\n", checked, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
