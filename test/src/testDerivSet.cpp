// Gate for DendroDerivatives::grad_set — the planned per-variable derivative
// entry point. For every one of the 512 masks, every bflag variant and the
// three engine families (matrix compact, explicit, CCFD), each requested
// output must match the corresponding individual grad_* call bit-for-bit on
// the active region. Also checks the null-output guard and the batch form.
//
// Exit code 0 = all identical, 1 = a mismatch (printed).
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "derivatives.h"

using namespace dendroderivs;

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

int main() {
    const unsigned int eo = 6, n = 2 * eo + 1, pw = eo / 2;
    const size_t tot = (size_t)n * n * n;
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
                out.x  = slots[0]; out.y  = slots[1]; out.z  = slots[2];
                out.xx = slots[3]; out.yy = slots[4]; out.zz = slots[5];
                out.xy = slots[6]; out.xz = slots[7]; out.yz = slots[8];
                dd.grad_set(out, u.data(), mask, dx, dy, dz, sz, bf, ws.data());
                static const char *names[9] = {"x",  "y",  "z",  "xx", "yy",
                                               "zz", "xy", "xz", "yz"};
                for (int b = 0; b < 9; b++) {
                    if (!(mask & (1u << b))) continue;
                    checked++;
                    const double md =
                        maxdiff_active(got[b].data(), ref[b].data(), n, pw);
                    if (md != 0.0) {
                        bad++;
                        if (bad <= 20)
                            std::printf("  MISMATCH %s/%s bflag=%u mask=%u %s "
                                        "maxdiff=%g\n",
                                        e.first.c_str(), e.second.c_str(), bf,
                                        mask, names[b], md);
                    }
                }
            }

            // batch form: two variables, same mask
            {
                std::vector<double> u2(tot);
                for (size_t i = 0; i < tot; i++) u2[i] = 0.5 * u[i] + 0.1;
                std::vector<double> o1(tot), o2(tot), r2(tot);
                DendroDerivatives::DerivSet outs[2];
                outs[0].x = o1.data();
                outs[1].x = o2.data();
                const double *us[2] = {u.data(), u2.data()};
                dd.grad_set_batch(outs, us, 2, DendroDerivatives::DM_X, dx, dy,
                                  dz, sz, bf);
                dd.grad_x_last(r2.data(), u2.data(), dx, sz, bf);
                checked += 2;
                if (maxdiff_active(o1.data(), ref[0].data(), n, pw) != 0.0 ||
                    maxdiff_active(o2.data(), r2.data(), n, pw) != 0.0) {
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
