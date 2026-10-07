// Gate for DendroDerivatives::grad_mixed_set: the trimmed feeders (E*Simd) must
// match the grad_x/grad_y/grad_y/grad_z/grad_z chain byte for byte on every
// point the mixed chains and the fused kernel read. Timing is informational.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "derivatives.h"

using namespace dendroderivs;

int main() {
    const unsigned int eo = 6, pw = 3, H = 3;
    const double dx = 0.05, dy = 0.07, dz = 0.11;
    unsigned long checked = 0, bad = 0;
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (const std::string eng : {"E6Simd", "E6"}) {
        for (unsigned int n : {13u, 19u, 25u, 31u, 15u, 37u}) {
            const size_t tot = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            DendroDerivatives d(eng, eng, eo);
            d.set_maximum_block_size(tot);
            std::vector<double> u(tot);
            for (auto &v : u) v = dist(rng);
            const double nan = std::numeric_limits<double>::quiet_NaN();
            std::vector<std::vector<double>> r(5, std::vector<double>(tot, 7.0)), t(5, std::vector<double>(tot, nan));
            d.grad_x(r[3].data(), u.data(), dx, sz, 0);
            d.grad_y(r[4].data(), u.data(), dy, sz, 0);
            d.grad_y(r[0].data(), r[3].data(), dy, sz, 0);
            d.grad_z(r[1].data(), r[3].data(), dz, sz, 0);
            d.grad_z(r[2].data(), r[4].data(), dz, sz, 0);
            d.grad_mixed_set(t[0].data(), t[1].data(), t[2].data(), t[3].data(), t[4].data(), u.data(), dx, dy, dz, sz, 0);
            // consumed box per output: {j0, j1, k0, k1}, x always active
            const unsigned int P = pw, a = pw - H, E = n - pw;
            struct Box { int out; unsigned int j0, j1, k0, k1; };
            const Box boxes[] = {{0, P, E, P, E}, {1, P, E, P, E}, {2, P, E, P, E},
                                 {3, a, n - a, P, E}, {3, P, E, a, n - a}, {4, P, E, a, n - a}};
            unsigned long pts = 0, nbad = 0;
            for (const Box &b : boxes)
                for (unsigned int k = b.k0; k < b.k1; k++)
                    for (unsigned int j = b.j0; j < b.j1; j++)
                        for (unsigned int i = P; i < E; i++) {
                            const size_t p = i + n * (j + n * (size_t)k);
                            pts++;
                            if (std::memcmp(&r[b.out][p], &t[b.out][p], sizeof(double))) nbad++;
                        }
            checked += pts; bad += nbad;
            auto once = [&](auto &&f) {
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 1000; i++) f();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 1000;
            };
            auto chain = [&] {
                d.grad_x(r[3].data(), u.data(), dx, sz, 0);
                d.grad_y(r[4].data(), u.data(), dy, sz, 0);
                d.grad_y(r[0].data(), r[3].data(), dy, sz, 0);
                d.grad_z(r[1].data(), r[3].data(), dz, sz, 0);
                d.grad_z(r[2].data(), r[4].data(), dz, sz, 0);
            };
            auto trim = [&] { d.grad_mixed_set(t[0].data(), t[1].data(), t[2].data(), t[3].data(), t[4].data(), u.data(), dx, dy, dz, sz, 0); };
            double tc = 1e30, tt = 1e30;
            for (int rep = 0; rep < 15; rep++) { tc = std::min(tc, once(chain)); tt = std::min(tt, once(trim)); }
            std::printf("  %-6s n=%2u  %6lu pts  %s  chain %.3f us  trimmed %.3f us  (%.2fx)\n", eng.c_str(), n, pts,
                        nbad ? "MISMATCH" : "identical", tc, tt, tc / tt);
        }
    }
    std::printf("%lu points checked, %lu differ\n", checked, bad);
    return bad ? 1 : 0;
}
