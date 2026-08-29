// Gate for engine copies meeting a block size they have never seen: a remesh
// can introduce one after the per-thread clones were made, and the clone must
// build its own D storage (filter included) and match the original bit for bit.
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "derivatives.h"

using namespace dendroderivs;

int main() {
    struct Cfg { std::string d1, d2, filt; };
    const std::vector<Cfg> cfgs = {{"JTT6", "JTT6", "none"}, {"JTP6", "JTP6", "none"},
                                   {"JTT6", "JTT6", "KIM"},
                                   {"E6", "E6", "none"},     {"E6Simd", "E6Simd", "none"}};
    const unsigned int eo = 6, pw = 3;
    unsigned long checked = 0, bad = 0;
    for (const Cfg &c : cfgs) {
        std::unique_ptr<DendroDerivatives> orig;
        try {
            orig = std::make_unique<DendroDerivatives>(c.d1, c.d2, eo, std::vector<double>(),
                                                       std::vector<double>(), 0, 0, c.filt, c.filt);
        } catch (const std::exception &e) {
            std::printf("  [skip] %s/%s filter %s: %s\n", c.d1.c_str(), c.d2.c_str(), c.filt.c_str(), e.what());
            continue;
        }
        orig->set_maximum_block_size(13 * 13 * 13);
        orig->pre_create_for_size(13);
        DendroDerivatives lazy(*orig);       // used on the new size with no warm-up
        DendroDerivatives warmed(*orig);     // pre_create_for_size first, as the solver does
        DendroDerivatives twice(lazy);       // copy of a copy
        // init() builds n = 13..55 eagerly (DDERIVS_MAX_BLOCKS_INIT); these are not in the map
        for (unsigned int n : {61u, 67u}) {
            const size_t tot = (size_t)n * n * n;
            const unsigned int sz[3] = {n, n, n};
            std::vector<double> u(tot), ws(tot);
            for (size_t i = 0; i < tot; i++) u[i] = std::sin(0.31 * i) + 0.2 * std::cos(0.007 * i * i);
            for (DendroDerivatives *d : {orig.get(), &lazy, &warmed, &twice}) d->set_maximum_block_size(tot);
            warmed.pre_create_for_size(n);
            auto run = [&](DendroDerivatives &d, std::vector<std::vector<double>> &r) {
                DendroDerivatives::DerivSet out;
                out.x = r[0].data(); out.y = r[1].data(); out.z = r[2].data();
                out.xx = r[3].data(); out.yy = r[4].data(); out.zz = r[5].data();
                out.xy = r[6].data(); out.xz = r[7].data(); out.yz = r[8].data();
                d.grad_set(out, u.data(), DendroDerivatives::DM_ALL, 0.05, 0.07, 0.11, sz, 0, ws.data());
            };
            std::vector<std::vector<double>> ro(9, std::vector<double>(tot)), rc(9, std::vector<double>(tot));
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
                                if (ro[b][p] != rc[b][p]) { same = false; break; }
                            }
                    checked++;
                    if (!same) { bad++; std::printf("  MISMATCH %s/%s filter %s n=%u out=%d\n", c.d1.c_str(), c.d2.c_str(), c.filt.c_str(), n, b); }
                }
            }
        }
        std::printf("  %s/%s filter %-6s: copies agree on unseen sizes 61, 67\n", c.d1.c_str(), c.d2.c_str(), c.filt.c_str());
    }
    std::printf("clone-then-new-size: %lu checks, %lu failures -> %s\n", checked, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
