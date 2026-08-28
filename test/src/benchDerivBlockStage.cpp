// Per-block derivative-stage benchmark on the real BSSN call pattern:
// 24 evolved variables, the first 11 (alpha, chi, beta, gt) also needing
// pure second and mixed derivatives — 138 grad_* calls per block, every
// output in its own block-sized array (as the RHS deriv workspace is laid
// out), so cache behaviour is realistic.
//
// Three ways of issuing the same work through DendroDerivatives:
//   plain    — x, xx, y, yy, z, zz, then mixed chained on grad_x/grad_y
//              (what BSSN_GR/scripts/bssnrhs_derivs.h does today)
//   _last    — the same list with terminal calls routed to the _last API
//   grad_set — one planned call per variable with its derivative mask
// All three must agree bit-for-bit on the active region (exit code 1 if
// not). Also reports the KO filter cost on the same block and per-thread
// scaling with one facade clone per OpenMP thread.
//
// usage: benchDerivBlockStage [scheme1=JTT6] [scheme2=JTT6] [eleorder=6] [iters=2000]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <omp.h>

#include "derivatives.h"
#include "derivatives/derivs_utils.h"

using namespace dendroderivs;

static const int NV = 24, NV2 = 11;

struct BlockWork {
    std::vector<std::vector<double>> u;  // NV inputs
    std::vector<std::vector<double>> d;  // 3*NV first, 3*NV2 second, 3*NV2 mixed
    explicit BlockWork(size_t tot)
        : u(NV, std::vector<double>(tot)),
          d(3 * NV + 6 * NV2, std::vector<double>(tot, 0.0)) {
        for (int v = 0; v < NV; v++)
            for (size_t i = 0; i < tot; i++)
                u[v][i] = std::sin(0.37 * i + 0.1 * v) + 0.01 * v;
    }
    double *first(int v, int a) { return d[3 * v + a].data(); }
    double *second(int v, int a) { return d[3 * NV + 3 * v + a].data(); }
    double *mixed(int v, int a) { return d[3 * NV + 3 * NV2 + 3 * v + a].data(); }
};

static double maxdiff_active(const double *a, const double *b, unsigned int n,
                             unsigned int pw) {
    double m = 0.0;
    for (unsigned int k = pw; k < n - pw; k++)
        for (unsigned int j = pw; j < n - pw; j++)
            for (unsigned int i = pw; i < n - pw; i++)
                m = std::max(m, std::fabs(a[i + n * (j + n * k)] -
                                          b[i + n * (j + n * k)]));
    return m;
}

enum Mode { PLAIN, LAST, SET };

static void block_stage(DendroDerivatives &D, BlockWork &w, Mode mode,
                        double dx, const unsigned int *sz, unsigned int bf) {
    for (int v = 0; v < NV; v++) {
        const bool second = v < NV2;
        const double *uu  = w.u[v].data();
        double *dxv = w.first(v, 0), *dyv = w.first(v, 1), *dzv = w.first(v, 2);
        if (mode == SET) {
            DendroDerivatives::DerivSet out;
            out.x = dxv; out.y = dyv; out.z = dzv;
            unsigned int mask = DendroDerivatives::DM_FIRST;
            if (second) {
                out.xx = w.second(v, 0); out.yy = w.second(v, 1); out.zz = w.second(v, 2);
                out.xy = w.mixed(v, 0);  out.xz = w.mixed(v, 1);  out.yz = w.mixed(v, 2);
                mask = DendroDerivatives::DM_ALL;
            }
            D.grad_set(out, uu, mask, dx, dx, dx, sz, bf);
            continue;
        }
        const bool last = (mode == LAST);
        // x, xx, y, yy, z, zz, then the chained mixed — the generated order
        if (second || !last) D.grad_x(dxv, uu, dx, sz, bf); else D.grad_x_last(dxv, uu, dx, sz, bf);
        if (second) { if (last) D.grad_xx_last(w.second(v, 0), uu, dx, sz, bf); else D.grad_xx(w.second(v, 0), uu, dx, sz, bf); }
        if (second || !last) D.grad_y(dyv, uu, dx, sz, bf); else D.grad_y_last(dyv, uu, dx, sz, bf);
        if (second) { if (last) D.grad_yy_last(w.second(v, 1), uu, dx, sz, bf); else D.grad_yy(w.second(v, 1), uu, dx, sz, bf); }
        D.grad_z(dzv, uu, dx, sz, bf);
        if (second) D.grad_zz(w.second(v, 2), uu, dx, sz, bf);
        if (second) {
            if (last) D.grad_y_last(w.mixed(v, 0), dxv, dx, sz, bf); else D.grad_y(w.mixed(v, 0), dxv, dx, sz, bf);
            D.grad_z(w.mixed(v, 1), dxv, dx, sz, bf);
            D.grad_z(w.mixed(v, 2), dyv, dx, sz, bf);
        }
    }
}

template <typename Fn>
static double time_us(Fn &&fn, unsigned int iters) {
    for (unsigned int i = 0; i < 20; i++) fn();
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < iters; i++) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

int main(int argc, char **argv) {
    const std::string s1 = argc > 1 ? argv[1] : "JTT6";
    const std::string s2 = argc > 2 ? argv[2] : "JTT6";
    const unsigned int eo = argc > 3 ? std::atoi(argv[3]) : 6;
    const unsigned int iters = argc > 4 ? std::atoi(argv[4]) : 2000;
    const unsigned int n = 2 * eo + 1, pw = eo / 2;
    const size_t tot = (size_t)n * n * n;
    const unsigned int sz[3] = {n, n, n};
    const double dx = 0.05;
    const unsigned int bf = 0;

    DendroDerivatives dd(s1, s2, eo);
    dd.set_maximum_block_size(tot);
    prewarm_kernel_cache({{n, n, n}}, pw);
    std::printf("block stage: %s, n=%u pw=%u, %d vars (%d with 2nd+mixed), %u iters\n",
                dd.toString().c_str(), n, pw, NV, NV2, iters);

    BlockWork ref(tot), wk(tot);
    block_stage(dd, ref, PLAIN, dx, sz, bf);
    auto check = [&](BlockWork &w) {
        double m = 0.0;
        for (size_t a = 0; a < w.d.size(); a++)
            m = std::max(m, maxdiff_active(w.d[a].data(), ref.d[a].data(), n, pw));
        return m;
    };

    const double t_plain = time_us([&]() { block_stage(dd, wk, PLAIN, dx, sz, bf); }, iters);
    const double t_last  = time_us([&]() { block_stage(dd, wk, LAST, dx, sz, bf); }, iters);
    const double md_last = check(wk);
    const double t_set   = time_us([&]() { block_stage(dd, wk, SET, dx, sz, bf); }, iters);
    const double md_set  = check(wk);
    std::printf("  plain (as generated today) : %8.1f us/block\n", t_plain);
    std::printf("  terminal calls via _last   : %8.1f us/block  (%.2fx)  maxdiff=%g\n", t_last, t_plain / t_last, md_last);
    std::printf("  grad_set per variable      : %8.1f us/block  (%.2fx)  maxdiff=%g\n", t_set, t_plain / t_set, md_set);

    {
        std::vector<double> out(tot), wx(tot), wy(tot), wz(tot), coeff(tot, 0.1);
        const double t_ko = time_us([&]() {
            dd.filter_cako(wk.u[0].data(), out.data(), wx.data(), wy.data(), wz.data(), dx, dx, dx, coeff.data(), sz, bf);
        }, iters);
        std::printf("  KO filter_cako             : %8.2f us/call  (x%d = %.1f us/block)\n", t_ko, NV, NV * t_ko);
    }

    std::printf("OMP scaling, one facade clone + workspace per thread (us/block/thread):\n");
    for (int nt : {1, 2, 4, 8}) {
        if (nt > omp_get_max_threads()) break;
        std::vector<double> tp(nt), ts(nt);
#pragma omp parallel num_threads(nt)
        {
            const int t = omp_get_thread_num();
            DendroDerivatives dl(dd);
            dl.set_maximum_block_size(tot);
            BlockWork w(tot);
            const unsigned int it = iters / 2;
            for (unsigned int i = 0; i < 20; i++) block_stage(dl, w, PLAIN, dx, sz, bf);
#pragma omp barrier
            double t0 = omp_get_wtime();
            for (unsigned int i = 0; i < it; i++) block_stage(dl, w, PLAIN, dx, sz, bf);
            tp[t] = (omp_get_wtime() - t0) * 1e6 / it;
#pragma omp barrier
            for (unsigned int i = 0; i < 20; i++) block_stage(dl, w, SET, dx, sz, bf);
#pragma omp barrier
            t0 = omp_get_wtime();
            for (unsigned int i = 0; i < it; i++) block_stage(dl, w, SET, dx, sz, bf);
            ts[t] = (omp_get_wtime() - t0) * 1e6 / it;
        }
        double mp = 0, ms = 0;
        for (int t = 0; t < nt; t++) { mp = std::max(mp, tp[t]); ms = std::max(ms, ts[t]); }
        std::printf("  threads=%d  plain %.1f  grad_set %.1f  (%.2fx)\n", nt, mp, ms, mp / ms);
    }

    const bool ok = (md_last == 0.0 && md_set == 0.0);
    std::printf("%s\n", ok ? "PASS — all three issue paths bit-identical on the active region"
                           : "FAIL — issue paths differ");
    return ok ? 0 : 1;
}
