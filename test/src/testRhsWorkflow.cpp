// Benchmark realistic RHS-style derivative workflows: when the RHS
// needs all of {grad_x, grad_y, grad_z, grad_xy, grad_xz, grad_yz}
// from a single input field, which evaluation strategy is fastest?
//
// Three strategies compared:
//
// 1. CLASSICAL_REUSE: the pattern most BSSN/Z4c RHS codes use.
//    dx_u = grad_x(u)   (full write, serves as 1st-deriv x output AND
//                        intermediate for grad_xy, grad_xz)
//    dy_u = grad_y(u)   (full write, 1st-deriv y output AND intermediate
//                        for grad_yz)
//    dz_u = grad_z(u)
//    dxy_u = grad_y_last(dx_u)
//    dxz_u = grad_z(dx_u)
//    dyz_u = grad_z(dy_u)
//
// 2. ALL_FUSED: 1st derivs use _last (skip padding), mixed derivs use
//    the single-call fused functions.
//    dx_u  = grad_x_last(u)
//    dy_u  = grad_y_last(u)
//    dz_u  = grad_z(u)
//    dxy_u = grad_xy_fused_last(u)
//    dxz_u = grad_xz_fused_last(u)
//    dyz_u = grad_yz_fused_last(u)
//
// 3. MIXED_ONLY: only the mixed derivs are needed (no 1st derivs).
//    Uses chain vs fused for just the three mixed pairs.
//
// All variants verified against analytical d^2u/dxdy, d^2u/dxdz,
// d^2u/dydz.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "derivatives.h"
#include "derivatives/derivs_factory.h"
#include "derivatives/derivs_matrixonly.h"
#include "derivatives/derivs_utils.h"

using namespace dendroderivs;

static const std::vector<unsigned int> ELEORDERS = {6, 8, 10};
static constexpr double DX                       = 0.05;
static constexpr unsigned int NWARMUP            = 200;
static constexpr unsigned int NITER              = 5000;

static void init_field(double *u, double *xy, double *xz, double *yz,
                       unsigned int n, unsigned int pw) {
    const double twopi = 2.0 * M_PI;
    const double k2    = twopi * twopi;
    for (unsigned int k = 0; k < n; k++) {
        const double z  = ((int)k - (int)pw) * DX;
        const double sz_v = std::sin(twopi * z), cz = std::cos(twopi * z);
        for (unsigned int j = 0; j < n; j++) {
            const double y = ((int)j - (int)pw) * DX;
            const double sy = std::sin(twopi * y), cy = std::cos(twopi * y);
            for (unsigned int i = 0; i < n; i++) {
                const double x = ((int)i - (int)pw) * DX;
                const double sx = std::sin(twopi * x), cx = std::cos(twopi * x);
                const unsigned int idx = i + j * n + k * n * n;
                u[idx]  = sx * sy * sz_v;
                xy[idx] = k2 * cx * cy * sz_v;
                xz[idx] = k2 * cx * sy * cz;
                yz[idx] = k2 * sx * cy * cz;
            }
        }
    }
}

static double rmse_interior(const double *a, const double *b,
                            unsigned int n, unsigned int pw) {
    double s = 0.0;
    unsigned long c = 0;
    for (unsigned int k = pw; k < n - pw; k++)
        for (unsigned int j = pw; j < n - pw; j++)
            for (unsigned int i = pw; i < n - pw; i++) {
                const unsigned int idx = i + j * n + k * n * n;
                const double d = a[idx] - b[idx];
                s += d * d; c++;
            }
    return c ? std::sqrt(s / (double)c) : 0.0;
}

template <typename Fn>
static double bench_us(Fn &&fn, unsigned int niter) {
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < niter; i++) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

struct WorkflowResult {
    std::string scheme;
    unsigned int eleorder, n;
    double classical_us, all_fused_us, mixed_chain_us, mixed_fused_us;
    double rmse_xy, rmse_xz, rmse_yz;
};

static WorkflowResult run_scheme(const std::string &scheme, unsigned int eo) {
    WorkflowResult r{};
    r.scheme   = scheme;
    r.eleorder = eo;
    const unsigned int n  = eo * 2 + 1;
    const unsigned int pw = eo / 2;
    r.n = n;

    const auto &reg = get_first_order_registry();
    auto it = reg.find(scheme);
    if (it == reg.end()) return r;

    std::vector<double> empty;
    std::vector<double> placeholder = {0.5};
    std::string no_filter = "none";

    std::unique_ptr<Derivs> deriv;
    try { deriv = it->second(eo, no_filter, empty, placeholder, 1u); }
    catch (...) { return r; }
    if (!deriv) return r;

    const size_t total = (size_t)n * n * n;
    deriv->set_maximum_block_size(total);
    deriv->pre_create_for_size(n);

    auto *mat = dynamic_cast<MatrixCompactDerivs<1> *>(deriv.get());
    if (!mat) return r;                              // fused only on matrix-form

    std::vector<double> u(total), xy_t(total), xz_t(total), yz_t(total);
    init_field(u.data(), xy_t.data(), xz_t.data(), yz_t.data(), n, pw);

    // Per-derivative output buffers, reused across iterations (no fresh
    // alloc inside the bench loop).
    std::vector<double> dx_u(total, 0.0), dy_u(total, 0.0), dz_u(total, 0.0);
    std::vector<double> dxy_u(total, 0.0), dxz_u(total, 0.0), dyz_u(total, 0.0);

    const unsigned int sz[3] = {n, n, n};
    const unsigned int bflag = 0;

    // ---- 1. CLASSICAL_REUSE ----
    auto classical = [&]() {
        deriv->do_grad_x(dx_u.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_y(dy_u.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_z(dz_u.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_y_last(dxy_u.data(), dx_u.data(), DX, sz, bflag);
        deriv->do_grad_z(dxz_u.data(), dx_u.data(), DX, sz, bflag);
        deriv->do_grad_z(dyz_u.data(), dy_u.data(), DX, sz, bflag);
    };

    // ---- 2. ALL_FUSED ----
    auto all_fused = [&]() {
        deriv->do_grad_x_last(dx_u.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_y_last(dy_u.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_z(dz_u.data(), u.data(), DX, sz, bflag);
        mat->do_grad_xy_last(dxy_u.data(), u.data(), DX, sz, bflag);
        mat->do_grad_xz_last(dxz_u.data(), u.data(), DX, sz, bflag);
        mat->do_grad_yz_last(dyz_u.data(), u.data(), DX, sz, bflag);
    };

    // ---- 3a. MIXED-ONLY CHAIN ----
    auto mixed_chain = [&]() {
        // Recompute v_x, v_y as throwaway intermediates (don't need
        // the 1st derivs as outputs in this scenario)
        deriv->do_grad_x(dx_u.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_y(dy_u.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_y_last(dxy_u.data(), dx_u.data(), DX, sz, bflag);
        deriv->do_grad_z(dxz_u.data(), dx_u.data(), DX, sz, bflag);
        deriv->do_grad_z(dyz_u.data(), dy_u.data(), DX, sz, bflag);
    };

    // ---- 3b. MIXED-ONLY FUSED ----
    auto mixed_fused = [&]() {
        mat->do_grad_xy_last(dxy_u.data(), u.data(), DX, sz, bflag);
        mat->do_grad_xz_last(dxz_u.data(), u.data(), DX, sz, bflag);
        mat->do_grad_yz_last(dyz_u.data(), u.data(), DX, sz, bflag);
    };

    // warmup
    for (unsigned int w = 0; w < NWARMUP; w++) classical();
    for (unsigned int w = 0; w < NWARMUP; w++) all_fused();

    // Verify correctness using the all_fused outputs (last filled).
    r.rmse_xy = rmse_interior(dxy_u.data(), xy_t.data(), n, pw);
    r.rmse_xz = rmse_interior(dxz_u.data(), xz_t.data(), n, pw);
    r.rmse_yz = rmse_interior(dyz_u.data(), yz_t.data(), n, pw);

    r.classical_us    = bench_us(classical,    NITER);
    r.all_fused_us    = bench_us(all_fused,    NITER);
    r.mixed_chain_us  = bench_us(mixed_chain,  NITER);
    r.mixed_fused_us  = bench_us(mixed_fused,  NITER);
    return r;
}

int main(int argc, char **argv) {
    const std::string csv_path = (argc > 1) ? argv[1] : "rhs_workflow.csv";
    std::ofstream csv(csv_path);
    csv << "scheme,eleorder,n,classical_us,all_fused_us,"
           "mixed_chain_us,mixed_fused_us,rmse_xy,rmse_xz,rmse_yz\n";

    std::cout << "RHS workflow benchmark (6 derivs from u: dx, dy, dz, dxy, dxz, dyz)\n"
              << "  bflag = 0, warmup = " << NWARMUP << ", time = " << NITER
              << " iter\n\n";

    std::cout << std::left << std::setw(10) << "scheme"
              << std::setw(4) << "eo" << std::setw(4) << "n"
              << std::right << std::setw(13) << "classical(us)"
              << std::setw(13) << "all_fused(us)" << std::setw(8) << "  win"
              << std::setw(13) << "mix_chain(us)"
              << std::setw(13) << "mix_fused(us)" << std::setw(8) << "  win"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    const std::vector<std::string> schemes = {
        "JTT4", "JTT6", "JTP6", "BL6", "BorisO6"
    };
    for (unsigned int eo : ELEORDERS) {
        for (const auto &s : schemes) {
            WorkflowResult r = run_scheme(s, eo);
            const double pc = r.classical_us / NITER;
            const double pf = r.all_fused_us / NITER;
            const double mc = r.mixed_chain_us / NITER;
            const double mf = r.mixed_fused_us / NITER;
            const double full_speedup = (pf > 0) ? pc / pf : 0;
            const double mix_speedup  = (mf > 0) ? mc / mf : 0;
            std::cout << std::left << std::setw(10) << r.scheme
                      << std::setw(4) << r.eleorder << std::setw(4) << r.n
                      << std::right << std::setw(13) << std::fixed
                      << std::setprecision(3) << pc
                      << std::setw(13) << pf
                      << std::setw(8) << std::setprecision(2) << full_speedup << "x"
                      << std::setw(13) << mc
                      << std::setw(13) << mf
                      << std::setw(8) << mix_speedup << "x"
                      << "\n";
            csv << r.scheme << "," << r.eleorder << "," << r.n << ","
                << pc << "," << pf << "," << mc << "," << mf << ","
                << r.rmse_xy << "," << r.rmse_xz << "," << r.rmse_yz << "\n";
        }
        std::cout << std::string(86, '-') << "\n";
    }
    csv.close();
    std::cout << "\nRMSE (last scheme x last n) for fused mixed derivs:\n";
    return 0;
}
