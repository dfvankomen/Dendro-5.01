// Prototype: fused d^2u/dxdy in a single call vs the chained
// grad_x then grad_y_last sequence.
//
// Compares both wall time per call and RMSE against the analytical
// mixed 2nd-derivative of u = sin(2pi x) sin(2pi y) sin(2pi z). The
// fused function lives on MatrixCompactDerivs<1>; we dynamic_cast to
// reach it.
//
// Schemes timed: JTT4, JTT6, JTP6, BL6, BorisO6 (representative 1st-
// order matrix-form schemes). Block size sweep: eleorder 6, 8, 10.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

static void init_field(double *u, double *du_xy, unsigned int n,
                       unsigned int pw) {
    const double twopi = 2.0 * M_PI;
    const double k2    = twopi * twopi;
    for (unsigned int k = 0; k < n; k++) {
        const double z  = ((int)k - (int)pw) * DX;
        const double sz_v = std::sin(twopi * z);
        for (unsigned int j = 0; j < n; j++) {
            const double y  = ((int)j - (int)pw) * DX;
            const double cy = std::cos(twopi * y);
            const double sy = std::sin(twopi * y);
            for (unsigned int i = 0; i < n; i++) {
                const double x  = ((int)i - (int)pw) * DX;
                const double cx = std::cos(twopi * x);
                const double sx = std::sin(twopi * x);
                const unsigned int idx = i + j * n + k * n * n;
                u[idx]     = sx * sy * sz_v;
                du_xy[idx] = k2 * cx * cy * sz_v;
            }
        }
    }
}

// RMSE over the active interior region only (we don't care about
// padding-cell output; the fused path leaves those unwritten).
static double rmse_interior(const double *a, const double *b,
                            unsigned int n, unsigned int pw) {
    double s        = 0.0;
    unsigned long c = 0;
    for (unsigned int k = pw; k < n - pw; k++) {
        for (unsigned int j = pw; j < n - pw; j++) {
            for (unsigned int i = pw; i < n - pw; i++) {
                const unsigned int idx = i + j * n + k * n * n;
                const double d         = a[idx] - b[idx];
                s += d * d;
                c++;
            }
        }
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

struct Row {
    std::string scheme;
    unsigned int eleorder, n;
    double chain_us, fused_us, chain_rmse, fused_rmse;
    bool fused_available;
};

static Row time_scheme(const std::string &scheme, unsigned int eleorder) {
    Row r{};
    r.scheme   = scheme;
    r.eleorder = eleorder;
    const unsigned int n  = eleorder * 2 + 1;
    const unsigned int pw = eleorder / 2;
    r.n = n;

    const auto &reg = get_first_order_registry();
    auto it = reg.find(scheme);
    if (it == reg.end()) return r;

    std::vector<double> empty;
    std::vector<double> placeholder = {0.5};
    std::string no_filter           = "none";

    std::unique_ptr<Derivs> deriv;
    try {
        deriv = it->second(eleorder, no_filter, empty, placeholder, 1u);
    } catch (...) { return r; }
    if (!deriv) return r;

    const size_t total = (size_t)n * n * n;
    deriv->set_maximum_block_size(total);
    deriv->pre_create_for_size(n);

    std::vector<double> u(total), du_xy_true(total);
    std::vector<double> intermediate(total, 0.0);
    std::vector<double> w_chain(total, 0.0);
    std::vector<double> w_fused(total, 0.0);
    init_field(u.data(), du_xy_true.data(), n, pw);

    const unsigned int sz[3] = {n, n, n};
    const unsigned int bflag = 0;

    // chain: grad_x then grad_y_last
    auto run_chain = [&]() {
        deriv->do_grad_x(intermediate.data(), u.data(), DX, sz, bflag);
        deriv->do_grad_y_last(w_chain.data(), intermediate.data(), DX, sz,
                              bflag);
    };

    // fused: dynamic_cast to MatrixCompactDerivs<1>; call do_grad_xy_last
    auto *mat1 =
        dynamic_cast<MatrixCompactDerivs<1> *>(deriv.get());
    if (!mat1) {
        // not a matrix-form 1st-order; fused path unavailable
        for (unsigned int w = 0; w < NWARMUP; w++) run_chain();
        r.chain_rmse = rmse_interior(w_chain.data(), du_xy_true.data(), n,
                                      pw);
        r.chain_us       = bench_us(run_chain, NITER);
        r.fused_available = false;
        return r;
    }

    auto run_fused = [&]() {
        mat1->do_grad_xy_last(w_fused.data(), u.data(), DX, sz, bflag);
    };

    // warmup
    for (unsigned int w = 0; w < NWARMUP; w++) run_chain();
    for (unsigned int w = 0; w < NWARMUP; w++) run_fused();
    r.chain_rmse = rmse_interior(w_chain.data(), du_xy_true.data(), n, pw);
    r.fused_rmse = rmse_interior(w_fused.data(), du_xy_true.data(), n, pw);

    r.chain_us       = bench_us(run_chain, NITER);
    r.fused_us       = bench_us(run_fused, NITER);
    r.fused_available = true;
    return r;
}

int main(int argc, char **argv) {
    const std::string csv_path =
        (argc > 1) ? argv[1] : "fused_mixed.csv";
    std::ofstream csv(csv_path);
    csv << "scheme,eleorder,n,chain_us,fused_us,speedup,chain_rmse,fused_rmse\n";

    std::cout << "fused d^2u/dxdy vs chain (grad_x; grad_y_last)\n"
              << "  warmup = " << NWARMUP << " calls, time = "
              << NITER << " calls, bflag = 0\n\n";

    std::cout << std::left << std::setw(10) << "scheme"
              << std::setw(4) << "eo" << std::setw(4) << "n"
              << std::right << std::setw(11) << "chain(us)"
              << std::setw(11) << "fused(us)" << std::setw(10) << "speedup"
              << std::setw(13) << "rmse(chain)" << std::setw(13)
              << "rmse(fused)" << "\n";
    std::cout << std::string(76, '-') << "\n";

    const std::vector<std::string> schemes = {
        "JTT4", "JTT6", "JTP6", "BL6", "BorisO6"
    };

    for (unsigned int eo : ELEORDERS) {
        for (const auto &s : schemes) {
            Row r = time_scheme(s, eo);
            const double speedup = (r.fused_available && r.fused_us > 0)
                                       ? r.chain_us / r.fused_us
                                       : std::nan("");
            std::cout << std::left << std::setw(10) << r.scheme
                      << std::setw(4) << r.eleorder << std::setw(4) << r.n
                      << std::right << std::setw(11) << std::fixed
                      << std::setprecision(3)
                      << (r.chain_us / NITER)
                      << std::setw(11)
                      << (r.fused_available ? r.fused_us / NITER
                                            : std::nan(""))
                      << std::setw(10) << std::setprecision(2) << speedup << "x"
                      << std::setw(13) << std::scientific
                      << std::setprecision(2) << r.chain_rmse
                      << std::setw(13)
                      << (r.fused_available ? r.fused_rmse : std::nan(""))
                      << "\n";
            csv << r.scheme << "," << r.eleorder << "," << r.n << ","
                << (r.chain_us / NITER) << ","
                << (r.fused_available ? r.fused_us / NITER : std::nan(""))
                << "," << speedup << "," << r.chain_rmse << ","
                << (r.fused_available ? r.fused_rmse : std::nan("")) << "\n";
        }
        std::cout << std::string(76, '-') << "\n";
    }
    csv.close();
    return 0;
}
