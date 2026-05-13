// Banded LAPACK solve vs. LIBXSMM precomputed-D GEMM: per-block kernel
// timing comparison at the standard test configuration.
//
// Constraints inherited from the banded path (see findings/07):
//   - Block size fixed at n = 2*eleorder + 1 = 13 (eleorder = 6); banded
//     factorization is one-size-only in the current implementation.
//   - bflag = 0 (NO_BOUNDARY); only that boundary variant is built on
//     the banded path. Matrix-form has all four variants.
//
// What this measures: per-call wall time for grad_x / grad_y / grad_z on
// a single 13^3 block, over NITER iterations after NWARMUP warmup calls.
// The matrix-form path's first call JITs its LIBXSMM kernels; the
// warmup ensures we measure the steady-state cost.
//
// What this does NOT measure:
//   - Setup cost (P factorization vs. D = P^{-1}Q + pre-scaling).
//   - Block-size sweep (banded can't do it).
//   - Boundary-variant cost (banded only does NO_BOUNDARY).
//   - End-to-end timestep cost (this is a kernel microbench).
//
// CSV columns: scheme, deriv_order, path, direction, niter, total_us,
// per_call_us, rmse.
//
// rmse is included as a sanity check: if the per-path implementation is
// broken at this config, RMSE will be O(1) and the timing isn't
// meaningful.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "derivatives.h"
#include "derivatives/derivs_factory.h"
#include "derivatives/derivs_utils.h"

using namespace dendroderivs;

// eleorder sweep — the banded path's p_n = 2*eleorder+1 is fixed at
// construction time, so this is how we get larger blocks for the
// comparison. eleorder = 6 / 8 / 10 / 12 corresponds to block sizes
// 13 / 17 / 21 / 25.
static const std::vector<unsigned int> ELEORDERS = {6, 8, 10, 12};
static constexpr double DX             = 0.05;
static constexpr unsigned int NWARMUP  = 200;
static constexpr unsigned int NITER    = 10000;

// 3D test function (same as testAllDerivs): u = sin(2*pi*x) sin(2*pi*y) sin(2*pi*z)
static void init_field(double *u, double *du_true_x, double *ddu_true_x,
                       unsigned int n, unsigned int pw) {
    const double twopi = 2.0 * M_PI;
    for (unsigned int k = 0; k < n; k++) {
        const double z = ((int)k - (int)pw) * DX;
        const double sz_v = std::sin(twopi * z);
        for (unsigned int j = 0; j < n; j++) {
            const double y = ((int)j - (int)pw) * DX;
            const double sy = std::sin(twopi * y);
            for (unsigned int i = 0; i < n; i++) {
                const double x = ((int)i - (int)pw) * DX;
                const double sx = std::sin(twopi * x);
                const double cx = std::cos(twopi * x);
                const unsigned int idx = i + j * n + k * n * n;
                u[idx]         = sx * sy * sz_v;
                du_true_x[idx] = twopi * cx * sy * sz_v;
                ddu_true_x[idx] = -(twopi * twopi) * sx * sy * sz_v;
            }
        }
    }
}

static double rmse_active(const double *a, const double *b,
                          unsigned int n, unsigned int pw) {
    double s = 0.0;
    unsigned long count = 0;
    for (unsigned int k = pw; k < n - pw; k++) {
        for (unsigned int j = pw; j < n - pw; j++) {
            for (unsigned int i = pw; i < n - pw; i++) {
                const unsigned int idx = i + j * n + k * n * n;
                const double d = a[idx] - b[idx];
                s += d * d;
                count++;
            }
        }
    }
    return count > 0 ? std::sqrt(s / (double)count) : 0.0;
}

template <typename Fn>
static double bench_us(Fn &&fn, unsigned int niter) {
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < niter; i++) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

struct PathTiming {
    std::string scheme;
    int deriv_order;
    std::string path;
    unsigned int eleorder;
    unsigned int n;
    double total_us[3] = {0, 0, 0};      // x, y, z
    double rmse_x = std::nan("");
    bool ok = false;
};

static PathTiming run_path(const std::string &scheme_name, int deriv_order,
                           const std::string &path_label,
                           unsigned int eleorder,
                           const std::vector<double> &u_init,
                           const std::vector<double> &truth_x) {
    PathTiming t;
    t.scheme = scheme_name;
    t.deriv_order = deriv_order;
    t.path = path_label;
    t.eleorder = eleorder;
    const unsigned int n = eleorder * 2 + 1;
    const unsigned int pw = eleorder / 2;
    t.n = n;

    const auto &reg = (deriv_order == 1) ? get_first_order_registry()
                                          : get_second_order_registry();
    auto it = reg.find(scheme_name);
    if (it == reg.end()) {
        std::cerr << "  [skip] " << scheme_name << " not in "
                  << (deriv_order == 1 ? "first" : "second") << "-order registry\n";
        return t;
    }

    std::vector<double> empty_coeffs;
    std::vector<double> placeholder = {0.5};
    std::string no_filter = "none";

    std::unique_ptr<Derivs> deriv;
    try {
        deriv = it->second(eleorder, no_filter, empty_coeffs, placeholder, 1u);
    } catch (const std::exception &e) {
        std::cerr << "  [skip] " << scheme_name << " (eleorder=" << eleorder
                  << ") construct threw: " << e.what() << "\n";
        return t;
    }
    if (!deriv) {
        std::cerr << "  [skip] " << scheme_name
                  << " (eleorder=" << eleorder << ") construct returned null\n";
        return t;
    }

    const size_t total = (size_t)n * n * n;
    deriv->set_maximum_block_size(total);
    deriv->pre_create_for_size(n);                        // JIT/setup eagerly

    std::vector<double> du(total, 0.0);
    const unsigned int sz[3] = {n, n, n};
    const unsigned int bflag = 0;                          // NO_BOUNDARY

    // path_label controls which variant of grad_* is invoked. "matrix"
    // and "banded" call the default (safe) do_grad_*; "matrix_last"
    // calls do_grad_*_last on the matrix-form path.
    const bool use_last = (path_label == "matrix_last");

    auto call_x = [&]() {
        if (use_last)
            deriv->do_grad_x_last(du.data(), u_init.data(), DX, sz, bflag);
        else
            deriv->do_grad_x(du.data(), u_init.data(), DX, sz, bflag);
    };
    auto call_y = [&]() {
        if (use_last)
            deriv->do_grad_y_last(du.data(), u_init.data(), DX, sz, bflag);
        else
            deriv->do_grad_y(du.data(), u_init.data(), DX, sz, bflag);
    };
    auto call_z = [&]() {
        // no _last variant for z (do_grad_z already skips unconditionally)
        deriv->do_grad_z(du.data(), u_init.data(), DX, sz, bflag);
    };

    // warmup
    for (unsigned int w = 0; w < NWARMUP; w++) call_x();
    t.rmse_x = rmse_active(du.data(), truth_x.data(), n, pw);

    t.total_us[0] = bench_us(call_x, NITER);

    for (unsigned int w = 0; w < NWARMUP; w++) call_y();
    t.total_us[1] = bench_us(call_y, NITER);

    for (unsigned int w = 0; w < NWARMUP; w++) call_z();
    t.total_us[2] = bench_us(call_z, NITER);

    t.ok = true;
    return t;
}

static void emit_csv_rows(std::ofstream &csv, const PathTiming &t) {
    if (!t.ok) return;
    static const char *dir_names[3] = {"x", "y", "z"};
    for (int d = 0; d < 3; d++) {
        csv << t.scheme << "," << t.deriv_order << "," << t.path << ","
            << t.eleorder << "," << t.n << "," << dir_names[d] << ","
            << NITER << ","
            << std::scientific << std::setprecision(6) << t.total_us[d] << ","
            << (t.total_us[d] / (double)NITER) << ","
            << t.rmse_x << "\n";
    }
}

static void print_summary_pair(const PathTiming &mat, const PathTiming &bnd) {
    static const char *dir_names[3] = {"x", "y", "z"};
    for (int d = 0; d < 3; d++) {
        const double mp = mat.ok ? mat.total_us[d] / NITER : std::nan("");
        const double bp = bnd.ok ? bnd.total_us[d] / NITER : std::nan("");
        const double ratio = (mat.ok && bnd.ok && mp > 0)
                                 ? bp / mp
                                 : std::nan("");
        std::cout << std::left << std::setw(7) << mat.scheme
                  << std::setw(4) << (mat.deriv_order == 1 ? "1st" : "2nd")
                  << std::setw(4) << mat.n
                  << std::setw(4) << dir_names[d]
                  << std::right << std::setw(11) << std::fixed
                  << std::setprecision(3) << mp
                  << std::setw(11) << bp
                  << std::setw(9) << std::setprecision(2) << ratio
                  << std::setw(12) << std::scientific << std::setprecision(2)
                  << mat.rmse_x
                  << std::setw(12) << bnd.rmse_x
                  << "\n";
    }
}

int main(int argc, char **argv) {
    const std::string csv_path =
        (argc > 1) ? argv[1] : "banded_vs_xsmm.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Could not open " << csv_path << "\n";
        return 1;
    }
    csv << "scheme,deriv_order,path,eleorder,n,direction,niter,total_us,"
           "per_call_us,rmse\n";

    std::cout << "banded vs. LIBXSMM kernel microbenchmark\n"
              << "  eleorder sweep:";
    for (auto e : ELEORDERS) std::cout << " " << e << "(n=" << e*2+1 << ")";
    std::cout << ", dx = " << DX << "\n"
              << "  warmup = " << NWARMUP << " calls, time = "
              << NITER << " calls\n"
              << "  bflag = 0 (NO_BOUNDARY); banded path only supports this\n"
              << "  CSV: " << csv_path << "\n\n";

    struct SchemePair {
        const char *base_name;
        const char *banded_name;
    };
    const SchemePair schemes[] = {
        {"JTT4", "JTT4Banded"},
        {"JTT6", "JTT6Banded"},
        {"JTP6", "JTP6Banded"},
    };

    std::cout << std::left
              << std::setw(7) << "scheme" << std::setw(4) << "ord"
              << std::setw(4) << "n" << std::setw(4) << "dir" << std::right
              << std::setw(11) << "matrix_us" << std::setw(11) << "banded_us"
              << std::setw(9) << "b/m"
              << std::setw(12) << "rmse_mat" << std::setw(12) << "rmse_bnd"
              << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (unsigned int eleorder : ELEORDERS) {
        const unsigned int n = eleorder * 2 + 1;
        const unsigned int pw = eleorder / 2;
        const size_t total = (size_t)n * n * n;

        // per-eleorder test buffers (size depends on n)
        std::vector<double> u(total), du_true_x(total), ddu_true_x(total);
        init_field(u.data(), du_true_x.data(), ddu_true_x.data(), n, pw);

        for (int order = 1; order <= 2; order++) {
            const std::vector<double> &truth =
                (order == 1) ? du_true_x : ddu_true_x;
            for (const auto &p : schemes) {
                PathTiming mat = run_path(p.base_name,   order, "matrix",
                                          eleorder, u, truth);
                PathTiming bnd = run_path(p.banded_name, order, "banded",
                                          eleorder, u, truth);
                PathTiming lst = run_path(p.base_name,   order, "matrix_last",
                                          eleorder, u, truth);
                print_summary_pair(mat, bnd);
                emit_csv_rows(csv, mat);
                emit_csv_rows(csv, bnd);
                emit_csv_rows(csv, lst);
            }
        }
        std::cout << std::string(72, '-') << "\n";
    }
    csv.close();
    std::cout << "\nCSV written to " << csv_path << "\n";
    return 0;
}
