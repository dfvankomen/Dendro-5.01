#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "derivatives.h"
#include "derivatives/derivs_factory.h"

using namespace dendroderivs;

// simple sine test function: u = sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
// analytical dx = 2*pi * cos(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
// analytical dxx = -4*pi^2 * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
static void init_sine(double *u, double *du_true, double *duu_true,
                      const unsigned int *sz, double dx, double dy, double dz,
                      unsigned int pw) {
    const double twopi = 2.0 * M_PI;
    const unsigned int nx = sz[0], ny = sz[1], nz = sz[2];

    for (unsigned int k = 0; k < nz; k++) {
        double z = (k - (double)pw) * dz;
        for (unsigned int j = 0; j < ny; j++) {
            double y = (j - (double)pw) * dy;
            for (unsigned int i = 0; i < nx; i++) {
                double x = (i - (double)pw) * dx;
                unsigned int idx = i + j * nx + k * nx * ny;

                double sx = sin(twopi * x), cx = cos(twopi * x);
                double sy = sin(twopi * y);
                double sz_v = sin(twopi * z);

                u[idx] = sx * sy * sz_v;
                if (du_true)
                    du_true[idx] = twopi * cx * sy * sz_v;
                if (duu_true)
                    duu_true[idx] = -twopi * twopi * sx * sy * sz_v;
            }
        }
    }
}

static double compute_rmse(const double *a, const double *b, unsigned int n,
                           unsigned int pw, const unsigned int *sz) {
    // only compare interior points (skip padding)
    double sum = 0.0;
    unsigned int count = 0;
    const unsigned int nx = sz[0], ny = sz[1], nz = sz[2];

    for (unsigned int k = pw; k < nz - pw; k++) {
        for (unsigned int j = pw; j < ny - pw; j++) {
            for (unsigned int i = pw; i < nx - pw; i++) {
                unsigned int idx = i + j * nx + k * nx * ny;
                double diff = a[idx] - b[idx];
                sum += diff * diff;
                count++;
            }
        }
    }
    return count > 0 ? sqrt(sum / count) : 0.0;
}

int main() {
    const unsigned int eleorder = 6;
    const unsigned int pw = eleorder / 2;
    const unsigned int n = eleorder * 2 + 1;  // single block size
    const unsigned int sz[3] = {n, n, n};
    const unsigned int total = n * n * n;
    const double dx = 0.05, dy = 0.05, dz = 0.05;

    std::vector<double> u(total), du_true(total), duu_true(total);
    std::vector<double> du_computed(total, 0.0);

    init_sine(u.data(), du_true.data(), duu_true.data(), sz, dx, dy, dz, pw);

    // default args for factory
    std::vector<double> empty_coeffs;
    std::vector<double> test_coeffs = {0.5};  // some schemes need at least one coeff
    std::string no_filter = "none";
    unsigned int matID = 1;

    auto &first_reg = get_first_order_registry();
    auto &second_reg = get_second_order_registry();

    int pass = 0, fail = 0, total_tested = 0;

    std::cout << std::left << std::setw(20) << "Scheme"
              << std::setw(8) << "Order"
              << std::setw(15) << "RMSE (x)"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(53, '-') << std::endl;

    // test all first-order schemes
    for (auto &[name, creator] : first_reg) {
        total_tested++;
        std::fill(du_computed.begin(), du_computed.end(), 0.0);

        try {
            auto deriv = creator(eleorder, no_filter, empty_coeffs,
                                 test_coeffs, matID);
            if (!deriv) {
                std::cout << std::setw(20) << name << std::setw(8) << "1st"
                          << std::setw(15) << "N/A"
                          << "FAIL (null)" << std::endl;
                fail++;
                continue;
            }

            // ===================================================
            // NOTE: set_maximum_block_size MUST be called on every
            // Derivs instance (and on each clone for OMP) before
            // do_grad_y/z. Matrix-based subclasses size a 2*Nx*Ny*Nz
            // workspace from this. Skipping it now lazy-grows on
            // demand, but you'll pay heap reallocations in the hot
            // loop — always size up front in real solvers.
            // ===================================================
            deriv->set_maximum_block_size(total);
            deriv->do_grad_x(du_computed.data(), u.data(), dx, sz, 0);

            double rmse = compute_rmse(du_computed.data(), du_true.data(),
                                       total, pw, sz);

            // check for NaN or unreasonable values
            bool has_nan = std::isnan(rmse) || std::isinf(rmse);
            bool reasonable = rmse < 10.0;  // very loose check

            if (!has_nan && reasonable) {
                std::cout << std::setw(20) << name << std::setw(8) << "1st"
                          << std::setw(15) << std::scientific
                          << std::setprecision(3) << rmse
                          << "OK" << std::endl;
                pass++;
            } else {
                std::cout << std::setw(20) << name << std::setw(8) << "1st"
                          << std::setw(15) << rmse
                          << "FAIL" << std::endl;
                fail++;
            }
        } catch (const std::exception &e) {
            std::cout << std::setw(20) << name << std::setw(8) << "1st"
                      << std::setw(15) << "N/A"
                      << "EXCEPT: " << e.what() << std::endl;
            fail++;
        }
    }

    // test all second-order schemes
    for (auto &[name, creator] : second_reg) {
        total_tested++;
        std::fill(du_computed.begin(), du_computed.end(), 0.0);

        try {
            auto deriv = creator(eleorder, no_filter, empty_coeffs,
                                 test_coeffs, matID);
            if (!deriv) {
                std::cout << std::setw(20) << name << std::setw(8) << "2nd"
                          << std::setw(15) << "N/A"
                          << "FAIL (null)" << std::endl;
                fail++;
                continue;
            }

            deriv->set_maximum_block_size(total);
            deriv->do_grad_x(du_computed.data(), u.data(), dx, sz, 0);

            double rmse = compute_rmse(du_computed.data(), duu_true.data(),
                                       total, pw, sz);

            bool has_nan = std::isnan(rmse) || std::isinf(rmse);
            bool reasonable = rmse < 100.0;  // 2nd order can be larger

            if (!has_nan && reasonable) {
                std::cout << std::setw(20) << name << std::setw(8) << "2nd"
                          << std::setw(15) << std::scientific
                          << std::setprecision(3) << rmse
                          << "OK" << std::endl;
                pass++;
            } else {
                std::cout << std::setw(20) << name << std::setw(8) << "2nd"
                          << std::setw(15) << rmse
                          << "FAIL" << std::endl;
                fail++;
            }
        } catch (const std::exception &e) {
            std::cout << std::setw(20) << name << std::setw(8) << "2nd"
                      << std::setw(15) << "N/A"
                      << "EXCEPT: " << e.what() << std::endl;
            fail++;
        }
    }

    std::cout << std::string(53, '-') << std::endl;
    std::cout << "Total: " << total_tested << "  Pass: " << pass
              << "  Fail: " << fail << std::endl;

    // ---- coefficient sensitivity test ----
    // verify that BYU/coeff-accepting schemes actually change output when
    // coefficients change. if the matrices are truly parameterized, different
    // coeffs should give different RMSE.
    std::cout << "\n===== Coefficient sensitivity test =====" << std::endl;
    std::cout << std::left << std::setw(16) << "Scheme"
              << std::setw(14) << "RMSE(c1)"
              << std::setw(14) << "RMSE(c2)"
              << std::setw(14) << "RMSE(c3)"
              << "Changed?" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // schemes that accept coefficients (pattern C)
    std::vector<std::string> coeff_schemes = {
        "BYUT4", "BYUT6", "BYUT6R2", "BYUT6R3", "BYUP6", "BYUP6R2",
        "BYUT4R1", "BYUT4R2", "BYUT64R3", "BYUP8",
    };

    std::vector<std::vector<double>> coeff_sets = {
        {0.1, 0.1, 0.1, 0.1, 0.1},
        {0.3, 0.2, 0.1, 0.05, 0.01},
        {0.5, 0.4, 0.3, 0.2, 0.1},
    };

    int coeff_pass = 0, coeff_fail = 0;

    for (auto &scheme_name : coeff_schemes) {
        auto it = first_reg.find(scheme_name);
        if (it == first_reg.end()) continue;

        double rmses[3] = {};
        bool all_ok = true;

        for (int ci = 0; ci < 3; ci++) {
            std::fill(du_computed.begin(), du_computed.end(), 0.0);
            try {
                auto deriv = it->second(eleorder, no_filter, empty_coeffs,
                                        coeff_sets[ci], matID);
                if (!deriv) { all_ok = false; break; }
                deriv->set_maximum_block_size(total);
                deriv->do_grad_x(du_computed.data(), u.data(), dx, sz, 0);
                rmses[ci] = compute_rmse(du_computed.data(), du_true.data(),
                                         total, pw, sz);
            } catch (...) {
                all_ok = false;
                break;
            }
        }

        if (!all_ok) {
            std::cout << std::setw(16) << scheme_name << "CONSTRUCTION FAILED"
                      << std::endl;
            coeff_fail++;
            continue;
        }

        // check that at least two of the three are different
        bool changed = (std::abs(rmses[0] - rmses[1]) > 1e-15) ||
                       (std::abs(rmses[1] - rmses[2]) > 1e-15);

        std::cout << std::setw(16) << scheme_name
                  << std::setw(14) << std::scientific << std::setprecision(3)
                  << rmses[0]
                  << std::setw(14) << rmses[1]
                  << std::setw(14) << rmses[2]
                  << (changed ? "YES" : "NO (PROBLEM)") << std::endl;

        if (changed) coeff_pass++;
        else coeff_fail++;
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Coeff test: " << coeff_pass << " pass, " << coeff_fail
              << " fail" << std::endl;

    // ---- batch API test ----
    // verify that grad_x_batch gives the same results as calling grad_x
    // individually for each variable. tests both explicit and compact paths.
    std::cout << "\n===== Batch API test =====" << std::endl;

    int batch_pass = 0, batch_fail = 0;

    std::vector<std::string> batch_test_types = {"E6", "JTT6"};

    for (auto &dtype : batch_test_types) {
        DendroDerivatives deriv_obj(dtype, dtype, eleorder);
        // NOTE: DendroDerivatives.set_maximum_block_size forwards to BOTH
        // the 1st- and 2nd-order Derivs it owns. This must be called once
        // per object (and once per copy/clone) — see the Derivs class
        // docstring "Workspace sizing" section for the full contract.
        deriv_obj.set_maximum_block_size(total);

        const unsigned int n_test_vars = 4;

        // build distinct inputs (different phase shifts)
        std::vector<std::vector<double>> inputs(n_test_vars, std::vector<double>(total));
        std::vector<std::vector<double>> du_single(n_test_vars, std::vector<double>(total, 0.0));
        std::vector<std::vector<double>> du_batch(n_test_vars, std::vector<double>(total, 0.0));

        for (unsigned int v = 0; v < n_test_vars; v++) {
            double phase = 0.3 * v;
            for (unsigned int k = 0; k < sz[2]; k++)
                for (unsigned int j = 0; j < sz[1]; j++)
                    for (unsigned int i = 0; i < sz[0]; i++) {
                        unsigned int idx = i + j * sz[0] + k * sz[0] * sz[1];
                        double x = (i - (double)pw) * dx;
                        double y = (j - (double)pw) * dy;
                        double z = (k - (double)pw) * dz;
                        inputs[v][idx] = sin(2.0 * M_PI * (x + phase)) *
                                         sin(2.0 * M_PI * y) *
                                         sin(2.0 * M_PI * z);
                    }
        }

        // individual calls
        for (unsigned int v = 0; v < n_test_vars; v++)
            deriv_obj.grad_x(du_single[v].data(), inputs[v].data(), dx, sz, 0);

        // batch call
        std::vector<double *> du_ptrs(n_test_vars);
        std::vector<const double *> u_ptrs(n_test_vars);
        for (unsigned int v = 0; v < n_test_vars; v++) {
            du_ptrs[v] = du_batch[v].data();
            u_ptrs[v]  = inputs[v].data();
        }
        deriv_obj.grad_x_batch(du_ptrs.data(), u_ptrs.data(), n_test_vars, dx, sz, 0);

        // compare
        bool all_match = true;
        for (unsigned int v = 0; v < n_test_vars; v++) {
            double max_diff = 0.0;
            for (unsigned int i = 0; i < total; i++) {
                double diff = std::abs(du_single[v][i] - du_batch[v][i]);
                max_diff = std::max(max_diff, diff);
            }
            if (max_diff > 1e-14) {
                std::cout << dtype << " var " << v << ": max_diff=" << max_diff
                          << " FAIL" << std::endl;
                all_match = false;
            }
        }

        if (all_match) {
            std::cout << dtype << " grad_x_batch: " << n_test_vars
                      << " vars match individual calls - OK" << std::endl;
            batch_pass++;
        } else {
            batch_fail++;
        }

        // also test y and z batch
        for (unsigned int v = 0; v < n_test_vars; v++) {
            std::fill(du_single[v].begin(), du_single[v].end(), 0.0);
            std::fill(du_batch[v].begin(), du_batch[v].end(), 0.0);
        }
        for (unsigned int v = 0; v < n_test_vars; v++)
            deriv_obj.grad_y(du_single[v].data(), inputs[v].data(), dy, sz, 0);
        deriv_obj.grad_y_batch(du_ptrs.data(), u_ptrs.data(), n_test_vars, dy, sz, 0);

        all_match = true;
        for (unsigned int v = 0; v < n_test_vars; v++) {
            for (unsigned int i = 0; i < total; i++) {
                double diff = std::abs(du_single[v][i] - du_batch[v][i]);
                if (diff > 1e-14) { all_match = false; break; }
            }
        }
        std::cout << dtype << " grad_y_batch: "
                  << (all_match ? "OK" : "FAIL") << std::endl;
        if (all_match) batch_pass++; else batch_fail++;

        for (unsigned int v = 0; v < n_test_vars; v++) {
            std::fill(du_single[v].begin(), du_single[v].end(), 0.0);
            std::fill(du_batch[v].begin(), du_batch[v].end(), 0.0);
        }
        for (unsigned int v = 0; v < n_test_vars; v++)
            deriv_obj.grad_z(du_single[v].data(), inputs[v].data(), dz, sz, 0);
        deriv_obj.grad_z_batch(du_ptrs.data(), u_ptrs.data(), n_test_vars, dz, sz, 0);

        all_match = true;
        for (unsigned int v = 0; v < n_test_vars; v++) {
            for (unsigned int i = 0; i < total; i++) {
                double diff = std::abs(du_single[v][i] - du_batch[v][i]);
                if (diff > 1e-14) { all_match = false; break; }
            }
        }
        std::cout << dtype << " grad_z_batch: "
                  << (all_match ? "OK" : "FAIL") << std::endl;
        if (all_match) batch_pass++; else batch_fail++;
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Batch test: " << batch_pass << " pass, " << batch_fail
              << " fail" << std::endl;

    // ---- fused-block correctness + batch profiling sweep ----
    // fused blocks stack multiple element blocks into one cfd solve.
    // block size formula: eleorder * (ninterior + 1) + 1
    //   ninterior=1 -> 13 (the existing single-block case)
    //   ninterior=2 -> 19 (two blocks fused)
    //   ninterior=3 -> 25 (three blocks fused)
    // verifies correctness (RMSE of batch grad_x vs analytical) and
    // benchmarks individual vs batch at each fused size
    std::cout << "\n===== Fused-block sweep (correctness + batch profiling) ====="
              << std::endl;

    // E6 is explicit (stencil-based), E6Matrix and JTT6 are CFDs using the
    // matmul path — comparison shows CFD overhead vs explicit
    std::vector<std::string> profile_types = {"E6", "E6Matrix", "JTT6"};
    std::vector<unsigned int> fused_counts = {1, 2, 3};
    std::vector<unsigned int> var_counts = {1, 4, 8, 24};
    const unsigned int profile_iters = 5000;

    // prewarm libxsmm kernel caches for every shape we'll hit in this sweep.
    // exercises the thread-safe prewarm path and makes the first calls cheap
    std::vector<BlockShape> prewarm_shapes;
    for (auto ninterior : fused_counts) {
        unsigned int fn = eleorder * (ninterior + 1) + 1;
        prewarm_shapes.push_back({fn, fn, fn});
    }
    prewarm_kernel_cache(prewarm_shapes, pw);

    int fused_pass = 0, fused_fail = 0;

    for (auto &dtype : profile_types) {
        // size the derivs object for the largest block we'll see in this sweep
        unsigned int max_n = eleorder * (*std::max_element(fused_counts.begin(),
                                                           fused_counts.end()) + 1) + 1;
        unsigned int max_total = max_n * max_n * max_n;

        DendroDerivatives deriv_obj(dtype, dtype, eleorder);
        deriv_obj.set_maximum_block_size(max_total);

        std::cout << "\n--- " << dtype << " ---" << std::endl;
        std::cout << std::left << std::setw(6) << "n"
                  << std::setw(11) << "rmse_x"
                  << std::setw(6) << "nvars"
                  << std::setw(11) << "x_batch"
                  << std::setw(11) << "y_batch"
                  << std::setw(11) << "z_batch"
                  << std::setw(11) << "total(us)"
                  << std::endl;

        for (auto ninterior : fused_counts) {
            unsigned int fn = eleorder * (ninterior + 1) + 1;
            unsigned int fsz[3] = {fn, fn, fn};
            unsigned int ftotal = fn * fn * fn;

            // build a sine-derivative ground truth at this block size
            std::vector<double> u_true(ftotal), du_true_x(ftotal);
            init_sine(u_true.data(), du_true_x.data(), nullptr, fsz, dx, dy, dz, pw);

            // run batch grad_x on a single-variable batch and compare rmse
            // to analytical — easy check that the larger n "just works"
            std::vector<double> du_out(ftotal, 0.0);
            double *du_ptr_single          = du_out.data();
            const double *u_ptr_single     = u_true.data();
            deriv_obj.grad_x_batch(&du_ptr_single, &u_ptr_single, 1, dx, fsz, 0);
            double rmse_x = compute_rmse(du_out.data(), du_true_x.data(),
                                         ftotal, pw, fsz);
            bool ok = std::isfinite(rmse_x) && rmse_x < 10.0;
            if (ok) fused_pass++; else fused_fail++;

            for (auto nv : var_counts) {
                // allocate variable data at this block size
                std::vector<std::vector<double>> u_data(nv,
                    std::vector<double>(ftotal));
                std::vector<std::vector<double>> du_data(nv,
                    std::vector<double>(ftotal, 0.0));

                for (unsigned int v = 0; v < nv; v++) {
                    double phase = 0.3 * v;
                    for (unsigned int idx = 0; idx < ftotal; idx++) {
                        unsigned int i = idx % fsz[0];
                        unsigned int j = (idx / fsz[0]) % fsz[1];
                        double x = (i - (double)pw) * dx;
                        double y = (j - (double)pw) * dy;
                        u_data[v][idx] = sin(2.0 * M_PI * (x + phase) + y);
                    }
                }

                std::vector<double *> du_ptrs(nv);
                std::vector<const double *> u_ptrs(nv);
                for (unsigned int v = 0; v < nv; v++) {
                    du_ptrs[v] = du_data[v].data();
                    u_ptrs[v]  = u_data[v].data();
                }

                // warmup each direction
                deriv_obj.grad_x_batch(du_ptrs.data(), u_ptrs.data(), nv, dx,
                                       fsz, 0);
                deriv_obj.grad_y_batch(du_ptrs.data(), u_ptrs.data(), nv, dy,
                                       fsz, 0);
                deriv_obj.grad_z_batch(du_ptrs.data(), u_ptrs.data(), nv, dz,
                                       fsz, 0);

                auto time_us = [&](auto fn_call) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (unsigned int iter = 0; iter < profile_iters; iter++)
                        fn_call();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    return std::chrono::duration<double, std::micro>(t1 - t0)
                               .count() /
                           profile_iters;
                };

                double xb = time_us([&]() {
                    deriv_obj.grad_x_batch(du_ptrs.data(), u_ptrs.data(), nv,
                                           dx, fsz, 0);
                });
                double yb = time_us([&]() {
                    deriv_obj.grad_y_batch(du_ptrs.data(), u_ptrs.data(), nv,
                                           dy, fsz, 0);
                });
                double zb = time_us([&]() {
                    deriv_obj.grad_z_batch(du_ptrs.data(), u_ptrs.data(), nv,
                                           dz, fsz, 0);
                });

                std::cout << std::setw(6) << fn
                          << std::scientific << std::setprecision(1)
                          << std::setw(11) << rmse_x
                          << std::fixed << std::setprecision(3)
                          << std::setw(6) << nv
                          << std::setw(11) << xb
                          << std::setw(11) << yb
                          << std::setw(11) << zb
                          << std::setw(11) << (xb + yb + zb)
                          << std::endl;
            }
        }
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Fused-block correctness: " << fused_pass << " pass, "
              << fused_fail << " fail" << std::endl;

    return (fail > 0 || coeff_fail > 0 || batch_fail > 0 || fused_fail > 0)
               ? 1
               : 0;
}
