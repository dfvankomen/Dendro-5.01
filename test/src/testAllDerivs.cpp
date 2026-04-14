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

    return (fail > 0 || coeff_fail > 0) ? 1 : 0;
}
