#include <iostream>
#include <tuple>
#include <vector>

#include "compact_derivs.h"
#include "derivs.h"
#include "profiler.h"

#define UNIFORM_RAND_0_TO_X(X) ((double_t)rand() / (double_t)RAND_MAX * X)

namespace helpers {
uint32_t padding;

profiler_t t_deriv_x;
profiler_t t_deriv_y;
profiler_t t_deriv_z;

profiler_t t_compact_deriv_x;
profiler_t t_compact_deriv_y;
profiler_t t_compact_deriv_z;

void print_profiler_results(uint64_t num_runs) {
    long double num_runs_d = (long double)num_runs;

    std::cout << YLW << "==== PROFILING RESULTS ====" << NRM << std::endl;
    std::cout << "Over " << num_runs << " total runs each" << std::endl;

    std::cout << "\t =< Original Stencils >=" << std::endl;
    std::cout << "\tx deriv: total=" << t_deriv_x.seconds
              << " average=" << t_deriv_x.seconds / num_runs_d << std::endl;
    std::cout << "\ty deriv: total=" << t_deriv_y.seconds
              << " average=" << t_deriv_y.seconds / num_runs_d << std::endl;
    std::cout << "\tz deriv: total=" << t_deriv_z.seconds
              << " average=" << t_deriv_z.seconds / num_runs_d << std::endl;

    std::cout << std::endl;

    std::cout << "\t =< Compact Stencils >=" << std::endl;
    std::cout << "\tx deriv: total=" << t_compact_deriv_x.seconds
              << " average=" << t_compact_deriv_x.seconds / num_runs_d
              << std::endl;
    std::cout << "\ty deriv: total=" << t_compact_deriv_y.seconds
              << " average=" << t_compact_deriv_y.seconds / num_runs_d
              << std::endl;
    std::cout << "\tz deriv: total=" << t_compact_deriv_z.seconds
              << " average=" << t_compact_deriv_z.seconds / num_runs_d
              << std::endl;
}

}  // namespace helpers

void sine_init(double_t *u_var, const uint32_t *sz, const double_t *deltas) {
    const double_t x_start = 0.0;
    const double_t y_start = 0.0;
    const double_t z_start = 0.0;
    const double_t dx = deltas[0];
    const double_t dy = deltas[1];
    const double_t dz = deltas[2];

    const double_t amplitude = 0.01;

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        double_t z = z_start + k * dz;
        for (uint16_t j = 0; j < ny; j++) {
            double_t y = y_start + j * dy;
            for (uint16_t i = 0; i < nx; i++) {
                double x = x_start + i * dx;
                u_var[IDX(i, j, k)] = 1.0 * sin(2 * x + 0.1) +
                                      2.0 * sin(3 * y - 0.1) +
                                      0.5 * sin(0.5 * z);
            }
        }
    }
}

void boris_init(double_t *u_var, const uint32_t *sz, const double_t *deltas,
                double_t *u_dx = nullptr, double_t *u_dy = nullptr,
                double_t *u_dz = nullptr, double_t *u_dxx = nullptr,
                double_t *u_dyy = nullptr, double_t *u_dzz = nullptr) {
    const double_t x_start = 0.0;
    const double_t y_start = 1.0;
    const double_t z_start = 1.5;
    const double_t dx = deltas[0];
    const double_t dy = deltas[1];
    const double_t dz = deltas[2];

    const double_t amplitude = 0.01;

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        double_t z = z_start + k * dz;
        for (uint16_t j = 0; j < ny; j++) {
            double_t y = y_start + j * dy;
            for (uint16_t i = 0; i < nx; i++) {
                double_t x = x_start + i * dx;
                u_var[IDX(i, j, k)] =
                    0.5 * exp(-1.0 * sin(2 * (x - 3.14159)) -
                              sin(2 * (y - 3.14159)) - sin(2 * (z - 3.14159)));
                
                if (k == 0 && j == 0) {
                    // std::cout << x << " ";
                }
            }
            if (k == 0) {
                // std::cout << y << " ";
            }
        }
        std::cout << z << " ";
    }
    std::cout << std::endl;

    if (u_dx != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dx[IDX(i, j, k)] =
                        -0.1 *
                        exp(sin(6.28318 - 2 * x) + sin(6.28318 - 2 * y) +
                            sin(6.28318 - 2 * z)) *
                        cos(6.28318 - 2 * x);
                }
            }
        }
    }

    if (u_dy != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dy[IDX(i, j, k)] =
                        -0.1 *
                        exp(sin(6.28318 - 2 * x) + sin(6.28318 - 2 * y) +
                            sin(6.28318 - 2 * z)) *
                        cos(6.28318 - 2 * y);
                }
            }
        }
    }

    if (u_dz != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dz[IDX(i, j, k)] =
                        -0.1 *
                        exp(sin(6.28318 - 2 * x) + sin(6.28318 - 2 * y) +
                            sin(6.28318 - 2 * z)) *
                        cos(6.28318 - 2 * z);
                }
            }
        }
    }

    if (u_dxx != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dxx[IDX(i, j, k)] =
                        (0.2 * cos(6.28318 - 2 * x) * cos(6.28318 - 2 * x) -
                         0.2 * sin(6.28318 - 2 * x)) *
                        exp(sin(6.28318 - 2 * x) + sin(6.28318 - 2 * y) +
                            sin(6.28318 - 2 * z));
                }
            }
        }
    }

    if (u_dyy != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dyy[IDX(i, j, k)] =
                        (0.2 * cos(6.28318 - 2 * y) * cos(6.28318 - 2 * y) -
                         0.2 * sin(6.28318 - 2 * y)) *
                        exp(sin(6.28318 - 2 * x) + sin(6.28318 - 2 * y) +
                            sin(6.28318 - 2 * z));
                }
            }
        }
    }

    if (u_dzz != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dzz[IDX(i, j, k)] =
                        (0.2 * cos(6.28318 - 2 * z) * cos(6.28318 - 2 * z) -
                         0.2 * sin(6.28318 - 2 * z)) *
                        exp(sin(6.28318 - 2 * x) + sin(6.28318 - 2 * y) +
                            sin(6.28318 - 2 * z));
                }
            }
        }
    }
}

void random_init(double_t *u_var, const uint32_t *sz) {
    const double_t amplitude = 0.001;
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                u_var[IDX(i, j, k)] =
                    amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
            }
        }
    }
}

void zero_init(double_t *u_var, const uint32_t *sz) {
    const double_t amplitude = 0.001;
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                u_var[IDX(i, j, k)] = 0.0;
            }
        }
    }
}

void init_data(const uint32_t init_type, double_t *u_var, const uint32_t *sz,
               const double *deltas, double_t *u_dx = nullptr,
               double_t *u_dy = nullptr, double_t *u_dz = nullptr,
               double_t *u_dxx = nullptr, double_t *u_dyy = nullptr,
               double_t *u_dzz = nullptr) {
    switch (init_type) {
        case 0:
            zero_init(u_var, sz);
            break;

        case 1:
            random_init(u_var, sz);
            break;

        case 2:
            boris_init(u_var, sz, deltas, u_dx, u_dy, u_dz, u_dxx, u_dyy,
                       u_dzz);
            break;

        default:
            std::cout << "UNRECOGNIZED INITIAL DATA FUNCTION... EXITING"
                      << std::endl;
            exit(0);
            break;
    }
}

void print_3d_mat(double_t *u_var, const uint32_t *sz) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];
    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                printf("%f ", u_var[IDX(i, j, k)]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

// void print_square_mat(double *m, const uint32_t n) {
//     // assumes "col" order in memory
//     // J is the row!
//     for (uint16_t i = 0; i < n; i++) {
//         printf("%3d : ", i);
//         // I is the column!
//         for (uint16_t j = 0; j < n; j++) {
//             printf("%8.3f ", m[INDEX_2D(i, j)]);
//         }
//         printf("\n");
//     }
// }

void print_square_mat_flat(double *m, const uint32_t n) {
    uint16_t j_count = 0;
    for (uint16_t i = 0; i < n * n; i++) {
        if (i % n == 0) {
            j_count++;
            printf("\n");
        }
        printf("%8.3f ", m[i]);
    }
}

std::tuple<double_t, double_t, double_t> calculate_mse(
    double_t *const x, double_t *const y, const uint32_t *sz,
    bool skip_pading = true) {
    // required for IDX function...
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double_t max_err = 0.0;
    double_t min_err = __DBL_MAX__;
    double_t mse = 0.0;

    const uint32_t i_start = skip_pading ? helpers::padding : 0;
    const uint32_t j_start = skip_pading ? helpers::padding : 0;
    const uint32_t k_start = skip_pading ? helpers::padding : 0;

    const uint32_t i_end = skip_pading ? sz[0] - helpers::padding : sz[0];
    const uint32_t j_end = skip_pading ? sz[1] - helpers::padding : sz[1];
    const uint32_t k_end = skip_pading ? sz[2] - helpers::padding : sz[2];

    // std::cout << i_start << " " << i_end << std::endl;

    const uint32_t total_points =
        (i_end - i_start) * (j_end - j_start) * (k_end - k_start);

    // std::cout << total_points << std::endl;

    for (uint16_t k = k_start; k < k_end; k++) {
        for (uint16_t j = j_start; j < j_end; j++) {
            for (uint16_t i = i_start; i < i_end; i++) {
                double_t temp = (x[IDX(i, j, k)] - y[IDX(i, j, k)]) *
                                (x[IDX(i, j, k)] - y[IDX(i, j, k)]);

                if (temp > max_err) {
                    max_err = temp;
                }
                if (temp < min_err) {
                    min_err = temp;
                }

                mse += temp;
            }
        }
    }

    mse /= (total_points);

    return std::make_tuple(mse, min_err, max_err);
}

double_t calc_l2_norm(double_t *const u_var, double_t *const v_var,
                      const uint32_t n) {
    double_t sum = 0.0;

    for (uint32_t ii = 0; ii < n; ii++) {
        sum += (u_var[ii] - v_var[ii]) * (u_var[ii] - v_var[ii]);
    }

    return sqrt(sum);
}

void test_cfd_with_original_stencil(
    double_t *const u_var, const uint32_t *sz, const double *deltas,
    dendro_cfd::CompactFiniteDiff *cfd, double_t *u_dx = nullptr,
    double_t *u_dy = nullptr, double_t *u_dz = nullptr,
    double_t *u_dxx = nullptr, double_t *u_dyy = nullptr,
    double_t *u_dzz = nullptr) {
    // allocate a double block of memory
    const uint32_t totalSize = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace = new double_t[totalSize * 3 * 4];

    double_t *const derivx_stencil = deriv_workspace + 0 * totalSize;
    double_t *const derivy_stencil = deriv_workspace + 1 * totalSize;
    double_t *const derivz_stencil = deriv_workspace + 2 * totalSize;

    double_t *const derivx_cfd = deriv_workspace + 3 * totalSize;
    double_t *const derivy_cfd = deriv_workspace + 4 * totalSize;
    double_t *const derivz_cfd = deriv_workspace + 5 * totalSize;

    double_t *const derivxx_stencil = deriv_workspace + 6 * totalSize;
    double_t *const derivyy_stencil = deriv_workspace + 7 * totalSize;
    double_t *const derivzz_stencil = deriv_workspace + 8 * totalSize;

    double_t *const derivxx_cfd = deriv_workspace + 9 * totalSize;
    double_t *const derivyy_cfd = deriv_workspace + 10 * totalSize;
    double_t *const derivzz_cfd = deriv_workspace + 11 * totalSize;

    // then compute!

    void (*deriv_use_x)(double *const, const double *const, const double,
                        const unsigned int *, unsigned);
    void (*deriv_use_y)(double *const, const double *const, const double,
                        const unsigned int *, unsigned);
    void (*deriv_use_z)(double *const, const double *const, const double,
                        const unsigned int *, unsigned);

    if (helpers::padding == 2) {
        deriv_use_x = &deriv42_x_2pad;
        deriv_use_y = &deriv42_y_2pad;
        deriv_use_z = &deriv42_z_2pad;
    } else if (helpers::padding == 3) {
        deriv_use_x = &deriv644_x;
        deriv_use_y = &deriv644_y;
        deriv_use_z = &deriv644_z;
    } else if (helpers::padding == 4) {
        deriv_use_x = &deriv8666_x;
        deriv_use_y = &deriv8666_y;
        deriv_use_z = &deriv8666_z;
    } else {
        // NOTE: this is now 5 points, so 10th order stencils which we just...
        // don't have haha
        deriv_use_x = &deriv42_x;
        deriv_use_y = &deriv42_y;
        deriv_use_z = &deriv42_z;
    }

    void (*deriv_use_xx)(double *const, const double *const, const double,
                         const unsigned int *, unsigned);
    void (*deriv_use_yy)(double *const, const double *const, const double,
                         const unsigned int *, unsigned);
    void (*deriv_use_zz)(double *const, const double *const, const double,
                         const unsigned int *, unsigned);

    if (helpers::padding == 2) {
        std::cout << "2ND DERIV WILL USE deriv24 w/ 2padding!" << std::endl;
        deriv_use_xx = &deriv42_xx_2pad;
        deriv_use_yy = &deriv42_yy_2pad;
        deriv_use_zz = &deriv42_zz_2pad;
    } else if (helpers::padding == 3) {
        std::cout << "2ND DERIV WILL USE deriv644" << std::endl;
        deriv_use_xx = &deriv644_xx;
        deriv_use_yy = &deriv644_yy;
        deriv_use_zz = &deriv644_zz;
    } else if (helpers::padding == 4) {
        std::cout << "2ND DERIV WILL USE deriv8666" << std::endl;
        deriv_use_xx = &deriv8666_xx;
        deriv_use_yy = &deriv8666_yy;
        deriv_use_zz = &deriv8666_zz;
    } else {
        std::cout << "2ND DERIV WILL USE deriv24 w/ 2padding! (other default)"
                  << std::endl;
        // NOTE: this is now 5 points, so 10th order stencils which we just...
        // don't have haha
        deriv_use_xx = &deriv42_xx;
        deriv_use_yy = &deriv42_yy;
        deriv_use_zz = &deriv42_zz;
    }

    // const unsigned int bflag = (1 << 6) - 1;
    const unsigned int bflag = 0;

    deriv_use_x(derivx_stencil, u_var, deltas[0], sz, bflag);
    deriv_use_y(derivy_stencil, u_var, deltas[1], sz, bflag);
    deriv_use_z(derivz_stencil, u_var, deltas[2], sz, bflag);

    deriv_use_xx(derivxx_stencil, u_var, deltas[0], sz, bflag);
    deriv_use_yy(derivyy_stencil, u_var, deltas[1], sz, bflag);
    deriv_use_zz(derivzz_stencil, u_var, deltas[2], sz, bflag);

    double *u_var_copy = new double[totalSize];

    std::copy_n(u_var, totalSize, u_var_copy);
    std::cout << "\nL2 before filts: "
              << calc_l2_norm(u_var, u_var_copy, totalSize) << std::endl;

    cfd->filter_cfd_x(u_var_copy, derivx_cfd, deltas[0], sz, bflag);
    std::cout << "L2 after X Filt: "
              << calc_l2_norm(u_var, u_var_copy, totalSize) << std::endl;
    cfd->filter_cfd_y(u_var_copy, derivy_cfd, deltas[0], sz, bflag);
    std::cout << "L2 after Y Filt: "
              << calc_l2_norm(u_var, u_var_copy, totalSize) << std::endl;
    cfd->filter_cfd_z(u_var_copy, derivz_cfd, deltas[0], sz, bflag);
    std::cout << "L2 after Z Filt: "
              << calc_l2_norm(u_var, u_var_copy, totalSize) << std::endl;
    cfd->cfd_x(derivx_cfd, u_var_copy, deltas[0], sz, bflag);
    cfd->cfd_y(derivy_cfd, u_var_copy, deltas[1], sz, bflag);
    cfd->cfd_z(derivz_cfd, u_var_copy, deltas[2], sz, bflag);

    cfd->cfd_xx(derivxx_cfd, u_var_copy, deltas[0], sz, bflag);
    cfd->cfd_yy(derivyy_cfd, u_var_copy, deltas[1], sz, bflag);
    cfd->cfd_zz(derivzz_cfd, u_var_copy, deltas[2], sz, bflag);

    delete[] u_var_copy;

    // then compute the "error" difference between the two
    double_t min_x, max_x, mse_x, min_y, max_y, mse_y, min_z, max_z, mse_z;
    std::tie(mse_x, min_x, max_x) =
        calculate_mse(derivx_stencil, derivx_cfd, sz);
    std::tie(mse_y, min_y, max_y) =
        calculate_mse(derivy_stencil, derivy_cfd, sz);
    std::tie(mse_z, min_z, max_z) =
        calculate_mse(derivz_stencil, derivz_cfd, sz);

    std::cout << std::endl
              << GRN << "===COMPARING CFD TO STENCIL TEST RESULTS===" << NRM
              << std::endl;
    std::cout << "   deriv_x : mse = \t" << mse_x << "\tmin_err = \t" << min_x
              << "\tmax_err = \t" << max_x << std::endl;
    std::cout << "   deriv_y : mse = \t" << mse_y << "\tmin_err = \t" << min_y
              << "\tmax_err = \t" << max_y << std::endl;
    std::cout << "   deriv_z : mse = \t" << mse_z << "\tmin_err = \t" << min_z
              << "\tmax_err = \t" << max_z << std::endl;

    std::tie(mse_x, min_x, max_x) =
        calculate_mse(derivxx_stencil, derivxx_cfd, sz);
    std::tie(mse_y, min_y, max_y) =
        calculate_mse(derivyy_stencil, derivyy_cfd, sz);
    std::tie(mse_z, min_z, max_z) =
        calculate_mse(derivzz_stencil, derivzz_cfd, sz);
    std::cout << std::endl
              << GRN
              << "===COMPARING CFD TO STENCIL TEST RESULTS - 2ND ORDER==="
              << NRM << std::endl;
    std::cout << "   deriv_xx : mse = \t" << mse_x << "\tmin_err = \t" << min_x
              << "\tmax_err = \t" << max_x << std::endl;
    std::cout << "   deriv_yy : mse = \t" << mse_y << "\tmin_err = \t" << min_y
              << "\tmax_err = \t" << max_y << std::endl;
    std::cout << "   deriv_zz : mse = \t" << mse_z << "\tmin_err = \t" << min_z
              << "\tmax_err = \t" << max_z << std::endl;

    if (u_dx != nullptr && u_dy != nullptr && u_dz != nullptr) {
        // then compute the "error" difference between the two
        std::tie(mse_x, min_x, max_x) = calculate_mse(derivx_stencil, u_dx, sz);
        std::tie(mse_y, min_y, max_y) = calculate_mse(derivy_stencil, u_dy, sz);
        std::tie(mse_z, min_z, max_z) = calculate_mse(derivz_stencil, u_dz, sz);

        std::cout << std::endl
                  << GRN << "===COMPARING STENCIL TO TRUTH RESULTS===" << NRM
                  << std::endl;
        std::cout << "   deriv_x : mse = \t" << mse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_y : mse = \t" << mse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_z : mse = \t" << mse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;

        // then compute the "error" difference between the two
        std::tie(mse_x, min_x, max_x) = calculate_mse(derivx_cfd, u_dx, sz);
        std::tie(mse_y, min_y, max_y) = calculate_mse(derivy_cfd, u_dy, sz);
        std::tie(mse_z, min_z, max_z) = calculate_mse(derivz_cfd, u_dz, sz);

        std::cout << std::endl
                  << GRN << "===COMPARING CFD TO TRUTH RESULTS===" << NRM
                  << std::endl;
        std::cout << "   deriv_x : mse = \t" << mse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_y : mse = \t" << mse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_z : mse = \t" << mse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;
    }

    if (u_dxx != nullptr && u_dyy != nullptr && u_dzz != nullptr) {
        // then compute the "error" difference between the two
        std::tie(mse_x, min_x, max_x) =
            calculate_mse(derivxx_stencil, u_dxx, sz);
        std::tie(mse_y, min_y, max_y) =
            calculate_mse(derivyy_stencil, u_dyy, sz);
        std::tie(mse_z, min_z, max_z) =
            calculate_mse(derivzz_stencil, u_dzz, sz);

        std::cout << std::endl
                  << GRN
                  << "===COMPARING STENCIL TO TRUTH RESULTS - 2ND ORDER==="
                  << NRM << std::endl;
        std::cout << "   deriv_xx : mse = \t" << mse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_yy : mse = \t" << mse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_zz : mse = \t" << mse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;

        // then compute the "error" difference between the two
        std::tie(mse_x, min_x, max_x) = calculate_mse(derivxx_cfd, u_dxx, sz);
        std::tie(mse_y, min_y, max_y) = calculate_mse(derivyy_cfd, u_dyy, sz);
        std::tie(mse_z, min_z, max_z) = calculate_mse(derivzz_cfd, u_dzz, sz);

        std::cout << std::endl
                  << GRN
                  << "===COMPARING CFD TO TRUTH RESULTS - 2ND ORDER===" << NRM
                  << std::endl;
        std::cout << "   deriv_xx : mse = \t" << mse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_yy : mse = \t" << mse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_zz : mse = \t" << mse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;
    }

    // dump for visaliztaion 
    std::ofstream file("testOutputDZZ.bin", std::ios::binary | std::ios::out);
    file.write((char*)&totalSize, sizeof(totalSize));
    file.write((char*)u_dzz, sizeof(double) * totalSize);
    file.write((char*)derivzz_stencil, sizeof(double) * totalSize);
    file.close();

    delete[] deriv_workspace;
}

void profile_compact_stencils(double_t *const u_var, const uint32_t *sz,
                              const double *deltas,
                              dendro_cfd::CompactFiniteDiff *cfd,
                              uint32_t num_runs) {
    const uint32_t totalSize = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace = new double_t[totalSize * 3];

    double_t *const derivx_cfd = deriv_workspace + 0 * totalSize;
    double_t *const derivy_cfd = deriv_workspace + 1 * totalSize;
    double_t *const derivz_cfd = deriv_workspace + 2 * totalSize;

    uint32_t bflag = 0;

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        cfd->cfd_x(derivx_cfd, u_var, deltas[0], sz, bflag);
    }

    helpers::t_compact_deriv_x.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        cfd->cfd_x(derivx_cfd, u_var, deltas[0], sz, bflag);
    }
    helpers::t_compact_deriv_x.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        cfd->cfd_y(derivy_cfd, u_var, deltas[1], sz, bflag);
    }

    helpers::t_compact_deriv_y.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        cfd->cfd_y(derivy_cfd, u_var, deltas[1], sz, bflag);
    }
    helpers::t_compact_deriv_y.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        cfd->cfd_z(derivz_cfd, u_var, deltas[2], sz, bflag);
    }

    helpers::t_compact_deriv_z.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        cfd->cfd_z(derivz_cfd, u_var, deltas[2], sz, bflag);
    }
    helpers::t_compact_deriv_z.stop();

    delete[] deriv_workspace;
}

void profile_original_stencils(double_t *const u_var, const uint32_t *sz,
                               const double *deltas, uint32_t num_runs) {
    const uint32_t totalSize = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace = new double_t[totalSize * 3];

    double_t *const derivx_stencil = deriv_workspace + 0 * totalSize;
    double_t *const derivy_stencil = deriv_workspace + 1 * totalSize;
    double_t *const derivz_stencil = deriv_workspace + 2 * totalSize;

    void (*deriv_use_x)(double *const, const double *const, const double,
                        const unsigned int *, unsigned);
    void (*deriv_use_y)(double *const, const double *const, const double,
                        const unsigned int *, unsigned);
    void (*deriv_use_z)(double *const, const double *const, const double,
                        const unsigned int *, unsigned);

    if (helpers::padding == 2) {
        deriv_use_x = &deriv42_x_2pad;
        deriv_use_y = &deriv42_y_2pad;
        deriv_use_z = &deriv42_z_2pad;
    } else if (helpers::padding == 3) {
        deriv_use_x = &deriv644_x;
        deriv_use_y = &deriv644_y;
        deriv_use_z = &deriv644_z;
    } else if (helpers::padding == 4) {
        deriv_use_x = &deriv8642_x;
        deriv_use_y = &deriv8642_y;
        deriv_use_z = &deriv8642_z;
    } else {
        // NOTE: this is now 5 points, so 10th order stencils which we just...
        // don't have haha
        deriv_use_x = &deriv42_x;
        deriv_use_y = &deriv42_y;
        deriv_use_z = &deriv42_z;
    }

    uint32_t bflag = 0;

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv_use_x(derivx_stencil, u_var, deltas[0], sz, bflag);
    }

    helpers::t_deriv_x.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv_use_x(derivx_stencil, u_var, deltas[0], sz, bflag);
    }
    helpers::t_deriv_x.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv_use_y(derivy_stencil, u_var, deltas[1], sz, bflag);
    }

    helpers::t_deriv_y.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv_use_y(derivy_stencil, u_var, deltas[1], sz, bflag);
    }
    helpers::t_deriv_y.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv_use_z(derivz_stencil, u_var, deltas[2], sz, bflag);
    }

    helpers::t_deriv_z.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv_use_z(derivz_stencil, u_var, deltas[2], sz, bflag);
    }
    helpers::t_deriv_z.stop();

    delete[] deriv_workspace;
}

int main(int argc, char **argv) {
    uint32_t eleorder = 8;
    dendro_cfd::DerType deriv_type = dendro_cfd::CFD_KIM_O4;
    dendro_cfd::DerType2nd deriv_type_2nd = dendro_cfd::CFD2ND_KIM_O4;
    dendro_cfd::FilterType filter_type = dendro_cfd::FILT_NONE;
    uint32_t num_tests = 1000;
    uint32_t data_init = 2;

    if (argc == 1) {
        std::cout << "Using default parameters." << std::endl;

        std::cout << "If you wish to change the default parameters pass them "
                     "as command line arguments:"
                  << std::endl;
        std::cout << "<eleorder> <deriv_type> <deriv_type_2nd> <filter_type> "
                     "<num_tests> <data_init>"
                  << std::endl;
    }

    if (argc > 1) {
        // read in the element order
        eleorder = atoi(argv[1]);
    }
    if (argc > 2) {
        uint32_t temp_deriv_type = atoi(argv[2]);
        // read in the deriv_type we want to use
        // if it's set to 0, we'll do the default derivatives
        deriv_type = static_cast<dendro_cfd::DerType>(temp_deriv_type);
    }
    if (argc > 3) {
        uint32_t temp_deriv_type = atoi(argv[3]);
        // read in the deriv_type we want to use
        // if it's set to 0, we'll do the default derivatives
        deriv_type_2nd = static_cast<dendro_cfd::DerType2nd>(temp_deriv_type);
    }
    if (argc > 4) {
        uint32_t temp_filt_type = atoi(argv[4]);
        filter_type = static_cast<dendro_cfd::FilterType>(temp_filt_type);
    }
    if (argc > 5) {
        num_tests = atoi(argv[5]);
    }
    if (argc > 6) {
        data_init = atoi(argv[6]);
    }
    helpers::padding = eleorder >> 1u;

    std::cout << YLW
              << "Will run with the following user parameters:" << std::endl;
    std::cout << "    eleorder    -> " << eleorder << std::endl;
    std::cout << "    deriv_type  -> " << deriv_type << std::endl;
    std::cout << "    deriv_type_2nd  -> " << deriv_type_2nd << std::endl;
    std::cout << "    filter_type -> " << filter_type << std::endl;
    std::cout << "    num_tests   -> " << num_tests << std::endl;
    std::cout << "    data_init   -> " << data_init << std::endl;
    std::cout << "    INFO: padding is " << helpers::padding << NRM
              << std::endl;

    // the size in each dimension
    uint32_t fullwidth = 2 * eleorder + 1;
    uint32_t sz[3] = {fullwidth, fullwidth, fullwidth};

    // now we can actually build up our test block

    double_t *u_var = new double_t[sz[0] * sz[1] * sz[2]];
    double_t *u_dx_true = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dy_true = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dz_true = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dxx_true = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dyy_true = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dzz_true = new double_t[sz[0] * sz[1] * sz[2]]();

    double_t deltas[3] = {0.1, 0.05, 0.025};

    init_data(data_init, u_var, sz, deltas, u_dx_true, u_dy_true, u_dz_true,
              u_dxx_true, u_dyy_true, u_dzz_true);

    // print_3d_mat(u_var, fullwidth, fullwidth, fullwidth);

    // build up the cfd object
    dendro_cfd::CompactFiniteDiff cfd(fullwidth, helpers::padding, deriv_type,
                                      deriv_type_2nd, filter_type);

    // run a short test to see what the errors are
    test_cfd_with_original_stencil((double_t *const)u_var, sz, deltas, &cfd,
                                   u_dx_true, u_dy_true, u_dz_true, u_dxx_true,
                                   u_dyy_true, u_dzz_true);

    // profile_original_stencils((double_t *const)u_var, sz, deltas, num_tests);

    // profile_compact_stencils((double_t *const)u_var, sz, deltas, &cfd,
    //                          num_tests);

    // then print the profiler results
    // helpers::print_profiler_results(num_tests);

    // var cleanup
    delete[] u_var;
    delete[] u_dx_true;
    delete[] u_dy_true;
    delete[] u_dz_true;
    delete[] u_dxx_true;
    delete[] u_dyy_true;
    delete[] u_dzz_true;
}
