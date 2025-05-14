#include <sys/types.h>

#include <cstdint>
#include <vector>

#include "derivatives.h"

#define IDX(i, j, k) ((i) + nx * ((j) + ny * (k)))

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
void ko_deriv42_x(double *const Du, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag) {
    double pre_factor_6_dx = -1.0 / 64.0 / dx;

    double smr3            = 59.0 / 48.0 * 64 * dx;
    double smr2            = 43.0 / 48.0 * 64 * dx;
    double smr1            = 49.0 / 48.0 * 64 * dx;
    double spr3            = smr3;
    double spr2            = smr2;
    double spr1            = smr1;

    const int nx           = sz[0];
    const int ny           = sz[1];
    const int nz           = sz[2];
    const int ib           = 3;
    const int jb           = 3;
    const int kb           = 3;
    const int ie           = sz[0] - 3;
    const int je           = sz[1] - 3;
    const int ke           = sz[2] - 3;

    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                Du[pp]       = pre_factor_6_dx *
                         (-u[pp - 3] + 6.0 * u[pp - 2] - 15.0 * u[pp - 1] +
                          20.0 * u[pp] - 15.0 * u[pp + 1] + 6.0 * u[pp + 2] -
                          u[pp + 3]);
            }
        }
    }

    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (int k = kb; k < ke; k++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int j = jb; j < je; j++) {
                Du[IDX(3, j, k)] = (u[IDX(6, j, k)] - 3.0 * u[IDX(5, j, k)] +
                                    3.0 * u[IDX(4, j, k)] - u[IDX(3, j, k)]) /
                                   smr3;
                Du[IDX(4, j, k)] =
                    (u[IDX(7, j, k)] - 6.0 * u[IDX(6, j, k)] +
                     12.0 * u[IDX(5, j, k)] - 10.0 * u[IDX(4, j, k)] +
                     3.0 * u[IDX(3, j, k)]) /
                    smr2;
                Du[IDX(5, j, k)] =
                    (u[IDX(8, j, k)] - 6.0 * u[IDX(7, j, k)] +
                     15.0 * u[IDX(6, j, k)] - 19.0 * u[IDX(5, j, k)] +
                     12.0 * u[IDX(4, j, k)] - 3.0 * u[IDX(3, j, k)]) /
                    smr1;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (int k = kb; k < ke; k++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int j = jb; j < je; j++) {
                Du[IDX(ie - 3, j, k)] =
                    (u[IDX(ie - 6, j, k)] - 6.0 * u[IDX(ie - 5, j, k)] +
                     15.0 * u[IDX(ie - 4, j, k)] - 19.0 * u[IDX(ie - 3, j, k)] +
                     12.0 * u[IDX(ie - 2, j, k)] - 3.0 * u[IDX(ie - 1, j, k)]) /
                    spr1;

                Du[IDX(ie - 2, j, k)] =
                    (u[IDX(ie - 5, j, k)] - 6.0 * u[IDX(ie - 4, j, k)] +
                     12.0 * u[IDX(ie - 3, j, k)] - 10.0 * u[IDX(ie - 2, j, k)] +
                     3.0 * u[IDX(ie - 1, j, k)]) /
                    spr2;

                Du[IDX(ie - 1, j, k)] =
                    (u[IDX(ie - 4, j, k)] - 3.0 * u[IDX(ie - 3, j, k)] +
                     3.0 * u[IDX(ie - 2, j, k)] - u[IDX(ie - 1, j, k)]) /
                    spr3;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Du[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
void ko_deriv42_y(double *const Du, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag) {
    double pre_factor_6_dy = -1.0 / 64.0 / dy;

    double smr3            = 59.0 / 48.0 * 64 * dy;
    double smr2            = 43.0 / 48.0 * 64 * dy;
    double smr1            = 49.0 / 48.0 * 64 * dy;
    double spr3            = smr3;
    double spr2            = smr2;
    double spr1            = smr1;

    const int nx           = sz[0];
    const int ny           = sz[1];
    const int nz           = sz[2];
    const int ib           = 3;
    const int jb           = 3;
    const int kb           = 3;
    const int ie           = sz[0] - 3;
    const int je           = sz[1] - 3;
    const int ke           = sz[2] - 3;

    for (int k = kb; k < ke; k++) {
        for (int i = ib; i < ie; i++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int j = jb; j < je; j++) {
                const int pp = IDX(i, j, k);
                Du[pp]       = pre_factor_6_dy *
                         (-u[pp - 3 * nx] + 6.0 * u[pp - 2 * nx] -
                          15.0 * u[pp - nx] + 20.0 * u[pp] - 15.0 * u[pp + nx] +
                          6.0 * u[pp + 2 * nx] - u[pp + 3 * nx]);
            }
        }
    }

    if (bflag & (1u << OCT_DIR_DOWN)) {
        for (int k = kb; k < ke; k++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, 3, k)] = (u[IDX(i, 6, k)] - 3.0 * u[IDX(i, 5, k)] +
                                    3.0 * u[IDX(i, 4, k)] - u[IDX(i, 3, k)]) /
                                   smr3;
                Du[IDX(i, 4, k)] =
                    (u[IDX(i, 7, k)] - 6.0 * u[IDX(i, 6, k)] +
                     12.0 * u[IDX(i, 5, k)] - 10.0 * u[IDX(i, 4, k)] +
                     3.0 * u[IDX(i, 3, k)]) /
                    smr2;
                Du[IDX(i, 5, k)] =
                    (u[IDX(i, 8, k)] - 6.0 * u[IDX(i, 7, k)] +
                     15.0 * u[IDX(i, 6, k)] - 19.0 * u[IDX(i, 5, k)] +
                     12.0 * u[IDX(i, 4, k)] - 3.0 * u[IDX(i, 3, k)]) /
                    smr1;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_UP)) {
        for (int k = kb; k < ke; k++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, je - 3, k)] =
                    (u[IDX(i, je - 6, k)] - 6.0 * u[IDX(i, je - 5, k)] +
                     15.0 * u[IDX(i, je - 4, k)] - 19.0 * u[IDX(i, je - 3, k)] +
                     12.0 * u[IDX(i, je - 2, k)] - 3.0 * u[IDX(i, je - 1, k)]) /
                    spr1;

                Du[IDX(i, je - 2, k)] =
                    (u[IDX(i, je - 5, k)] - 6.0 * u[IDX(i, je - 4, k)] +
                     12.0 * u[IDX(i, je - 3, k)] - 10.0 * u[IDX(i, je - 2, k)] +
                     3.0 * u[IDX(i, je - 1, k)]) /
                    spr2;

                Du[IDX(i, je - 1, k)] =
                    (u[IDX(i, je - 4, k)] - 3.0 * u[IDX(i, je - 3, k)] +
                     3.0 * u[IDX(i, je - 2, k)] - u[IDX(i, je - 1, k)]) /
                    spr3;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Du[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
void ko_deriv42_z(double *const Du, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag) {
    double pre_factor_6_dz = -1.0 / 64.0 / dz;

    double smr3            = 59.0 / 48.0 * 64 * dz;
    double smr2            = 43.0 / 48.0 * 64 * dz;
    double smr1            = 49.0 / 48.0 * 64 * dz;
    double spr3            = smr3;
    double spr2            = smr2;
    double spr1            = smr1;

    const int nx           = sz[0];
    const int ny           = sz[1];
    const int nz           = sz[2];
    const int ib           = 3;
    const int jb           = 3;
    const int kb           = 3;
    const int ie           = sz[0] - 3;
    const int je           = sz[1] - 3;
    const int ke           = sz[2] - 3;

    const int n            = nx * ny;

    for (int j = jb; j < je; j++) {
        for (int i = ib; i < ie; i++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int k = kb; k < ke; k++) {
                const int pp = IDX(i, j, k);
                Du[pp]       = pre_factor_6_dz *
                         (-u[pp - 3 * n] + 6.0 * u[pp - 2 * n] -
                          15.0 * u[pp - n] + 20.0 * u[pp] - 15.0 * u[pp + n] +
                          6.0 * u[pp + 2 * n] - u[pp + 3 * n]);
            }
        }
    }

    if (bflag & (1u << OCT_DIR_BACK)) {
        for (int j = jb; j < je; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, j, 3)] = (u[IDX(i, j, 6)] - 3.0 * u[IDX(i, j, 5)] +
                                    3.0 * u[IDX(i, j, 4)] - u[IDX(i, j, 3)]) /
                                   smr3;
                Du[IDX(i, j, 4)] =
                    (u[IDX(i, j, 7)] - 6.0 * u[IDX(i, j, 6)] +
                     12.0 * u[IDX(i, j, 5)] - 10.0 * u[IDX(i, j, 4)] +
                     3.0 * u[IDX(i, j, 3)]) /
                    smr2;
                Du[IDX(i, j, 5)] =
                    (u[IDX(i, j, 8)] - 6.0 * u[IDX(i, j, 7)] +
                     15.0 * u[IDX(i, j, 6)] - 19.0 * u[IDX(i, j, 5)] +
                     12.0 * u[IDX(i, j, 4)] - 3.0 * u[IDX(i, j, 3)]) /
                    smr1;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_FRONT)) {
        for (int j = jb; j < je; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, j, ke - 3)] =
                    (u[IDX(i, j, ke - 6)] - 6.0 * u[IDX(i, j, ke - 5)] +
                     15.0 * u[IDX(i, j, ke - 4)] - 19.0 * u[IDX(i, j, ke - 3)] +
                     12.0 * u[IDX(i, j, ke - 2)] - 3.0 * u[IDX(i, j, ke - 1)]) /
                    spr1;

                Du[IDX(i, j, ke - 2)] =
                    (u[IDX(i, j, ke - 5)] - 6.0 * u[IDX(i, j, ke - 4)] +
                     12.0 * u[IDX(i, j, ke - 3)] - 10.0 * u[IDX(i, j, ke - 2)] +
                     3.0 * u[IDX(i, j, ke - 1)]) /
                    spr2;

                Du[IDX(i, j, ke - 1)] =
                    (u[IDX(i, j, ke - 4)] - 3.0 * u[IDX(i, j, ke - 3)] +
                     3.0 * u[IDX(i, j, ke - 2)] - u[IDX(i, j, ke - 1)]) /
                    spr3;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Du[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

void boris_init(double_t *u_var, const double_t *corner, const uint32_t *sz,
                const double_t *deltas) {
    const double_t x_start   = corner[0];
    const double_t y_start   = corner[1];
    const double_t z_start   = corner[2];
    const double_t dx        = deltas[0];
    const double_t dy        = deltas[1];
    const double_t dz        = deltas[2];

    const double_t amplitude = 0.01;

    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        double_t z = z_start + k * dz;
        for (uint16_t j = 0; j < ny; j++) {
            double_t y = y_start + j * dy;
            for (uint16_t i = 0; i < nx; i++) {
                double_t x = x_start + i * dx;
                u_var[IDX(i, j, k)] =
                    0.5 * exp(-sin(2 * x) - sin(2 * y) - sin(2 * z));

                if (k == 0 && j == 0) {
                    // std::cout << x << " ";
                }
            }
            if (k == 0) {
                // std::cout << y << " ";
            }
        }
        // std::cout << z << " ";
    }
    // std::cout << std::endl;
}

void heat_eq_update(double *du, double_t *duxx, double_t *duyy, double_t *duzz,
                    const uint32_t *sz, uint16_t pw) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (uint16_t k = pw; k < sz[2] - pw; k++) {
        for (uint16_t j = pw; j < sz[1] - pw; j++) {
            for (uint16_t i = pw; i < sz[0] - pw; i++) {
                du[IDX(i, j, k)] = duxx[IDX(i, j, k)] + duyy[IDX(i, j, k)] +
                                   duzz[IDX(i, j, k)];
            }
        }
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

std::tuple<double_t, double_t, double_t> calculate_rmse(
    const double_t *const x, const double_t *const y, const uint32_t *sz,
    const uint32_t padding, bool skip_pading = true) {
    // required for IDX function...
    const unsigned int nx  = sz[0];
    const unsigned int ny  = sz[1];
    const unsigned int nz  = sz[2];

    double_t max_err       = 0.0;
    double_t min_err       = __DBL_MAX__;
    double_t rmse          = 0.0;

    const uint32_t i_start = skip_pading ? padding : 0;
    const uint32_t j_start = skip_pading ? padding : 0;
    const uint32_t k_start = skip_pading ? padding : 0;

    const uint32_t i_end   = skip_pading ? sz[0] - padding : sz[0];
    const uint32_t j_end   = skip_pading ? sz[1] - padding : sz[1];
    const uint32_t k_end   = skip_pading ? sz[2] - padding : sz[2];

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

                rmse += temp;
            }
        }
    }

    rmse /= (total_points);

    return std::make_tuple(sqrt(rmse), min_err, max_err);
}

void do_old_kodiss(double_t *u_var, double_t *u_rhs, double_t *u_rhs_output,
                   const double sigma, const uint32_t padding,
                   const double *deltas, const uint32_t *sz,
                   const uint32_t bflag) {
    const unsigned int nx        = sz[0];
    const unsigned int ny        = sz[1];
    const unsigned int nz        = sz[2];

    std::vector<double_t> u_dx_t = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dy_t = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dz_t = std::vector<double_t>(sz[0] * sz[1] * sz[2]);

    ko_deriv42_x(u_dx_t.data(), u_var, deltas[0], sz, bflag);
    ko_deriv42_y(u_dy_t.data(), u_var, deltas[1], sz, bflag);
    ko_deriv42_z(u_dz_t.data(), u_var, deltas[2], sz, bflag);

    // compared to zero
    std::vector<double_t> temp =
        std::vector<double_t>(sz[0] * sz[1] * sz[2], 0.0);
    double_t rmse_val, min_val, max_val;

    std::tie(rmse_val, min_val, max_val) =
        calculate_rmse(u_dx_t.data(), temp.data(), sz, padding);
    std::cout << "OLD-X 'RMSE': " << rmse_val << "    - min : " << min_val
              << " max : " << max_val << std::endl;
    std::tie(rmse_val, min_val, max_val) =
        calculate_rmse(u_dy_t.data(), temp.data(), sz, padding);
    std::cout << "OLD-Y 'RMSE': " << rmse_val << "    - min : " << min_val
              << " max : " << max_val << std::endl;
    std::tie(rmse_val, min_val, max_val) =
        calculate_rmse(u_dz_t.data(), temp.data(), sz, padding);
    std::cout << "OLD-Z 'RMSE': " << rmse_val << "    - min : " << min_val
              << " max : " << max_val << std::endl;

    for (unsigned int k = padding; k < nz - padding; k++) {
        for (unsigned int j = padding; j < ny - padding; j++) {
            for (unsigned int i = padding; i < nx - padding; i++) {
                const unsigned int pp = i + nx * (j + ny * k);
                u_rhs_output[pp] =
                    u_rhs[pp] + sigma * (u_dx_t[pp] + u_dy_t[pp] + u_dz_t[pp]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    double_t rmse_val, min_val, max_val;
    const uint32_t eleorder      = 6;

    const uint32_t padding       = eleorder / 2;

    const uint32_t fullwidth     = 2 * eleorder + 1;

    uint32_t sz[3]               = {fullwidth, fullwidth, fullwidth};

    std::vector<double_t> u_var  = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_rhs  = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dx_t = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dy_t = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dz_t = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dx_c = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dy_c = std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_dz_c = std::vector<double_t>(sz[0] * sz[1] * sz[2]);

    std::cout << "Building derivative information..." << std::endl;
    dendroderivs::DendroDerivatives deriv("E6", "E6", eleorder, {}, {}, 0, 0,
                                          "none", "none", {}, {}, "KO4");

    // initial data
    double_t deltas[3] = {0.02, 0.01, 0.001};
    double_t corner[3] = {0.0, 0.0, 0.0};

    uint32_t bflag     = 0;

    boris_init(u_var.data(), corner, sz, deltas);

    // then calculate the derivatives
    deriv.deriv_xx(u_dx_c.data(), u_var.data(), deltas[0], sz, bflag);
    deriv.deriv_yy(u_dy_c.data(), u_var.data(), deltas[1], sz, bflag);
    deriv.deriv_zz(u_dz_c.data(), u_var.data(), deltas[1], sz, bflag);

    heat_eq_update(u_rhs.data(), u_dx_c.data(), u_dy_c.data(), u_dz_c.data(),
                   sz, padding);

    const double_t sigma = 0.01;

    std::vector<double_t> u_rhs_kodiss_old =
        std::vector<double_t>(sz[0] * sz[1] * sz[2]);
    std::vector<double_t> u_rhs_kodiss_new = u_rhs;

    // check the URHS
    std::vector<double_t> temp =
        std::vector<double_t>(sz[0] * sz[1] * sz[2], 0.0);
    std::tie(rmse_val, min_val, max_val) =
        calculate_rmse(temp.data(), u_rhs.data(), sz, padding);
    std::cout << "URHS: " << rmse_val << "    - min error: " << min_val
              << " max error: " << max_val << std::endl;

    // do the old stuff
    do_old_kodiss(u_var.data(), u_rhs.data(), u_rhs_kodiss_old.data(), sigma,
                  padding, deltas, sz, bflag);

    // do the new one
    deriv.filter(u_var.data(), u_rhs_kodiss_new.data(), u_dx_t.data(),
                 u_dy_t.data(), u_dz_t.data(), deltas[0], deltas[1], deltas[2],
                 sigma, sz, bflag);

    std::tie(rmse_val, min_val, max_val) =
        calculate_rmse(u_rhs.data(), u_rhs_kodiss_old.data(), sz, padding);
    std::cout << "URHS/KODISSOLD DIFFERENCE RMSE: " << rmse_val
              << "    - min error: " << min_val << " max error: " << max_val
              << std::endl;

    std::tie(rmse_val, min_val, max_val) =
        calculate_rmse(u_rhs.data(), u_rhs_kodiss_new.data(), sz, padding);
    std::cout << "URHS/KODISSNEW DIFFERENCE RMSE: " << rmse_val
              << "    - min error: " << min_val << " max error: " << max_val
              << std::endl;

    std::tie(rmse_val, min_val, max_val) = calculate_rmse(
        u_rhs_kodiss_old.data(), u_rhs_kodiss_new.data(), sz, padding);
    std::cout << "KODISS NEW/OLD DIFFERENCE RMSE: " << rmse_val
              << "    - min error: " << min_val << " max error: " << max_val
              << std::endl;

    return 0;
}
