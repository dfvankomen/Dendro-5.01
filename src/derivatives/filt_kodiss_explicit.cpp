#include "derivatives/filt_kodiss_explicit.h"

namespace dendroderivs {
/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
template <unsigned int P>
void ko_deriv42_x(double *const Du, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 2 and 5 (for now)!");
    double pre_factor_6_dx  = -1.0 / 64.0 / dx;

    double smr3             = 59.0 / 48.0 * 64 * dx;
    double smr2             = 43.0 / 48.0 * 64 * dx;
    double smr1             = 49.0 / 48.0 * 64 * dx;
    double spr3             = smr3;
    double spr2             = smr2;
    double spr1             = smr1;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    // KO DISS only needs interior points
    static constexpr int ib = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    const int ie            = nx - ib;
    const int je            = ny - jb;
    const int ke            = nz - kb;

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
                Du[IDX(ib, j, k)] =
                    (u[IDX(ib + 3, j, k)] - 3.0 * u[IDX(ib + 2, j, k)] +
                     3.0 * u[IDX(ib + 1, j, k)] - u[IDX(ib, j, k)]) /
                    smr3;
                Du[IDX(ib + 1, j, k)] =
                    (u[IDX(ib + 4, j, k)] - 6.0 * u[IDX(ib + 3, j, k)] +
                     12.0 * u[IDX(ib + 2, j, k)] - 10.0 * u[IDX(ib + 1, j, k)] +
                     3.0 * u[IDX(ib, j, k)]) /
                    smr2;
                Du[IDX(ib + 2, j, k)] =
                    (u[IDX(ib + 5, j, k)] - 6.0 * u[IDX(ib + 4, j, k)] +
                     15.0 * u[IDX(ib + 3, j, k)] - 19.0 * u[IDX(ib + 2, j, k)] +
                     12.0 * u[IDX(ib + 1, j, k)] - 3.0 * u[IDX(ib, j, k)]) /
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
template <unsigned int P>
void ko_deriv42_y(double *const Du, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 2 and 5 (for now)!");
    double pre_factor_6_dy  = -1.0 / 64.0 / dy;

    double smr3             = 59.0 / 48.0 * 64 * dy;
    double smr2             = 43.0 / 48.0 * 64 * dy;
    double smr1             = 49.0 / 48.0 * 64 * dy;
    double spr3             = smr3;
    double spr2             = smr2;
    double spr1             = smr1;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    static constexpr int ib = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    const int ie            = nx - ib;
    const int je            = ny - jb;
    const int ke            = nz - kb;

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
                Du[IDX(i, jb, k)] =
                    (u[IDX(i, jb + 3, k)] - 3.0 * u[IDX(i, jb + 2, k)] +
                     3.0 * u[IDX(i, jb + 1, k)] - u[IDX(i, jb, k)]) /
                    smr3;
                Du[IDX(i, jb + 1, k)] =
                    (u[IDX(i, jb + 4, k)] - 6.0 * u[IDX(i, jb + 3, k)] +
                     12.0 * u[IDX(i, jb + 2, k)] - 10.0 * u[IDX(i, jb + 1, k)] +
                     3.0 * u[IDX(i, jb, k)]) /
                    smr2;
                Du[IDX(i, jb + 2, k)] =
                    (u[IDX(i, 8, k)] - 6.0 * u[IDX(i, jb + 4, k)] +
                     15.0 * u[IDX(i, jb + 3, k)] - 19.0 * u[IDX(i, jb + 2, k)] +
                     12.0 * u[IDX(i, jb + 1, k)] - 3.0 * u[IDX(i, jb, k)]) /
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
template <unsigned int P>
void ko_deriv42_z(double *const Du, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag) {
    double pre_factor_6_dz  = -1.0 / 64.0 / dz;

    double smr3             = 59.0 / 48.0 * 64 * dz;
    double smr2             = 43.0 / 48.0 * 64 * dz;
    double smr1             = 49.0 / 48.0 * 64 * dz;
    double spr3             = smr3;
    double spr2             = smr2;
    double spr1             = smr1;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    // KO DISS only needs interior points
    static constexpr int ib = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    const int ie            = nx - ib;
    const int je            = ny - jb;
    const int ke            = nz - kb;

    const int n             = nx * ny;

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
                Du[IDX(i, j, kb)] =
                    (u[IDX(i, j, kb + 3)] - 3.0 * u[IDX(i, j, kb + 2)] +
                     3.0 * u[IDX(i, j, kb + 1)] - u[IDX(i, j, kb)]) /
                    smr3;
                Du[IDX(i, j, kb + 1)] =
                    (u[IDX(i, j, kb + 4)] - 6.0 * u[IDX(i, j, kb + 3)] +
                     12.0 * u[IDX(i, j, kb + 2)] - 10.0 * u[IDX(i, j, kb + 1)] +
                     3.0 * u[IDX(i, j, kb)]) /
                    smr2;
                Du[IDX(i, j, kb + 2)] =
                    (u[IDX(i, j, kb + 5)] - 6.0 * u[IDX(i, j, kb + 4)] +
                     15.0 * u[IDX(i, j, kb + 3)] - 19.0 * u[IDX(i, j, kb + 2)] +
                     12.0 * u[IDX(i, j, kb + 1)] - 3.0 * u[IDX(i, j, kb)]) /
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

template void ko_deriv42_x<2>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_x<3>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_x<4>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_x<5>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);

template void ko_deriv42_y<2>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_y<3>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_y<4>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_y<5>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);

template void ko_deriv42_z<2>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_z<3>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_z<4>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_z<5>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);

}  // namespace dendroderivs
