#include "derivatives/filt_kodiss_explicit.h"

namespace dendroderivs {

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
template <unsigned int P>
void ko_deriv21_x(double *const Du, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 3 and 5 (for now)!");
    // According to
    // https://einsteintoolkit.org/thornguide/CactusNumerical/Dissipation/documentation.html#Xkreiss-oliger
    // this order has a -1 leading
    // The internal stencils are the "pure" stencils and the factors out front
    // will be negative
    double pre_factor_6_dx  = -1.0 / 16.0 / dx;

    double smr2             = 4 * dx;
    double smr1             = 4 * dx;
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
                         (1.0 * u[pp - 2] - 4.0 * u[pp - 1] + 6.0 * u[pp] -
                          4.0 * u[pp + 1] + 1.0 * u[pp + 2]);
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
                    (u[IDX(ib, j, k)] - 2.0 * u[IDX(ib + 1, j, k)] +
                     u[IDX(ib + 2, j, k)]) /
                    smr2;
                Du[IDX(ib, j, k)] =
                    (u[IDX(ib, j, k)] - 2.0 * u[IDX(ib + 1, j, k)] +
                     u[IDX(ib + 2, j, k)]) /
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
                Du[IDX(ie - 2, j, k)] =
                    (u[IDX(ie - 3, j, k)] - 2.0 * u[IDX(ie - 2, j, k)] +
                     u[IDX(ie - 1, j, k)]) /
                    spr1;

                Du[IDX(ie - 1, j, k)] =
                    (u[IDX(ie - 3, j, k)] - 2.0 * u[IDX(ie - 2, j, k)] +
                     u[IDX(ie - 1, j, k)]) /
                    spr2;
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
void ko_deriv21_y(double *const Du, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 2 and 5 (for now)!");
    // According to
    // https://einsteintoolkit.org/thornguide/CactusNumerical/Dissipation/documentation.html#Xkreiss-oliger
    // this order has a -1 leading
    // The internal stencils are the "pure" stencils and the factors out front
    // will be negative
    double pre_factor_6_dy  = -1.0 / 16.0 / dy;

    double smr2             = 4 * dy;
    double smr1             = 4 * dy;
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
                Du[pp] =
                    pre_factor_6_dy *
                    (1.0 * u[pp - 2 * nx] - 4.0 * u[pp - 1 * nx] + 6.0 * u[pp] -
                     4.0 * u[pp + 1 * nx] + 1.0 * u[pp + 2 * nx]);
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
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, jb, k)] =
                    (u[IDX(i, jb, k)] - 2.0 * u[IDX(i, jb + 1, k)] +
                     u[IDX(i, jb + 2, k)]) /
                    smr2;
                Du[IDX(i, jb + 1, k)] =
                    (u[IDX(i, jb, k)] - 2.0 * u[IDX(i, jb + 1, k)] +
                     u[IDX(i, jb + 2, k)]) /
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
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, je - 2, k)] =
                    (u[IDX(i, je - 3, k)] - 2.0 * u[IDX(i, je - 2, k)] +
                     u[IDX(i, je - 1, k)]) /
                    spr1;

                Du[IDX(i, je - 1, k)] =
                    (u[IDX(i, je - 3, k)] - 2.0 * u[IDX(i, je - 2, k)] +
                     u[IDX(i, je - 1, k)]) /
                    spr2;
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
void ko_deriv21_z(double *const Du, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 2 and 5 (for now)!");
    // According to
    // https://einsteintoolkit.org/thornguide/CactusNumerical/Dissipation/documentation.html#Xkreiss-oliger
    // this order has a -1 leading
    // The internal stencils are the "pure" stencils and the factors out front
    // will be negative
    double pre_factor_6_dz  = -1.0 / 16.0 / dz;

    double smr2             = 4 * dz;
    double smr1             = 4 * dz;
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
                Du[pp] =
                    pre_factor_6_dz *
                    (1.0 * u[pp - 2 * n] - 4.0 * u[pp - 1 * n] + 6.0 * u[pp] -
                     4.0 * u[pp + 1 * n] + 1.0 * u[pp + 2 * n]);
            }
        }
    }

    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (int j = jb; j < je; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, j, kb)] =
                    (u[IDX(i, j, kb)] - 2.0 * u[IDX(i, j, kb + 1)] +
                     u[IDX(i, j, kb + 2)]) /
                    smr2;
                Du[IDX(i, j, kb + 1)] =
                    (u[IDX(i, j, kb)] - 2.0 * u[IDX(i, j, kb + 1)] +
                     u[IDX(i, j, kb + 2)]) /
                    smr1;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (int j = jb; j < je; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                Du[IDX(i, j, ke - 2)] =
                    (u[IDX(i, j, ke - 3)] - 2.0 * u[IDX(i, j, ke - 2)] +
                     u[IDX(i, j, ke - 1)]) /
                    spr1;

                Du[IDX(i, j, ke - 1)] =
                    (u[IDX(i, j, ke - 3)] - 2.0 * u[IDX(i, j, ke - 2)] +
                     u[IDX(i, j, ke - 1)]) /
                    spr2;
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

template void ko_deriv21_x<2>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_x<3>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_x<4>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_x<5>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);

template void ko_deriv21_y<2>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_y<3>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_y<4>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_y<5>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);

template void ko_deriv21_z<2>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_z<3>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_z<4>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv21_z<5>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
template <unsigned int P>
void ko_deriv42_x(double *const Du, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 3 && P <= 5, "P must be between 3 and 5 (for now)!");
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
    static constexpr int ib = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 3) ? 3 : (P == 4) ? 4 : 5;
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
    static_assert(P >= 3 && P <= 5, "P must be between 3 and 5 (for now)!");
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

    static constexpr int ib = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 3) ? 3 : (P == 4) ? 4 : 5;
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
                    (u[IDX(i, jb + 5, k)] - 6.0 * u[IDX(i, jb + 4, k)] +
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
    static_assert(P >= 3 && P <= 5, "P must be between 3 and 5 (for now)!");
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
    static constexpr int ib = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 3) ? 3 : (P == 4) ? 4 : 5;
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

template void ko_deriv42_x<3>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_x<4>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_x<5>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
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

template void ko_deriv42_z<3>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_z<4>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv42_z<5>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
template <unsigned int P>
void ko_deriv64_x(double *const Du, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be between 2 and 5 (for now)!");
    double pre_factor_8_dx  = -1.0 / 256.0 / dx;

    double smr4             = 17.0 / 48.0 * 256 * dx;
    double smr3             = 59.0 / 48.0 * 256 * dx;
    double smr2             = 43.0 / 48.0 * 256 * dx;
    double smr1             = 49.0 / 48.0 * 256 * dx;
    double spr4             = smr4;
    double spr3             = smr3;
    double spr2             = smr2;
    double spr1             = smr1;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    // KO DISS only needs interior points
    static constexpr int ib = (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 4) ? 4 : 5;
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
                Du[pp]       = pre_factor_8_dx *
                         (u[pp - 4] - 8.0 * u[pp - 3] + 28.0 * u[pp - 2] -
                          56.0 * u[pp - 1] + 70.0 * u[pp] - 56.0 * u[pp + 1] +
                          28.0 * u[pp + 2] - 8.0 * u[pp + 3] + u[pp + 4]);
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
                    (-u[IDX(ib, j, k)] + 4.0 * u[IDX(ib + 1, j, k)] -
                     6.0 * u[IDX(ib + 2, j, k)] + 4.0 * u[IDX(ib + 3, j, k)] -
                     u[IDX(ib + 4, j, k)]) /
                    smr4;

                Du[IDX(ib + 1, j, k)] =
                    (3.0 * u[IDX(ib, j, k)] - 11.0 * u[IDX(ib + 1, j, k)] +
                     15.0 * u[IDX(ib + 2, j, k)] - 9.0 * u[IDX(ib + 3, j, k)] +
                     2.0 * u[IDX(ib + 4, j, k)]) /
                    smr3;

                Du[IDX(ib + 2, j, k)] =
                    (-3.0 * u[IDX(ib, j, k)] + 9.0 * u[IDX(ib + 1, j, k)] -
                     8.0 * u[IDX(ib + 2, j, k)] + 3.0 * u[IDX(ib + 4, j, k)] -
                     u[IDX(ib + 5, j, k)]) /
                    smr2;

                Du[IDX(ib + 3, j, k)] =
                    (u[IDX(ib, j, k)] - u[IDX(ib + 1, j, k)] -
                     6.0 * u[IDX(ib + 2, j, k)] + 15.0 * u[IDX(ib + 3, j, k)] -
                     14.0 * u[IDX(ib + 4, j, k)] + 6.0 * u[IDX(ib + 5, j, k)] -
                     u[IDX(ib + 6, j, k)]) /
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
                Du[IDX(ie - 4, j, k)] =
                    (-u[IDX(ie - 7, j, k)] + 6.0 * u[IDX(ie - 6, j, k)] -
                     14.0 * u[IDX(ie - 5, j, k)] + 15.0 * u[IDX(ie - 4, j, k)] -
                     6.0 * u[IDX(ie - 3, j, k)] - u[IDX(ie - 2, j, k)] +
                     u[IDX(ie - 1, j, k)]) /
                    spr1;

                Du[IDX(ie - 3, j, k)] =
                    (-u[IDX(ie - 6, j, k)] + 3.0 * u[IDX(ie - 5, j, k)] -
                     8.0 * u[IDX(ie - 3, j, k)] + 9.0 * u[IDX(ie - 2, j, k)] -
                     3.0 * u[IDX(ie - 1, j, k)]) /
                    spr2;

                Du[IDX(ie - 2, j, k)] =
                    (2.0 * u[IDX(ie - 5, j, k)] - 9.0 * u[IDX(ie - 4, j, k)] +
                     15.0 * u[IDX(ie - 3, j, k)] - 11.0 * u[IDX(ie - 2, j, k)] +
                     3.0 * u[IDX(ie - 1, j, k)]) /
                    spr3;

                Du[IDX(ie - 1, j, k)] =
                    (-u[IDX(ie - 5, j, k)] + 4.0 * u[IDX(ie - 4, j, k)] -
                     6.0 * u[IDX(ie - 3, j, k)] + 4.0 * u[IDX(ie - 2, j, k)] -
                     u[IDX(ie - 1, j, k)]) /
                    spr4;
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
void ko_deriv64_y(double *const Du, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag) {
    double pre_factor_8_dy  = -1.0 / 256.0 / dy;

    double smr4             = 17.0 / 48.0 * 256 * dy;
    double smr3             = 59.0 / 48.0 * 256 * dy;
    double smr2             = 43.0 / 48.0 * 256 * dy;
    double smr1             = 49.0 / 48.0 * 256 * dy;
    double spr4             = smr4;
    double spr3             = smr3;
    double spr2             = smr2;
    double spr1             = smr1;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    // KO DISS only needs interior points
    static constexpr int ib = (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 4) ? 4 : 5;
    const int ie            = nx - ib;
    const int je            = ny - jb;
    const int ke            = nz - kb;

    const int n             = nx;

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
                Du[pp] =
                    pre_factor_8_dy *
                    (u[pp - 4 * n] - 8.0 * u[pp - 3 * n] +
                     28.0 * u[pp - 2 * n] - 56.0 * u[pp - n] + 70.0 * u[pp] -
                     56.0 * u[pp + n] + 28.0 * u[pp + 2 * n] -
                     8.0 * u[pp + 3 * n] + u[pp + 4 * n]);
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
                    (-u[IDX(i, jb, k)] + 4.0 * u[IDX(i, jb + 1, k)] -
                     6.0 * u[IDX(i, jb + 2, k)] + 4.0 * u[IDX(i, jb + 3, k)] -
                     u[IDX(i, jb + 4, k)]) /
                    smr4;

                Du[IDX(i, jb + 1, k)] =
                    (3.0 * u[IDX(i, jb, k)] - 11.0 * u[IDX(i, jb + 1, k)] +
                     15.0 * u[IDX(i, jb + 2, k)] - 9.0 * u[IDX(i, jb + 3, k)] +
                     2.0 * u[IDX(i, jb + 4, k)]) /
                    smr3;

                Du[IDX(i, jb + 2, k)] =
                    (-3.0 * u[IDX(i, jb, k)] + 9.0 * u[IDX(i, jb + 1, k)] -
                     8.0 * u[IDX(i, jb + 2, k)] + 3.0 * u[IDX(i, jb + 4, k)] -
                     u[IDX(i, jb + 5, k)]) /
                    smr2;

                Du[IDX(i, jb + 3, k)] =
                    (u[IDX(i, jb, k)] - u[IDX(i, jb + 1, k)] -
                     6.0 * u[IDX(i, jb + 2, k)] + 15.0 * u[IDX(i, jb + 3, k)] -
                     14.0 * u[IDX(i, jb + 4, k)] + 6.0 * u[IDX(i, jb + 5, k)] -
                     u[IDX(i, jb + 6, k)]) /
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
                Du[IDX(i, je - 4, k)] =
                    (-u[IDX(i, je - 7, k)] + 6.0 * u[IDX(i, je - 6, k)] -
                     14.0 * u[IDX(i, je - 5, k)] + 15.0 * u[IDX(i, je - 4, k)] -
                     6.0 * u[IDX(i, je - 3, k)] - u[IDX(i, je - 2, k)] +
                     u[IDX(i, je - 1, k)]) /
                    spr1;

                Du[IDX(i, je - 3, k)] =
                    (-u[IDX(i, je - 6, k)] + 3.0 * u[IDX(i, je - 5, k)] -
                     8.0 * u[IDX(i, je - 3, k)] + 9.0 * u[IDX(i, je - 2, k)] -
                     3.0 * u[IDX(i, je - 1, k)]) /
                    spr2;

                Du[IDX(i, je - 2, k)] =
                    (2.0 * u[IDX(i, je - 5, k)] - 9.0 * u[IDX(i, je - 4, k)] +
                     15.0 * u[IDX(i, je - 3, k)] - 11.0 * u[IDX(i, je - 2, k)] +
                     3.0 * u[IDX(i, je - 1, k)]) /
                    spr3;

                Du[IDX(i, je - 1, k)] =
                    (-u[IDX(i, je - 5, k)] + 4.0 * u[IDX(i, je - 4, k)] -
                     6.0 * u[IDX(i, je - 3, k)] + 4.0 * u[IDX(i, je - 2, k)] -
                     u[IDX(i, je - 1, k)]) /
                    spr4;
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
void ko_deriv64_z(double *const Du, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag) {
    double pre_factor_8_dz  = -1.0 / 256.0 / dz;

    double smr4             = 17.0 / 48.0 * 256 * dz;
    double smr3             = 59.0 / 48.0 * 256 * dz;
    double smr2             = 43.0 / 48.0 * 256 * dz;
    double smr1             = 49.0 / 48.0 * 256 * dz;
    double spr4             = smr4;
    double spr3             = smr3;
    double spr2             = smr2;
    double spr1             = smr1;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    // KO DISS only needs interior points
    static constexpr int ib = (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 4) ? 4 : 5;
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
                Du[pp] =
                    pre_factor_8_dz *
                    (u[pp - 4 * n] - 8.0 * u[pp - 3 * n] +
                     28.0 * u[pp - 2 * n] - 56.0 * u[pp - n] + 70.0 * u[pp] -
                     56.0 * u[pp + n] + 28.0 * u[pp + 2 * n] -
                     8.0 * u[pp + 3 * n] + u[pp + 4 * n]);
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
                    (-u[IDX(i, j, kb)] + 4.0 * u[IDX(i, j, kb + 1)] -
                     6.0 * u[IDX(i, j, kb + 2)] + 4.0 * u[IDX(i, j, kb + 3)] -
                     u[IDX(i, j, kb + 4)]) /
                    smr4;

                Du[IDX(i, j, kb + 1)] =
                    (3.0 * u[IDX(i, j, jb)] - 11.0 * u[IDX(i, j, kb + 1)] +
                     15.0 * u[IDX(i, j, kb + 2)] - 9.0 * u[IDX(i, j, kb + 3)] +
                     2.0 * u[IDX(i, j, kb + 4)]) /
                    smr3;

                Du[IDX(i, j, kb + 2)] =
                    (-3.0 * u[IDX(i, j, jb)] + 9.0 * u[IDX(i, j, kb + 1)] -
                     8.0 * u[IDX(i, j, kb + 2)] + 3.0 * u[IDX(i, j, kb + 4)] -
                     u[IDX(i, j, kb + 5)]) /
                    smr2;

                Du[IDX(i, j, kb + 3)] =
                    (u[IDX(i, j, jb)] - u[IDX(i, j, kb + 1)] -
                     6.0 * u[IDX(i, j, kb + 2)] + 15.0 * u[IDX(i, j, kb + 3)] -
                     14.0 * u[IDX(i, j, kb + 4)] + 6.0 * u[IDX(i, j, kb + 5)] -
                     u[IDX(i, j, kb + 6)]) /
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
                Du[IDX(i, j, ke - 4)] =
                    (-u[IDX(i, j, ke - 7)] + 6.0 * u[IDX(i, j, ke - 6)] -
                     14.0 * u[IDX(i, j, ke - 5)] + 15.0 * u[IDX(i, j, ke - 4)] -
                     6.0 * u[IDX(i, j, ke - 3)] - u[IDX(i, j, ke - 2)] +
                     u[IDX(i, j, ke - 1)]) /
                    spr1;

                Du[IDX(i, j, ke - 3)] =
                    (-u[IDX(i, j, ke - 6)] + 3.0 * u[IDX(i, j, ke - 5)] -
                     8.0 * u[IDX(i, j, ke - 3)] + 9.0 * u[IDX(i, j, ke - 2)] -
                     3.0 * u[IDX(i, j, ke - 1)]) /
                    spr2;

                Du[IDX(i, j, ke - 2)] =
                    (2.0 * u[IDX(i, j, ke - 5)] - 9.0 * u[IDX(i, j, ke - 4)] +
                     15.0 * u[IDX(i, j, ke - 3)] - 11.0 * u[IDX(i, j, ke - 2)] +
                     3.0 * u[IDX(i, j, ke - 1)]) /
                    spr3;

                Du[IDX(i, j, ke - 1)] =
                    (-u[IDX(i, j, ke - 5)] + 4.0 * u[IDX(i, j, ke - 4)] -
                     6.0 * u[IDX(i, j, ke - 3)] + 4.0 * u[IDX(i, j, ke - 2)] -
                     u[IDX(i, j, ke - 1)]) /
                    spr4;
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

template void ko_deriv64_x<4>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv64_x<5>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);

template void ko_deriv64_y<4>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv64_y<5>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);

template void ko_deriv64_z<4>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void ko_deriv64_z<5>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);

// KO8 interior (radius-5) + KO6 single layer (radius-3) + KO4 (inner) + KO2 (outer)
// Formatting mirrors the KO6 derivatives: ib/jb/kb from P, explicit band loops, bflag gating.

template <unsigned int P>
void ko_deriv8_x(double *const Du, const double *const u, const double dx,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be between 4 and 5 (for now)!");

    const double pref8 = -1.0 / (1024.0 * dx); // KO8 radius-5 interior
    const double pref6 = -1.0 / (   64.0 * dx); // KO6 one layer
    const double pref4 = -1.0 / (   16.0 * dx); // KO4 inner layer
    const double inv_s2 = 1.0 / (4.0 * dx);     // KO2 outer layer (3-pt)

    const int nx = sz[0];
    const int ny = sz[1];
    const int nz = sz[2];

    static constexpr int ib = (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 4) ? 4 : 5;
    const int ie = nx - ib;
    const int je = ny - jb;
    const int ke = nz - kb;

    // -----------------------
    // Interior KO8 (radius 5)
    // -----------------------
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = ib; i < ie; i++) {
                const int p = IDX(i, j, k);
                Du[p] = pref8 *
                        (  u[IDX(i-5,j,k)] - 10.0*u[IDX(i-4,j,k)]
                         +45.0*u[IDX(i-3,j,k)] -120.0*u[IDX(i-2,j,k)]
                        +210.0*u[IDX(i-1,j,k)] -252.0*u[p]
                        +210.0*u[IDX(i+1,j,k)] -120.0*u[IDX(i+2,j,k)]
                         +45.0*u[IDX(i+3,j,k)] - 10.0*u[IDX(i+4,j,k)]
                          +1.0*u[IDX(i+5,j,k)]);
            }
        }
    }

    // -----------------------------
    // Boundary bands — LEFT / RIGHT
    // -----------------------------
    if (bflag & (1u << OCT_DIR_LEFT)) {
        const int i6  = (P == 5 ? 4 : 3);                // KO6 layer (middle)
        const int i4  = (P == 5 ? 3 : 2);                // KO4 inner layer
        const int i2  = (P == 5 ? 2 : 1);                // KO2 outer layer

        // KO6 single layer (radius 3) at i6
        for (int k = 3; k < nz - 3; k++) {
            for (int j = 3; j < ny - 3; j++) {
                const int p = IDX(i6, j, k);
                Du[p] = pref6 *
                        (  u[IDX(i6-3,j,k)] - 6.0*u[IDX(i6-2,j,k)]
                         +15.0*u[IDX(i6-1,j,k)] -20.0*u[p]
                         +15.0*u[IDX(i6+1,j,k)] - 6.0*u[IDX(i6+2,j,k)]
                          +1.0*u[IDX(i6+3,j,k)]);
            }
        }

        // KO4 inner layer (radius 2) at i4
        for (int k = 2; k < nz - 2; k++) {
            for (int j = 2; j < ny - 2; j++) {
                const int i = i4;
                const int p = IDX(i, j, k);
                Du[p] = pref4 *
                        (  u[IDX(i-2,j,k)] - 4.0*u[IDX(i-1,j,k)]
                         + 6.0*u[p]
                         - 4.0*u[IDX(i+1,j,k)] + 1.0*u[IDX(i+2,j,k)]);
            }
        }

        // KO2 outer layer (3-pt forward) at i2
        for (int k = 1; k < nz - 1; k++) {
            for (int j = 1; j < ny - 1; j++) {
                const int i = i2;
                Du[IDX(i, j, k)] =
                    ( u[IDX(i, j, k)] - 2.0*u[IDX(i+1, j, k)] + u[IDX(i+2, j, k)] ) * inv_s2;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        const int i6  = (P == 5 ? nx - 5 : nx - 4);      // KO6 layer (middle)
        const int i4  = (P == 5 ? nx - 4 : nx - 3);      // KO4 inner layer
        const int i2  = (P == 5 ? nx - 3 : nx - 2);      // KO2 outer layer

        // KO6 single layer (radius 3) at i6
        for (int k = 3; k < nz - 3; k++) {
            for (int j = 3; j < ny - 3; j++) {
                const int p = IDX(i6, j, k);
                Du[p] = pref6 *
                        (  u[IDX(i6-3,j,k)] - 6.0*u[IDX(i6-2,j,k)]
                         +15.0*u[IDX(i6-1,j,k)] -20.0*u[p]
                         +15.0*u[IDX(i6+1,j,k)] - 6.0*u[IDX(i6+2,j,k)]
                          +1.0*u[IDX(i6+3,j,k)]);
            }
        }

        // KO4 inner layer (radius 2) at i4
        for (int k = 2; k < nz - 2; k++) {
            for (int j = 2; j < ny - 2; j++) {
                const int i = i4;
                const int p = IDX(i, j, k);
                Du[p] = pref4 *
                        (  u[IDX(i-2,j,k)] - 4.0*u[IDX(i-1,j,k)]
                         + 6.0*u[p]
                         - 4.0*u[IDX(i+1,j,k)] + 1.0*u[IDX(i+2,j,k)]);
            }
        }

        // KO2 outer layer (3-pt backward) at i2
        for (int k = 1; k < nz - 1; k++) {
            for (int j = 1; j < ny - 1; j++) {
                const int i = i2;
                Du[IDX(i, j, k)] =
                    ( u[IDX(i-2, j, k)] - 2.0*u[IDX(i-1, j, k)] + u[IDX(i, j, k)] ) * inv_s2;
            }
        }
    }
}

template <unsigned int P>
void ko_deriv8_y(double *const Du, const double *const u, const double dy,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be 4 or 5!");

    const double pref8 = -1.0 / (1024.0 * dy);
    const double pref6 = -1.0 / (   64.0 * dy);
    const double pref4 = -1.0 / (   16.0 * dy);
    const double inv_s2 = 1.0 / (4.0 * dy);

    const int nx = sz[0], ny = sz[1], nz = sz[2];

    static constexpr int ib = (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 4) ? 4 : 5;
    const int ie = nx - ib, je = ny - jb, ke = nz - kb;

    // Interior KO8
    for (int k = kb; k < ke; k++) {
        for (int i = ib; i < ie; i++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int j = jb; j < je; j++) {
                const int p = IDX(i, j, k);
                Du[p] = pref8 *
                        (  u[IDX(i,j-5,k)] - 10.0*u[IDX(i,j-4,k)]
                         +45.0*u[IDX(i,j-3,k)] -120.0*u[IDX(i,j-2,k)]
                        +210.0*u[IDX(i,j-1,k)] -252.0*u[p]
                        +210.0*u[IDX(i,j+1,k)] -120.0*u[IDX(i,j+2,k)]
                         +45.0*u[IDX(i,j+3,k)] - 10.0*u[IDX(i,j+4,k)]
                          +1.0*u[IDX(i,j+5,k)]);
            }
        }
    }

    // DOWN band
    if (bflag & (1u << OCT_DIR_DOWN)) {
        const int j6 = (P == 5 ? 4 : 3);
        const int j4 = (P == 5 ? 3 : 2);
        const int j2 = (P == 5 ? 2 : 1);

        // KO6 at j6
        for (int k = 3; k < nz - 3; k++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = 3; i < nx - 3; i++) {
                const int p = IDX(i, j6, k);
                Du[p] = pref6 *
                        (  u[IDX(i,j6-3,k)] - 6.0*u[IDX(i,j6-2,k)]
                         +15.0*u[IDX(i,j6-1,k)] -20.0*u[p]
                         +15.0*u[IDX(i,j6+1,k)] - 6.0*u[IDX(i,j6+2,k)]
                          +1.0*u[IDX(i,j6+3,k)]);
            }
        }

        // KO4 inner at j4
        for (int k = 2; k < nz - 2; k++) {
            for (int i = 2; i < nx - 2; i++) {
                const int j = j4;
                const int p = IDX(i, j, k);
                Du[p] = pref4 *
                        (  u[IDX(i,j-2,k)] - 4.0*u[IDX(i,j-1,k)]
                         + 6.0*u[p]
                         - 4.0*u[IDX(i,j+1,k)] + 1.0*u[IDX(i,j+2,k)]);
            }
        }

        // KO2 outer (forward) at j2
        for (int k = 1; k < nz - 1; k++) {
            for (int i = 1; i < nx - 1; i++) {
                const int j = j2;
                Du[IDX(i, j, k)] =
                    ( u[IDX(i, j, k)] - 2.0*u[IDX(i, j+1, k)] + u[IDX(i, j+2, k)] ) * inv_s2;
            }
        }
    }

    // UP band
    if (bflag & (1u << OCT_DIR_UP)) {
        const int j6 = (P == 5 ? ny - 5 : ny - 4);
        const int j4 = (P == 5 ? ny - 4 : ny - 3);
        const int j2 = (P == 5 ? ny - 3 : ny - 2);

        // KO6 at j6
        for (int k = 3; k < nz - 3; k++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = 3; i < nx - 3; i++) {
                const int p = IDX(i, j6, k);
                Du[p] = pref6 *
                        (  u[IDX(i,j6-3,k)] - 6.0*u[IDX(i,j6-2,k)]
                         +15.0*u[IDX(i,j6-1,k)] -20.0*u[p]
                         +15.0*u[IDX(i,j6+1,k)] - 6.0*u[IDX(i,j6+2,k)]
                          +1.0*u[IDX(i,j6+3,k)]);
            }
        }

        // KO4 inner at j4
        for (int k = 2; k < nz - 2; k++) {
            for (int i = 2; i < nx - 2; i++) {
                const int j = j4;
                const int p = IDX(i, j, k);
                Du[p] = pref4 *
                        (  u[IDX(i,j-2,k)] - 4.0*u[IDX(i,j-1,k)]
                         + 6.0*u[p]
                         - 4.0*u[IDX(i,j+1,k)] + 1.0*u[IDX(i,j+2,k)]);
            }
        }

        // KO2 outer (backward) at j2
        for (int k = 1; k < nz - 1; k++) {
            for (int i = 1; i < nx - 1; i++) {
                const int j = j2;
                Du[IDX(i, j, k)] =
                    ( u[IDX(i, j-2, k)] - 2.0*u[IDX(i, j-1, k)] + u[IDX(i, j, k)] ) * inv_s2;
            }
        }
    }
}


template <unsigned int P>
void ko_deriv8_z(double *const Du, const double *const u, const double dz,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be 4 or 5!");

    const double pref8 = -1.0 / (1024.0 * dz);
    const double pref6 = -1.0 / (   64.0 * dz);
    const double pref4 = -1.0 / (   16.0 * dz);
    const double inv_s2 = 1.0 / (4.0 * dz);

    const int nx  = sz[0];
    const int ny  = sz[1];
    const int nz  = sz[2];
    const int nxy = nx * ny;

    static constexpr int ib = (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 4) ? 4 : 5;
    const int ie = nx - ib;
    const int je = ny - jb;
    const int ke = nz - kb;

    // Interior KO8
    for (int j = jb; j < je; j++) {
        for (int i = ib; i < ie; i++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int k = kb; k < ke; k++) {
                const int p = IDX(i, j, k);
                Du[p] = pref8 *
                        (  u[p - 5*nxy] - 10.0*u[p - 4*nxy]
                         +45.0*u[p - 3*nxy] -120.0*u[p - 2*nxy]
                        +210.0*u[p - 1*nxy] -252.0*u[p]
                        +210.0*u[p + 1*nxy] -120.0*u[p + 2*nxy]
                         +45.0*u[p + 3*nxy] - 10.0*u[p + 4*nxy]
                          +1.0*u[p + 5*nxy]);
            }
        }
    }

    // BACK band
    if (bflag & (1u << OCT_DIR_BACK)) {
        const int k6  = (P == 5 ? 4 : 3);
        const int k4  = (P == 5 ? 3 : 2);
        const int k2  = (P == 5 ? 2 : 1);

        // KO6 at k6
        for (int j = 3; j < ny - 3; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = 3; i < nx - 3; i++) {
                const int p = IDX(i, j, k6);
                Du[p] = pref6 *
                        (  u[p - 3*nxy] - 6.0*u[p - 2*nxy]
                         +15.0*u[p - 1*nxy] -20.0*u[p]
                         +15.0*u[p + 1*nxy] - 6.0*u[p + 2*nxy]
                          +1.0*u[p + 3*nxy]);
            }
        }

        // KO4 inner at k4
        for (int j = 2; j < ny - 2; j++) {
            for (int i = 2; i < nx - 2; i++) {
                const int k = k4;
                const int p = IDX(i, j, k);
                Du[p] = pref4 *
                        (  u[p - 2*nxy] - 4.0*u[p - 1*nxy]
                         + 6.0*u[p]
                         - 4.0*u[p + 1*nxy] + 1.0*u[p + 2*nxy]);
            }
        }

        // KO2 outer (forward) at k2
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                const int k = k2;
                const int p = IDX(i, j, k);
                Du[p] = ( u[p] - 2.0*u[p + 1*nxy] + u[p + 2*nxy] ) * inv_s2;
            }
        }
    }

    // FRONT band
    if (bflag & (1u << OCT_DIR_FRONT)) {
        const int k6  = (P == 5 ? nz - 5 : nz - 4);
        const int k4  = (P == 5 ? nz - 4 : nz - 3);
        const int k2  = (P == 5 ? nz - 3 : nz - 2);

        // KO6 at k6
        for (int j = 3; j < ny - 3; j++) {
#ifdef DERIV_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__DERIV_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (int i = 3; i < nx - 3; i++) {
                const int p = IDX(i, j, k6);
                Du[p] = pref6 *
                        (  u[p - 3*nxy] - 6.0*u[p - 2*nxy]
                         +15.0*u[p - 1*nxy] -20.0*u[p]
                         +15.0*u[p + 1*nxy] - 6.0*u[p + 2*nxy]
                          +1.0*u[p + 3*nxy]);
            }
        }

        // KO4 inner at k4
        for (int j = 2; j < ny - 2; j++) {
            for (int i = 2; i < nx - 2; i++) {
                const int k = k4;
                const int p = IDX(i, j, k);
                Du[p] = pref4 *
                        (  u[p - 2*nxy] - 4.0*u[p - 1*nxy]
                         + 6.0*u[p]
                         - 4.0*u[p + 1*nxy] + 1.0*u[p + 2*nxy]);
            }
        }

        // KO2 outer (backward) at k2
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                const int k = k2;
                const int p = IDX(i, j, k);
                Du[p] = ( u[p - 2*nxy] - 2.0*u[p - 1*nxy] + u[p] ) * inv_s2;
            }
        }
    }
}



// Only instantiate for P=5
template void ko_deriv8_x<5>(double *const, const double *const,
                             const double, const unsigned int *, unsigned);
template void ko_deriv8_y<5>(double *const, const double *const,
                             const double, const unsigned int *, unsigned);
template void ko_deriv8_z<5>(double *const, const double *const,
                             const double, const unsigned int *, unsigned);


}  // namespace dendroderivs
