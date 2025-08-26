#include "derivatives/derivs_explicit.h"

namespace dendroderivs {

template <unsigned int P>
void deriv42_x(double *const Dxu, const double *const u, const double dx,
               const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 2 and 5 (for now)!");
    const double idx        = 1.0 / dx;
    const double idx_by_2   = 0.5 * idx;
    const double idx_by_12  = idx / 12.0;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    // based on the incoming P, which is our padding width, we will want to
    // create a different starting and ending point

    static constexpr int ib = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 2) ? 0 : (P == 3) ? 1 : (P == 4) ? 2 : 3;
    static constexpr int kb = (P == 2) ? 0 : (P == 3) ? 1 : (P == 4) ? 2 : 3;
    const int ie            = nx - ib;
    const int je            = ny - jb;
    const int ke            = nz - kb;

    const int n             = 1;

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
                Dxu[pp]      = (u[pp - 2] - 8.0 * u[pp - 1] + 8.0 * u[pp + 1] -
                           u[pp + 2]) *
                          idx_by_12;
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
                Dxu[IDX(ib, j, k)] =
                    (-3.0 * u[IDX(ib, j, k)] + 4.0 * u[IDX(ib + 1, j, k)] -
                     u[IDX(ib + 2, j, k)]) *
                    idx_by_2;
                Dxu[IDX(ib + 1, j, k)] =
                    (-u[IDX(ib, j, k)] + u[IDX(ib + 2, j, k)]) * idx_by_2;
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
                Dxu[IDX(ie - 2, j, k)] =
                    (-u[IDX(ie - 3, j, k)] + u[IDX(ie - 1, j, k)]) * idx_by_2;

                Dxu[IDX(ie - 1, j, k)] =
                    (u[IDX(ie - 3, j, k)] - 4.0 * u[IDX(ie - 2, j, k)] +
                     3.0 * u[IDX(ie - 1, j, k)]) *
                    idx_by_2;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (isnan(Dxu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
    return;
};

template <unsigned int P>
void deriv42_y(double *const Dyu, const double *const u, const double dy,
               const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 2 and 5 (for now)!");
    const double idy        = 1.0 / dy;
    const double idy_by_2   = 0.50 * idy;
    const double idy_by_12  = idy / 12.0;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    static constexpr int ib = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 2) ? 2 : (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 2) ? 0 : (P == 3) ? 1 : (P == 4) ? 2 : 3;
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
                Dyu[pp]      = (u[pp - 2 * nx] - 8.0 * u[pp - nx] +
                           8.0 * u[pp + nx] - u[pp + 2 * nx]) *
                          idy_by_12;
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
                Dyu[IDX(i, jb, k)] =
                    (-3.0 * u[IDX(i, jb, k)] + 4.0 * u[IDX(i, jb + 1, k)] -
                     u[IDX(i, jb + 2, k)]) *
                    idy_by_2;

                Dyu[IDX(i, jb + 1, k)] =
                    (-u[IDX(i, jb, k)] + u[IDX(i, jb + 2, k)]) * idy_by_2;
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
                Dyu[IDX(i, je - 2, k)] =
                    (-u[IDX(i, je - 3, k)] + u[IDX(i, je - 1, k)]) * idy_by_2;

                Dyu[IDX(i, je - 1, k)] =
                    (u[IDX(i, je - 3, k)] - 4.0 * u[IDX(i, je - 2, k)] +
                     3.0 * u[IDX(i, je - 1, k)]) *
                    idy_by_2;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Dyu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv42_z(double *const Dzu, const double *const u, const double dz,
               const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 2 and 5 (for now)!");
    const double idz        = 1.0 / dz;
    const double idz_by_2   = 0.50 * idz;
    const double idz_by_12  = idz / 12.0;

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
                Dzu[pp] = (u[pp - 2 * n] - 8.0 * u[pp - n] + 8.0 * u[pp + n] -
                           u[pp + 2 * n]) *
                          idz_by_12;
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
                Dzu[IDX(i, j, kb)] =
                    (-3.0 * u[IDX(i, j, kb)] + 4.0 * u[IDX(i, j, kb + 1)] -
                     u[IDX(i, j, kb + 2)]) *
                    idz_by_2;

                Dzu[IDX(i, j, kb + 1)] =
                    (-u[IDX(i, j, kb)] + u[IDX(i, j, kb + 2)]) * idz_by_2;
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
                Dzu[IDX(i, j, ke - 2)] =
                    (-u[IDX(i, j, ke - 3)] + u[IDX(i, j, ke - 1)]) * idz_by_2;

                Dzu[IDX(i, j, ke - 1)] =
                    (u[IDX(i, j, ke - 3)] - 4.0 * u[IDX(i, j, ke - 2)] +
                     3.0 * u[IDX(i, j, ke - 1)]) *
                    idz_by_2;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Dzu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

// Explicit instiation for P = 2, 3, 4 and 5 for deriv42_[dim]
template void deriv42_x<2>(double *const Dxu, const double *const u,
                           const double dx, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_x<3>(double *const Dxu, const double *const u,
                           const double dx, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_x<4>(double *const Dxu, const double *const u,
                           const double dx, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_x<5>(double *const Dxu, const double *const u,
                           const double dx, const unsigned int *sz,
                           unsigned bflag);

template void deriv42_y<2>(double *const Dyu, const double *const u,
                           const double dy, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_y<3>(double *const Dyu, const double *const u,
                           const double dy, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_y<4>(double *const Dyu, const double *const u,
                           const double dy, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_y<5>(double *const Dyu, const double *const u,
                           const double dy, const unsigned int *sz,
                           unsigned bflag);

template void deriv42_z<2>(double *const Dzu, const double *const u,
                           const double dz, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_z<3>(double *const Dzu, const double *const u,
                           const double dz, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_z<4>(double *const Dzu, const double *const u,
                           const double dz, const unsigned int *sz,
                           unsigned bflag);
template void deriv42_z<5>(double *const Dzu, const double *const u,
                           const double dz, const unsigned int *sz,
                           unsigned bflag);

template <unsigned int P>
void deriv42_xx(double *const DxDxu, const double *const u, const double dx,
                const unsigned int *sz, unsigned bflag) {
    const double idx_sqrd       = 1.0 / (dx * dx);
    const double idx_sqrd_by_12 = idx_sqrd / 12.0;

    const int nx                = sz[0];
    const int ny                = sz[1];
    const int nz                = sz[2];

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
                DxDxu[pp]    = (-u[pp - 2] + 16.0 * u[pp - 1] - 30.0 * u[pp] +
                             16.0 * u[pp + 1] - u[pp + 2]) *
                            idx_sqrd_by_12;
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
                DxDxu[IDX(ib, j, k)] =
                    (2.0 * u[IDX(ib, j, k)] - 5.0 * u[IDX(ib + 1, j, k)] +
                     4.0 * u[IDX(ib + 2, j, k)] - u[IDX(ib + 3, j, k)]) *
                    idx_sqrd;

                DxDxu[IDX(ib + 1, j, k)] =
                    (u[IDX(ib, j, k)] - 2.0 * u[IDX(ib + 1, j, k)] +
                     u[IDX(ib + 2, j, k)]) *
                    idx_sqrd;
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
                DxDxu[IDX(ie - 2, j, k)] =
                    (u[IDX(ie - 3, j, k)] - 2.0 * u[IDX(ie - 2, j, k)] +
                     u[IDX(ie - 1, j, k)]) *
                    idx_sqrd;

                DxDxu[IDX(ie - 1, j, k)] =
                    (-u[IDX(ie - 4, j, k)] + 4.0 * u[IDX(ie - 3, j, k)] -
                     5.0 * u[IDX(ie - 2, j, k)] + 2.0 * u[IDX(ie - 1, j, k)]) *
                    idx_sqrd;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DxDxu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv42_yy(double *const DyDyu, const double *const u, const double dy,
                const unsigned int *sz, unsigned bflag) {
    const double idy_sqrd       = 1.0 / (dy * dy);
    const double idy_sqrd_by_12 = idy_sqrd / 12.0;

    const int nx                = sz[0];
    const int ny                = sz[1];
    const int nz                = sz[2];

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
                DyDyu[pp] =
                    (-u[pp - 2 * nx] + 16.0 * u[pp - nx] - 30.0 * u[pp] +
                     16.0 * u[pp + nx] - u[pp + 2 * nx]) *
                    idy_sqrd_by_12;
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
                DyDyu[IDX(i, jb, k)] =
                    (2.0 * u[IDX(i, jb, k)] - 5.0 * u[IDX(i, jb + 1, k)] +
                     4.0 * u[IDX(i, jb + 2, k)] - u[IDX(i, jb + 3, k)]) *
                    idy_sqrd;

                DyDyu[IDX(i, jb + 1, k)] =
                    (u[IDX(i, jb, k)] - 2.0 * u[IDX(i, jb + 1, k)] +
                     u[IDX(i, jb + 2, k)]) *
                    idy_sqrd;
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
                DyDyu[IDX(i, je - 2, k)] =
                    (u[IDX(i, je - 3, k)] - 2.0 * u[IDX(i, je - 2, k)] +
                     u[IDX(i, je - 1, k)]) *
                    idy_sqrd;

                DyDyu[IDX(i, je - 1, k)] =
                    (-u[IDX(i, je - 4, k)] + 4.0 * u[IDX(i, je - 3, k)] -
                     5.0 * u[IDX(i, je - 2, k)] + 2.0 * u[IDX(i, je - 1, k)]) *
                    idy_sqrd;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DyDyu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv42_zz(double *const DzDzu, const double *const u, const double dz,
                const unsigned int *sz, unsigned bflag) {
    const double idz_sqrd       = 1.0 / (dz * dz);
    const double idz_sqrd_by_12 = idz_sqrd / 12.0;

    const int nx                = sz[0];
    const int ny                = sz[1];
    const int nz                = sz[2];

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
                DzDzu[pp] = (-u[pp - 2 * n] + 16.0 * u[pp - n] - 30.0 * u[pp] +
                             16.0 * u[pp + n] - u[pp + 2 * n]) *
                            idz_sqrd_by_12;
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
                DzDzu[IDX(i, j, kb)] =
                    (2.0 * u[IDX(i, j, kb)] - 5.0 * u[IDX(i, j, kb + 1)] +
                     4.0 * u[IDX(i, j, kb + 2)] - u[IDX(i, j, kb + 3)]) *
                    idz_sqrd;

                DzDzu[IDX(i, j, kb + 1)] =
                    (u[IDX(i, j, kb)] - 2.0 * u[IDX(i, j, kb + 1)] +
                     u[IDX(i, j, kb + 2)]) *
                    idz_sqrd;
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
                DzDzu[IDX(i, j, ke - 2)] =
                    (u[IDX(i, j, ke - 3)] - 2.0 * u[IDX(i, j, ke - 2)] +
                     u[IDX(i, j, ke - 1)]) *
                    idz_sqrd;

                DzDzu[IDX(i, j, ke - 1)] =
                    (-u[IDX(i, j, ke - 4)] + 4.0 * u[IDX(i, j, ke - 3)] -
                     5.0 * u[IDX(i, j, ke - 2)] + 2.0 * u[IDX(i, j, ke - 1)]) *
                    idz_sqrd;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DzDzu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

// Explicit instiation for P = 2, 3, 4 and 5 for deriv42_[dim]
template void deriv42_xx<2>(double *const DxDxu, const double *const u,
                            const double dx, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_xx<3>(double *const DxDxu, const double *const u,
                            const double dx, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_xx<4>(double *const DxDxu, const double *const u,
                            const double dx, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_xx<5>(double *const DxDxu, const double *const u,
                            const double dx, const unsigned int *sz,
                            unsigned bflag);

template void deriv42_yy<2>(double *const DyDyu, const double *const u,
                            const double dy, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_yy<3>(double *const DyDyu, const double *const u,
                            const double dy, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_yy<4>(double *const DyDyu, const double *const u,
                            const double dy, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_yy<5>(double *const DyDyu, const double *const u,
                            const double dy, const unsigned int *sz,
                            unsigned bflag);

template void deriv42_zz<2>(double *const DzDzu, const double *const u,
                            const double dz, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_zz<3>(double *const DzDzu, const double *const u,
                            const double dz, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_zz<4>(double *const DzDzu, const double *const u,
                            const double dz, const unsigned int *sz,
                            unsigned bflag);
template void deriv42_zz<5>(double *const DzDzu, const double *const u,
                            const double dz, const unsigned int *sz,
                            unsigned bflag);

template <unsigned int P>
void deriv644_x(double *const Dxu, const double *const u, const double dx,
                const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 3 && P <= 5, "P must be between 3 and 5 (for now)!");
    const double idx        = 1.0 / dx;
    const double idx_by_12  = idx / 12.0;
    const double idx_by_60  = idx / 60.0;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    static constexpr int ib = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 3) ? 0 : (P == 4) ? 1 : 2;
    static constexpr int kb = (P == 3) ? 0 : (P == 4) ? 1 : 2;
    const int ie            = nx - ib;
    const int je            = ny - jb;
    const int ke            = nz - kb;

    const int n             = 1;

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
                Dxu[pp] = (-u[pp - 3] + 9.0 * u[pp - 2] - 45.0 * u[pp - 1] +
                           45.0 * u[pp + 1] - 9.0 * u[pp + 2] + u[pp + 3]) *
                          idx_by_60;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (int k = kb; k < ke; k++) {
            for (int j = jb; j < je; j++) {
                Dxu[IDX(ib, j, k)] =
                    (-25.0 * u[IDX(ib, j, k)] + 48.0 * u[IDX(ib + 1, j, k)] -
                     36.0 * u[IDX(ib + 2, j, k)] + 16.0 * u[IDX(ib + 3, j, k)] -
                     3.0 * u[IDX(ib + 4, j, k)]) *
                    idx_by_12;

                Dxu[IDX(ib + 1, j, k)] =
                    (-3.0 * u[IDX(ib, j, k)] - 10.0 * u[IDX(ib + 1, j, k)] +
                     18.0 * u[IDX(ib + 2, j, k)] - 6.0 * u[IDX(ib + 3, j, k)] +
                     u[IDX(ib + 4, j, k)]) *
                    idx_by_12;

                Dxu[IDX(ib + 2, j, k)] =
                    (+u[IDX(ib, j, k)] - 8.0 * u[IDX(ib + 1, j, k)] +
                     8.0 * u[IDX(ib + 3, j, k)] - u[IDX(ib + 4, j, k)]) *
                    idx_by_12;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (int k = kb; k < ke; k++) {
            for (int j = jb; j < je; j++) {
                Dxu[IDX(ie - 3, j, k)] =
                    (+u[IDX(ie - 5, j, k)] - 8.0 * u[IDX(ie - 4, j, k)] +
                     8.0 * u[IDX(ie - 2, j, k)] - u[IDX(ie - 1, j, k)]) *
                    idx_by_12;

                Dxu[IDX(ie - 2, j, k)] =
                    (-u[IDX(ie - 5, j, k)] + 6.0 * u[IDX(ie - 4, j, k)] -
                     18.0 * u[IDX(ie - 3, j, k)] + 10.0 * u[IDX(ie - 2, j, k)] +
                     3.0 * u[IDX(ie - 1, j, k)]) *
                    idx_by_12;

                Dxu[IDX(ie - 1, j, k)] =
                    (3.0 * u[IDX(ie - 5, j, k)] - 16.0 * u[IDX(ie - 4, j, k)] +
                     36.0 * u[IDX(ie - 3, j, k)] - 48.0 * u[IDX(ie - 2, j, k)] +
                     25.0 * u[IDX(ie - 1, j, k)]) *
                    idx_by_12;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (isnan(Dxu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv644_y(double *const Dyu, const double *const u, const double dy,
                const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 3 && P <= 5, "P must be between 3 and 5 (for now)!");
    const double idy        = 1.0 / dy;
    const double idy_by_12  = idy / 12.0;
    const double idy_by_60  = idy / 60.0;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

    static constexpr int ib = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb = (P == 3) ? 0 : (P == 4) ? 1 : 2;
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
                Dyu[pp]      = (-u[pp - 3 * nx] + 9.0 * u[pp - 2 * nx] -
                           45.0 * u[pp - nx] + 45.0 * u[pp + nx] -
                           9.0 * u[pp + 2 * nx] + u[pp + 3 * nx]) *
                          idy_by_60;
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
                Dyu[IDX(i, jb, k)] =
                    (-25.0 * u[IDX(i, jb, k)] + 48.0 * u[IDX(i, jb + 1, k)] -
                     36.0 * u[IDX(i, jb + 2, k)] + 16.0 * u[IDX(i, jb + 3, k)] -
                     3.0 * u[IDX(i, jb + 4, k)]) *
                    idy_by_12;

                Dyu[IDX(i, jb + 1, k)] =
                    (-3.0 * u[IDX(i, jb, k)] - 10.0 * u[IDX(i, jb + 1, k)] +
                     18.0 * u[IDX(i, jb + 2, k)] - 6.0 * u[IDX(i, jb + 3, k)] +
                     u[IDX(i, jb + 4, k)]) *
                    idy_by_12;

                Dyu[IDX(i, jb + 2, k)] =
                    (u[IDX(i, jb, k)] - 8.0 * u[IDX(i, jb + 1, k)] +
                     8.0 * u[IDX(i, jb + 3, k)] - u[IDX(i, jb + 4, k)]) *
                    idy_by_12;
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
                Dyu[IDX(i, je - 3, k)] =
                    (u[IDX(i, je - 5, k)] - 8.0 * u[IDX(i, je - 4, k)] +
                     8.0 * u[IDX(i, je - 2, k)] - u[IDX(i, je - 1, k)]) *
                    idy_by_12;

                Dyu[IDX(i, je - 2, k)] =
                    (-u[IDX(i, je - 5, k)] + 6.0 * u[IDX(i, je - 4, k)] -
                     18.0 * u[IDX(i, je - 3, k)] + 10.0 * u[IDX(i, je - 2, k)] +
                     3.0 * u[IDX(i, je - 1, k)]) *
                    idy_by_12;

                Dyu[IDX(i, je - 1, k)] =
                    (+3.0 * u[IDX(i, je - 5, k)] - 16.0 * u[IDX(i, je - 4, k)] +
                     36.0 * u[IDX(i, je - 3, k)] - 48.0 * u[IDX(i, je - 2, k)] +
                     25.0 * u[IDX(i, je - 1, k)]) *
                    idy_by_12;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Dyu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv644_z(double *const Dzu, const double *const u, const double dz,
                const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 3 && P <= 5, "P must be between 3 and 5 (for now)!");
    const double idz        = 1.0 / dz;
    const double idz_by_12  = idz / 12.0;
    const double idz_by_60  = idz / 60.0;

    const int nx            = sz[0];
    const int ny            = sz[1];
    const int nz            = sz[2];

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
                Dzu[pp] =
                    (-u[pp - 3 * n] + 9.0 * u[pp - 2 * n] - 45.0 * u[pp - n] +
                     45.0 * u[pp + n] - 9.0 * u[pp + 2 * n] + u[pp + 3 * n]) *
                    idz_by_60;
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
                Dzu[IDX(i, j, kb)] =
                    (-25.0 * u[IDX(i, j, kb)] + 48.0 * u[IDX(i, j, kb + 1)] -
                     36.0 * u[IDX(i, j, kb + 2)] + 16.0 * u[IDX(i, j, kb + 3)] -
                     3.0 * u[IDX(i, j, kb + 4)]) *
                    idz_by_12;

                Dzu[IDX(i, j, kb + 1)] =
                    (-3.0 * u[IDX(i, j, kb)] - 10.0 * u[IDX(i, j, kb + 1)] +
                     18.0 * u[IDX(i, j, kb + 2)] - 6.0 * u[IDX(i, j, kb + 3)] +
                     u[IDX(i, j, kb + 4)]) *
                    idz_by_12;

                Dzu[IDX(i, j, kb + 2)] =
                    (u[IDX(i, j, kb)] - 8.0 * u[IDX(i, j, kb + 1)] +
                     8.0 * u[IDX(i, j, kb + 3)] - u[IDX(i, j, kb + 4)]) *
                    idz_by_12;
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
                Dzu[IDX(i, j, ke - 3)] =
                    (u[IDX(i, j, ke - 5)] - 8.0 * u[IDX(i, j, ke - 4)] +
                     8.0 * u[IDX(i, j, ke - 2)] - u[IDX(i, j, ke - 1)]) *
                    idz_by_12;

                Dzu[IDX(i, j, ke - 2)] =
                    (-u[IDX(i, j, ke - 5)] + 6.0 * u[IDX(i, j, ke - 4)] -
                     18.0 * u[IDX(i, j, ke - 3)] + 10.0 * u[IDX(i, j, ke - 2)] +
                     3.0 * u[IDX(i, j, ke - 1)]) *
                    idz_by_12;

                Dzu[IDX(i, j, ke - 1)] =
                    (3.0 * u[IDX(i, j, ke - 5)] - 16.0 * u[IDX(i, j, ke - 4)] +
                     36.0 * u[IDX(i, j, ke - 3)] - 48.0 * u[IDX(i, j, ke - 2)] +
                     25.0 * u[IDX(i, j, ke - 1)]) *
                    idz_by_12;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Dzu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

// Explicit instiation for P = 3, 4 and 5 for deriv644_[dim]
template void deriv644_x<3>(double *const Dxu, const double *const u,
                            const double dx, const unsigned int *sz,
                            unsigned bflag);
template void deriv644_x<4>(double *const Dxu, const double *const u,
                            const double dx, const unsigned int *sz,
                            unsigned bflag);
template void deriv644_x<5>(double *const Dxu, const double *const u,
                            const double dx, const unsigned int *sz,
                            unsigned bflag);

template void deriv644_y<3>(double *const Dyu, const double *const u,
                            const double dy, const unsigned int *sz,
                            unsigned bflag);
template void deriv644_y<4>(double *const Dyu, const double *const u,
                            const double dy, const unsigned int *sz,
                            unsigned bflag);
template void deriv644_y<5>(double *const Dyu, const double *const u,
                            const double dy, const unsigned int *sz,
                            unsigned bflag);

template void deriv644_z<3>(double *const Dzu, const double *const u,
                            const double dz, const unsigned int *sz,
                            unsigned bflag);
template void deriv644_z<4>(double *const Dzu, const double *const u,
                            const double dz, const unsigned int *sz,
                            unsigned bflag);
template void deriv644_z<5>(double *const Dzu, const double *const u,
                            const double dz, const unsigned int *sz,
                            unsigned bflag);

template <unsigned int P>
void deriv644_xx(double *const DxDxu, const double *const u, const double dx,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 3 and 5 (for now)!");
    const double idx_sqrd        = 1.0 / (dx * dx);
    const double idx_sqrd_by_180 = idx_sqrd / 180.0;
    const double idx_sqrd_by_12  = idx_sqrd / 12.0;

    const int nx                 = sz[0];
    const int ny                 = sz[1];
    const int nz                 = sz[2];

    static constexpr int ib      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    const int ie                 = nx - ib;
    const int je                 = ny - jb;
    const int ke                 = nz - kb;

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

                DxDxu[pp] =
                    (2.0 * u[pp - 3] - 27.0 * u[pp - 2] + 270.0 * u[pp - 1] -
                     490.0 * u[pp] + 270.0 * u[pp + 1] - 27.0 * u[pp + 2] +
                     2.0 * u[pp + 3]) *
                    idx_sqrd_by_180;
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
                // The above two should be replaced by 4th order approximations:
                DxDxu[IDX(ib, j, k)] =
                    (45.0 * u[IDX(ib, j, k)] - 154.0 * u[IDX(ib + 1, j, k)] +
                     214.0 * u[IDX(ib + 2, j, k)] -
                     156.0 * u[IDX(ib + 3, j, k)] +
                     61.0 * u[IDX(ib + 4, j, k)] -
                     10.0 * u[IDX(ib + 5, j, k)]) *
                    idx_sqrd_by_12;

                DxDxu[IDX(ib + 1, j, k)] =
                    (10.0 * u[IDX(ib, j, k)] - 15.0 * u[IDX(ib + 1, j, k)] -
                     4.0 * u[IDX(ib + 2, j, k)] + 14.0 * u[IDX(ib + 3, j, k)] -
                     6.0 * u[IDX(ib + 4, j, k)] + u[IDX(ib + 5, j, k)]) *
                    idx_sqrd_by_12;

                DxDxu[IDX(ib + 2, j, k)] =
                    (-u[IDX(ib, j, k)] + 16.0 * u[IDX(ib + 1, j, k)] -
                     30.0 * u[IDX(ib + 2, j, k)] + 16.0 * u[IDX(ib + 3, j, k)] -
                     u[IDX(ib + 4, j, k)]) *
                    idx_sqrd_by_12;
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
                DxDxu[IDX(ie - 3, j, k)] =
                    (-u[IDX(ie - 5, j, k)] + 16.0 * u[IDX(ie - 4, j, k)] -
                     30.0 * u[IDX(ie - 3, j, k)] + 16.0 * u[IDX(ie - 2, j, k)] -
                     u[IDX(ie - 1, j, k)]) *
                    idx_sqrd_by_12;

                // The above two should be replaced by 4th order approximations:
                DxDxu[IDX(ie - 2, j, k)] =
                    (u[IDX(ie - 6, j, k)] - 6.0 * u[IDX(ie - 5, j, k)] +
                     14.0 * u[IDX(ie - 4, j, k)] - 4.0 * u[IDX(ie - 3, j, k)] -
                     15.0 * u[IDX(ie - 2, j, k)] +
                     10.0 * u[IDX(ie - 1, j, k)]) *
                    idx_sqrd_by_12;

                DxDxu[IDX(ie - 1, j, k)] = (-10.0 * u[IDX(ie - 6, j, k)] +
                                            61.0 * u[IDX(ie - 5, j, k)] -
                                            156.0 * u[IDX(ie - 4, j, k)] +
                                            214.0 * u[IDX(ie - 3, j, k)] -
                                            154.0 * u[IDX(ie - 2, j, k)] +
                                            45.0 * u[IDX(ie - 1, j, k)]) *
                                           idx_sqrd_by_12;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DxDxu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv644_yy(double *const DyDyu, const double *const u, const double dy,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 3 and 5 (for now)!");
    const double idy_sqrd        = 1.0 / (dy * dy);
    const double idy_sqrd_by_180 = idy_sqrd / 180.0;
    const double idy_sqrd_by_12  = idy_sqrd / 12.0;

    const int nx                 = sz[0];
    const int ny                 = sz[1];
    const int nz                 = sz[2];

    static constexpr int ib      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    const int ie                 = nx - ib;
    const int je                 = ny - jb;
    const int ke                 = nz - kb;

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
                DyDyu[pp] =
                    (2.0 * u[pp - 3 * nx] - 27.0 * u[pp - 2 * nx] +
                     270.0 * u[pp - nx] - 490.0 * u[pp] + 270.0 * u[pp + nx] -
                     27.0 * u[pp + 2 * nx] + 2.0 * u[pp + 3 * nx]) *
                    idy_sqrd_by_180;
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
                // The above two should be replaced by 4th order approximations:
                DyDyu[IDX(i, jb, k)] =
                    (45.0 * u[IDX(i, jb, k)] - 154.0 * u[IDX(i, jb + 1, k)] +
                     214.0 * u[IDX(i, jb + 2, k)] -
                     156.0 * u[IDX(i, jb + 3, k)] +
                     61.0 * u[IDX(i, jb + 4, k)] -
                     10.0 * u[IDX(i, jb + 5, k)]) *
                    idy_sqrd_by_12;

                DyDyu[IDX(i, jb + 1, k)] =
                    (10.0 * u[IDX(i, jb, k)] - 15.0 * u[IDX(i, jb + 1, k)] -
                     4.0 * u[IDX(i, jb + 2, k)] + 14.0 * u[IDX(i, jb + 3, k)] -
                     6.0 * u[IDX(i, jb + 4, k)] + u[IDX(i, jb + 5, k)]) *
                    idy_sqrd_by_12;

                DyDyu[IDX(i, jb + 2, k)] =
                    (-u[IDX(i, jb, k)] + 16.0 * u[IDX(i, jb + 1, k)] -
                     30.0 * u[IDX(i, jb + 2, k)] + 16.0 * u[IDX(i, jb + 3, k)] -
                     u[IDX(i, jb + 4, k)]) *
                    idy_sqrd_by_12;
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
                DyDyu[IDX(i, je - 3, k)] =
                    (-u[IDX(i, je - 5, k)] + 16.0 * u[IDX(i, je - 4, k)] -
                     30.0 * u[IDX(i, je - 3, k)] + 16.0 * u[IDX(i, je - 2, k)] -
                     u[IDX(i, je - 1, k)]) *
                    idy_sqrd_by_12;

                // The above two should be replaced by 4th order approximations:
                DyDyu[IDX(i, je - 2, k)] =
                    (u[IDX(i, je - 6, k)] - 6.0 * u[IDX(i, je - 5, k)] +
                     14.0 * u[IDX(i, je - 4, k)] - 4.0 * u[IDX(i, je - 3, k)] -
                     15.0 * u[IDX(i, je - 2, k)] +
                     10.0 * u[IDX(i, je - 1, k)]) *
                    idy_sqrd_by_12;

                DyDyu[IDX(i, je - 1, k)] = (-10.0 * u[IDX(i, je - 6, k)] +
                                            61.0 * u[IDX(i, je - 5, k)] -
                                            156.0 * u[IDX(i, je - 4, k)] +
                                            214.0 * u[IDX(i, je - 3, k)] -
                                            154.0 * u[IDX(i, je - 2, k)] +
                                            45.0 * u[IDX(i, je - 1, k)]) *
                                           idy_sqrd_by_12;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DyDyu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv644_zz(double *const DzDzu, const double *const u, const double dz,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 2 && P <= 5, "P must be between 3 and 5 (for now)!");
    const double idz_sqrd        = 1.0 / (dz * dz);
    const double idz_sqrd_by_180 = idz_sqrd / 180.0;
    const double idz_sqrd_by_12  = idz_sqrd / 12.0;

    const int nx                 = sz[0];
    const int ny                 = sz[1];
    const int nz                 = sz[2];

    static constexpr int ib      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int jb      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    static constexpr int kb      = (P == 3) ? 3 : (P == 4) ? 4 : 5;
    const int ie                 = nx - ib;
    const int je                 = ny - jb;
    const int ke                 = nz - kb;

    const int n                  = nx * ny;

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
                DzDzu[pp] =
                    (2.0 * u[pp - 3 * n] - 27.0 * u[pp - 2 * n] +
                     270.0 * u[pp - n] - 490.0 * u[pp] + 270.0 * u[pp + n] -
                     27.0 * u[pp + 2 * n] + 2.0 * u[pp + 3 * n]) *
                    idz_sqrd_by_180;
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
                // The above two should be replaced by 4th order approximations:
                DzDzu[IDX(i, j, kb)] =
                    (45.0 * u[IDX(i, j, kb)] - 154.0 * u[IDX(i, j, kb + 1)] +
                     214.0 * u[IDX(i, j, kb + 2)] -
                     156.0 * u[IDX(i, j, kb + 3)] +
                     61.0 * u[IDX(i, j, kb + 4)] -
                     10.0 * u[IDX(i, j, kb + 5)]) *
                    idz_sqrd_by_12;

                DzDzu[IDX(i, j, kb + 1)] =
                    (10.0 * u[IDX(i, j, kb)] - 15.0 * u[IDX(i, j, kb + 1)] -
                     4.0 * u[IDX(i, j, kb + 2)] + 14.0 * u[IDX(i, j, kb + 3)] -
                     6.0 * u[IDX(i, j, kb + 4)] + u[IDX(i, j, kb + 5)]) *
                    idz_sqrd_by_12;

                DzDzu[IDX(i, j, kb + 2)] =
                    (-u[IDX(i, j, kb)] + 16.0 * u[IDX(i, j, kb + 1)] -
                     30.0 * u[IDX(i, j, kb + 2)] + 16.0 * u[IDX(i, j, kb + 3)] -
                     u[IDX(i, j, kb + 4)]) *
                    idz_sqrd_by_12;
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
                DzDzu[IDX(i, j, ke - 3)] =
                    (-u[IDX(i, j, ke - 5)] + 16.0 * u[IDX(i, j, ke - 4)] -
                     30.0 * u[IDX(i, j, ke - 3)] + 16.0 * u[IDX(i, j, ke - 2)] -
                     u[IDX(i, j, ke - 1)]) *
                    idz_sqrd_by_12;
                // The above two should be replaced by 4th order approximations:
                DzDzu[IDX(i, j, ke - 2)] =
                    (u[IDX(i, j, ke - 6)] - 6.0 * u[IDX(i, j, ke - 5)] +
                     14.0 * u[IDX(i, j, ke - 4)] - 4.0 * u[IDX(i, j, ke - 3)] -
                     15.0 * u[IDX(i, j, ke - 2)] +
                     10.0 * u[IDX(i, j, ke - 1)]) *
                    idz_sqrd_by_12;

                DzDzu[IDX(i, j, ke - 1)] = (-10.0 * u[IDX(i, j, ke - 6)] +
                                            61.0 * u[IDX(i, j, ke - 5)] -
                                            156.0 * u[IDX(i, j, ke - 4)] +
                                            214.0 * u[IDX(i, j, ke - 3)] -
                                            154.0 * u[IDX(i, j, ke - 2)] +
                                            45.0 * u[IDX(i, j, ke - 1)]) *
                                           idz_sqrd_by_12;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DzDzu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

// Explicit instiation for P = 3, 4 and 5 for deriv644_[dim]
template void deriv644_xx<3>(double *const DxDxu, const double *const u,
                             const double dx, const unsigned int *sz,
                             unsigned bflag);
template void deriv644_xx<4>(double *const DxDxu, const double *const u,
                             const double dx, const unsigned int *sz,
                             unsigned bflag);
template void deriv644_xx<5>(double *const DxDxu, const double *const u,
                             const double dx, const unsigned int *sz,
                             unsigned bflag);

template void deriv644_yy<3>(double *const DyDyu, const double *const u,
                             const double dy, const unsigned int *sz,
                             unsigned bflag);
template void deriv644_yy<4>(double *const DyDyu, const double *const u,
                             const double dy, const unsigned int *sz,
                             unsigned bflag);
template void deriv644_yy<5>(double *const DyDyu, const double *const u,
                             const double dy, const unsigned int *sz,
                             unsigned bflag);

template void deriv644_zz<3>(double *const DzDzu, const double *const u,
                             const double dz, const unsigned int *sz,
                             unsigned bflag);
template void deriv644_zz<4>(double *const DzDzu, const double *const u,
                             const double dz, const unsigned int *sz,
                             unsigned bflag);
template void deriv644_zz<5>(double *const DzDzu, const double *const u,
                             const double dz, const unsigned int *sz,
                             unsigned bflag);

template <unsigned int P>
void deriv8666_x(double *const Dxu, const double *const u, const double dx,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be between 4 and 5 (for now)!");
    const double idx         = 1.0 / dx;
    const double idx_by_2    = 0.5 * idx;
    const double idx_by_12   = idx / 12.0;
    const double idx_by_60   = idx / 60.0;
    const double idx_by_2520 = idx / 2520.0;

    const int nx             = sz[0];
    const int ny             = sz[1];
    const int nz             = sz[2];

    static constexpr int ib  = (P == 4) ? 4 : 5;
    static constexpr int jb  = (P == 4) ? 0 : 1;
    static constexpr int kb  = (P == 4) ? 0 : 1;
    const int ie             = nx - ib;
    const int je             = ny - jb;
    const int ke             = nz - kb;

    const int n              = 1;

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

                Dxu[pp] =
                    (9.0 * u[pp - 4] - 96.0 * u[pp - 3] + 504.0 * u[pp - 2] -
                     2016.0 * u[pp - 1] + 2016.0 * u[pp + 1] -
                     504.0 * u[pp + 2] + 96.0 * u[pp + 3] - 9.0 * u[pp + 4]) *
                    idx_by_2520;
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
                // This is a (totally) shifted sixth order stencil.
                Dxu[IDX(ib, j, k)] =
                    (-147.0 * u[IDX(ib, j, k)] + 360.0 * u[IDX(ib + 1, j, k)] -
                     450.0 * u[IDX(ib + 2, j, k)] +
                     400.0 * u[IDX(ib + 3, j, k)] -
                     225.0 * u[IDX(ib + 4, j, k)] +
                     72.0 * u[IDX(ib + 5, j, k)] -
                     10.0 * u[IDX(ib + 6, j, k)]) *
                    idx_by_60;

                // This is a shifted sixth order stencil.
                Dxu[IDX(ib + 1, j, k)] =
                    (-10.0 * u[IDX(ib, j, k)] - 77.0 * u[IDX(ib + 1, j, k)] +
                     150.0 * u[IDX(ib + 2, j, k)] -
                     100.0 * u[IDX(ib + 3, j, k)] +
                     50.0 * u[IDX(ib + 4, j, k)] - 15.0 * u[IDX(ib + 5, j, k)] +
                     2.0 * u[IDX(ib + 6, j, k)]) *
                    idx_by_60;

                // This is a shifted sixth order stencil.
                Dxu[IDX(ib + 2, j, k)] =
                    (2.0 * u[IDX(ib, j, k)] - 24.0 * u[IDX(ib + 1, j, k)] -
                     35.0 * u[IDX(ib + 2, j, k)] + 80.0 * u[IDX(ib + 3, j, k)] -
                     30.0 * u[IDX(ib + 4, j, k)] + 8.0 * u[IDX(ib + 5, j, k)] -
                     u[IDX(ib + 6, j, k)]) *
                    idx_by_60;

                // This is a centered sixth order stencil.
                Dxu[IDX(ib + 3, j, k)] =
                    (-u[IDX(ib, j, k)] + 9.0 * u[IDX(ib + 1, j, k)] -
                     45.0 * u[IDX(ib + 2, j, k)] + 45.0 * u[IDX(ib + 4, j, k)] -
                     9.0 * u[IDX(ib + 5, j, k)] + u[IDX(ib + 6, j, k)]) *
                    idx_by_60;
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
                // This is a centered sixth order stencil.
                Dxu[IDX(ie - 4, j, k)] =
                    (-u[IDX(ie - 7, j, k)] + 9.0 * u[IDX(ie - 6, j, k)] -
                     45.0 * u[IDX(ie - 5, j, k)] + 45.0 * u[IDX(ie - 3, j, k)] -
                     9.0 * u[IDX(ie - 2, j, k)] + u[IDX(ie - 1, j, k)]) *
                    idx_by_60;

                // This is a shifted sixth order stencil.
                Dxu[IDX(ie - 3, j, k)] =
                    (u[IDX(ie - 7, j, k)] - 8.0 * u[IDX(ie - 6, j, k)] +
                     30.0 * u[IDX(ie - 5, j, k)] - 80.0 * u[IDX(ie - 4, j, k)] +
                     35.0 * u[IDX(ie - 3, j, k)] + 24.0 * u[IDX(ie - 2, j, k)] -
                     2.0 * u[IDX(ie - 1, j, k)]) *
                    idx_by_60;

                // This is a shifted sixth order stencil.
                Dxu[IDX(ie - 2, j, k)] =
                    (-2.0 * u[IDX(ie - 7, j, k)] + 15.0 * u[IDX(ie - 6, j, k)] -
                     50.0 * u[IDX(ie - 5, j, k)] +
                     100.0 * u[IDX(ie - 4, j, k)] -
                     150.0 * u[IDX(ie - 3, j, k)] +
                     77.0 * u[IDX(ie - 2, j, k)] +
                     10.0 * u[IDX(ie - 1, j, k)]) *
                    idx_by_60;

                // This is a shifted sixth order stencil.
                Dxu[IDX(ie - 1, j, k)] =
                    (10.0 * u[IDX(ie - 7, j, k)] - 72.0 * u[IDX(ie - 6, j, k)] +
                     225.0 * u[IDX(ie - 5, j, k)] -
                     400.0 * u[IDX(ie - 4, j, k)] +
                     450.0 * u[IDX(ie - 3, j, k)] -
                     360.0 * u[IDX(ie - 2, j, k)] +
                     147.0 * u[IDX(ie - 1, j, k)]) *
                    idx_by_60;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < sz[2] - 4; k++) {
        for (int j = jb; j < sz[1] - 4; j++) {
            for (int i = ib; i < sz[0] - 4; i++) {
                const int pp = IDX(i, j, k);
                if (isnan(Dxu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv8666_y(double *const Dyu, const double *const u, const double dy,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be between 4 and 5 (for now)!");
    const double idy         = 1.0 / dy;
    const double idy_by_2    = 0.5 * idy;
    const double idy_by_12   = idy / 12.0;
    const double idy_by_60   = idy / 60.0;
    const double idy_by_2520 = idy / 2520.0;

    const int nx             = sz[0];
    const int ny             = sz[1];
    const int nz             = sz[2];

    static constexpr int ib  = (P == 4) ? 4 : 5;
    static constexpr int jb  = (P == 4) ? 4 : 5;
    static constexpr int kb  = (P == 4) ? 0 : 1;
    const int ie             = nx - ib;
    const int je             = ny - jb;
    const int ke             = nz - kb;

    const int n              = nx;

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

                Dyu[pp]      = (9.0 * u[pp - 4 * n] - 96.0 * u[pp - 3 * n] +
                           504.0 * u[pp - 2 * n] - 2016.0 * u[pp - n] +
                           2016.0 * u[pp + n] - 504.0 * u[pp + 2 * n] +
                           96.0 * u[pp + 3 * n] - 9.0 * u[pp + 4 * n]) *
                          idy_by_2520;
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
                Dyu[IDX(i, jb, k)] =
                    (-147.0 * u[IDX(i, jb, k)] + 360.0 * u[IDX(i, jb + 1, k)] -
                     450.0 * u[IDX(i, jb + 2, k)] +
                     400.0 * u[IDX(i, jb + 3, k)] -
                     225.0 * u[IDX(i, jb + 4, k)] +
                     72.0 * u[IDX(i, jb + 5, k)] -
                     10.0 * u[IDX(i, jb + 6, k)]) *
                    idy_by_60;

                // This is a shifted sixth order stencil.
                Dyu[IDX(i, jb + 1, k)] =
                    (-10.0 * u[IDX(i, jb, k)] - 77.0 * u[IDX(i, jb + 1, k)] +
                     150.0 * u[IDX(i, jb + 2, k)] -
                     100.0 * u[IDX(i, jb + 3, k)] +
                     50.0 * u[IDX(i, jb + 4, k)] - 15.0 * u[IDX(i, jb + 5, k)] +
                     2.0 * u[IDX(i, jb + 6, k)]) *
                    idy_by_60;

                // This is a shifted sixth order stencil.
                Dyu[IDX(i, jb + 2, k)] =
                    (2.0 * u[IDX(i, jb, k)] - 24.0 * u[IDX(i, jb + 1, k)] -
                     35.0 * u[IDX(i, jb + 2, k)] + 80.0 * u[IDX(i, jb + 3, k)] -
                     30.0 * u[IDX(i, jb + 4, k)] + 8.0 * u[IDX(i, jb + 5, k)] -
                     u[IDX(i, jb + 6, k)]) *
                    idy_by_60;

                // This is a centered sixth order stencil.
                Dyu[IDX(i, jb + 3, k)] =
                    (-u[IDX(i, jb, k)] + 9.0 * u[IDX(i, jb + 1, k)] -
                     45.0 * u[IDX(i, jb + 2, k)] + 45.0 * u[IDX(i, jb + 4, k)] -
                     9.0 * u[IDX(i, jb + 5, k)] + u[IDX(i, jb + 6, k)]) *
                    idy_by_60;
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
                // This is a centered sixth order stencil.
                Dyu[IDX(i, je - 4, k)] =
                    (-u[IDX(i, je - 7, k)] + 9.0 * u[IDX(i, je - 6, k)] -
                     45.0 * u[IDX(i, je - 5, k)] + 45.0 * u[IDX(i, je - 3, k)] -
                     9.0 * u[IDX(i, je - 2, k)] + u[IDX(i, je - 1, k)]) *
                    idy_by_60;

                // This is a shifted sixth order stencil.
                Dyu[IDX(i, je - 3, k)] =
                    (u[IDX(i, je - 7, k)] - 8.0 * u[IDX(i, je - 6, k)] +
                     30.0 * u[IDX(i, je - 5, k)] - 80.0 * u[IDX(i, je - 4, k)] +
                     35.0 * u[IDX(i, je - 3, k)] + 24.0 * u[IDX(i, je - 2, k)] -
                     2.0 * u[IDX(i, je - 1, k)]) *
                    idy_by_60;

                // This is a shifted sixth order stencil.
                Dyu[IDX(i, je - 2, k)] =
                    (-2.0 * u[IDX(i, je - 7, k)] + 15.0 * u[IDX(i, je - 6, k)] -
                     50.0 * u[IDX(i, je - 5, k)] +
                     100.0 * u[IDX(i, je - 4, k)] -
                     150.0 * u[IDX(i, je - 3, k)] +
                     77.0 * u[IDX(i, je - 2, k)] +
                     10.0 * u[IDX(i, je - 1, k)]) *
                    idy_by_60;

                // This is a (totally) shifted sixth order stencil.
                Dyu[IDX(i, je - 1, k)] =
                    (10.0 * u[IDX(i, je - 7, k)] - 72.0 * u[IDX(i, je - 6, k)] +
                     225.0 * u[IDX(i, je - 5, k)] -
                     400.0 * u[IDX(i, je - 4, k)] +
                     450.0 * u[IDX(i, je - 3, k)] -
                     360.0 * u[IDX(i, je - 2, k)] +
                     147.0 * u[IDX(i, je - 1, k)]) *
                    idy_by_60;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < sz[2] - 4; k++) {
        for (int j = jb; j < sz[1] - 4; j++) {
            for (int i = ib; i < sz[0] - 4; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Dyu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv8666_z(double *const Dzu, const double *const u, const double dz,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be between 4 and 5 (for now)!");
    const double idz         = 1.0 / dz;
    const double idz_by_2    = 0.5 * idz;
    const double idz_by_12   = idz / 12.0;
    const double idz_by_60   = idz / 60.0;
    const double idz_by_2520 = idz / 2520.0;

    const int nx             = sz[0];
    const int ny             = sz[1];
    const int nz             = sz[2];
    static constexpr int ib  = (P == 4) ? 4 : 5;
    static constexpr int jb  = (P == 4) ? 4 : 5;
    static constexpr int kb  = (P == 4) ? 4 : 5;
    const int ie             = nx - ib;
    const int je             = ny - jb;
    const int ke             = nz - kb;

    const int n              = nx * ny;

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

                Dzu[pp]      = (9.0 * u[pp - 4 * n] - 96.0 * u[pp - 3 * n] +
                           504.0 * u[pp - 2 * n] - 2016.0 * u[pp - n] +
                           2016.0 * u[pp + n] - 504.0 * u[pp + 2 * n] +
                           96.0 * u[pp + 3 * n] - 9.0 * u[pp + 4 * n]) *
                          idz_by_2520;
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
                // This is a (totally) shifted sixth order stencil.
                Dzu[IDX(i, j, 4)] =
                    (-147.0 * u[IDX(i, j, 4)] + 360.0 * u[IDX(i, j, 5)] -
                     450.0 * u[IDX(i, j, 6)] + 400.0 * u[IDX(i, j, 7)] -
                     225.0 * u[IDX(i, j, 8)] + 72.0 * u[IDX(i, j, 9)] -
                     10.0 * u[IDX(i, j, 10)]) *
                    idz_by_60;

                // This is a shifted sixth order stencil.
                Dzu[IDX(i, j, 5)] =
                    (-10.0 * u[IDX(i, j, 4)] - 77.0 * u[IDX(i, j, 5)] +
                     150.0 * u[IDX(i, j, 6)] - 100.0 * u[IDX(i, j, 7)] +
                     50.0 * u[IDX(i, j, 8)] - 15.0 * u[IDX(i, j, 9)] +
                     2.0 * u[IDX(i, j, 10)]) *
                    idz_by_60;

                // This is a shifted sixth order stencil.
                Dzu[IDX(i, j, 6)] =
                    (2.0 * u[IDX(i, j, 4)] - 24.0 * u[IDX(i, j, 5)] -
                     35.0 * u[IDX(i, j, 6)] + 80.0 * u[IDX(i, j, 7)] -
                     30.0 * u[IDX(i, j, 8)] + 8.0 * u[IDX(i, j, 9)] -
                     u[IDX(i, j, 10)]) *
                    idz_by_60;

                // This is a centered sixth order stencil.
                Dzu[IDX(i, j, 7)] =
                    (-u[IDX(i, j, 4)] + 9.0 * u[IDX(i, j, 5)] -
                     45.0 * u[IDX(i, j, 6)] + 45.0 * u[IDX(i, j, 8)] -
                     9.0 * u[IDX(i, j, 9)] + u[IDX(i, j, 10)]) *
                    idz_by_60;

                Dzu[IDX(i, j, kb)] =
                    (-147.0 * u[IDX(i, j, kb)] + 360.0 * u[IDX(i, j, kb + 1)] -
                     450.0 * u[IDX(i, j, kb + 2)] +
                     400.0 * u[IDX(i, j, kb + 3)] -
                     225.0 * u[IDX(i, j, kb + 4)] +
                     72.0 * u[IDX(i, j, kb + 5)] -
                     10.0 * u[IDX(i, j, kb + 6)]) *
                    idz_by_60;

                // This is a shifted sixth order stencil.
                Dzu[IDX(i, j, kb + 1)] =
                    (-10.0 * u[IDX(i, j, kb)] - 77.0 * u[IDX(i, j, kb + 1)] +
                     150.0 * u[IDX(i, j, kb + 2)] -
                     100.0 * u[IDX(i, j, kb + 3)] +
                     50.0 * u[IDX(i, j, kb + 4)] - 15.0 * u[IDX(i, j, kb + 5)] +
                     2.0 * u[IDX(i, j, kb + 6)]) *
                    idz_by_60;

                // This is a shifted sixth order stencil.
                Dzu[IDX(i, j, kb + 2)] =
                    (2.0 * u[IDX(i, j, kb)] - 24.0 * u[IDX(i, j, kb + 1)] -
                     35.0 * u[IDX(i, j, kb + 2)] + 80.0 * u[IDX(i, j, kb + 3)] -
                     30.0 * u[IDX(i, j, kb + 4)] + 8.0 * u[IDX(i, j, kb + 5)] -
                     u[IDX(i, j, kb + 6)]) *
                    idz_by_60;

                // This is a centered sixth order stencil.
                Dzu[IDX(i, j, kb + 3)] =
                    (-u[IDX(i, j, kb)] + 9.0 * u[IDX(i, j, kb + 1)] -
                     45.0 * u[IDX(i, j, kb + 2)] + 45.0 * u[IDX(i, j, kb + 4)] -
                     9.0 * u[IDX(i, j, kb + 5)] + u[IDX(i, j, kb + 6)]) *
                    idz_by_60;
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
                // This is a centered sixth order stencil.
                Dzu[IDX(i, j, ke - 4)] =
                    (-u[IDX(i, j, ke - 7)] + 9.0 * u[IDX(i, j, ke - 6)] -
                     45.0 * u[IDX(i, j, ke - 5)] + 45.0 * u[IDX(i, j, ke - 3)] -
                     9.0 * u[IDX(i, j, ke - 2)] + u[IDX(i, j, ke - 1)]) *
                    idz_by_60;

                // This is a shifted sixth order stencil.
                Dzu[IDX(i, j, ke - 3)] =
                    (u[IDX(i, j, ke - 7)] - 8.0 * u[IDX(i, j, ke - 6)] +
                     30.0 * u[IDX(i, j, ke - 5)] - 80.0 * u[IDX(i, j, ke - 4)] +
                     35.0 * u[IDX(i, j, ke - 3)] + 24.0 * u[IDX(i, j, ke - 2)] -
                     2.0 * u[IDX(i, j, ke - 1)]) *
                    idz_by_60;

                // This is a (partially) shifted sixth order stencil.
                Dzu[IDX(i, j, ke - 2)] =
                    (-2.0 * u[IDX(i, j, ke - 7)] + 15.0 * u[IDX(i, j, ke - 6)] -
                     50.0 * u[IDX(i, j, ke - 5)] +
                     100.0 * u[IDX(i, j, ke - 4)] -
                     150.0 * u[IDX(i, j, ke - 3)] +
                     77.0 * u[IDX(i, j, ke - 2)] +
                     10.0 * u[IDX(i, j, ke - 1)]) *
                    idz_by_60;

                // This is a (totally) shifted sixth order stencil.
                Dzu[IDX(i, j, ke - 1)] =
                    (10.0 * u[IDX(i, j, ke - 7)] - 72.0 * u[IDX(i, j, ke - 6)] +
                     225.0 * u[IDX(i, j, ke - 5)] -
                     400.0 * u[IDX(i, j, ke - 4)] +
                     450.0 * u[IDX(i, j, ke - 3)] -
                     360.0 * u[IDX(i, j, ke - 2)] +
                     147.0 * u[IDX(i, j, ke - 1)]) *
                    idz_by_60;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < sz[2] - 4; k++) {
        for (int j = jb; j < sz[1] - 4; j++) {
            for (int i = ib; i < sz[0] - 4; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(Dzu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

// Explicit instiation for P = 4 and 5 for deriv866_[dim]
template void deriv8666_x<4>(double *const Dxu, const double *const u,
                             const double dx, const unsigned int *sz,
                             unsigned bflag);
template void deriv8666_x<5>(double *const Dxu, const double *const u,
                             const double dx, const unsigned int *sz,
                             unsigned bflag);

template void deriv8666_y<4>(double *const Dyu, const double *const u,
                             const double dy, const unsigned int *sz,
                             unsigned bflag);
template void deriv8666_y<5>(double *const Dyu, const double *const u,
                             const double dy, const unsigned int *sz,
                             unsigned bflag);

template void deriv8666_z<4>(double *const Dzu, const double *const u,
                             const double dz, const unsigned int *sz,
                             unsigned bflag);
template void deriv8666_z<5>(double *const Dzu, const double *const u,
                             const double dz, const unsigned int *sz,
                             unsigned bflag);

template <unsigned int P>
void deriv8666_xx(double *const DxDxu, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag) {
  const double idx_sqrd         = 1.0 / (dx * dx);
    const double idx_sqrd_by_12   = idx_sqrd / 12.0;
    const double idx_sqrd_by_180  = idx_sqrd / 180.0;
    const double idx_sqrd_by_5040 = idx_sqrd / 5040.0;

    const int nx                  = sz[0];
    const int ny                  = sz[1];
    const int nz                  = sz[2];
    const int ib                  = 4;
    const int jb                  = 4;
    const int kb                  = 4;
    const int ie                  = sz[0] - 4;
    const int je                  = sz[1] - 4;
    const int ke                  = sz[2] - 4;

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

                DxDxu[pp] =
                    (-9.0 * u[pp - 4] + 128.0 * u[pp - 3] - 1008.0 * u[pp - 2] +
                     8064.0 * u[pp - 1] - 14350.0 * u[pp] + 8064.0 * u[pp + 1] -
                     1008.0 * u[pp + 2] + 128.0 * u[pp + 3] - 9.0 * u[pp + 4]) *
                    idx_sqrd_by_5040;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (int k = kb; k < ke; k++) {
            for (int j = jb; j < je; j++) {
                // This is a (totally) shifted second order stencil.
                DxDxu[IDX(4, j, k)] =
                    (2.0 * u[IDX(4, j, k)] - 5.0 * u[IDX(5, j, k)] +
                     4.0 * u[IDX(6, j, k)] - u[IDX(7, j, k)]) *
                    idx_sqrd;

                // This is a centered second order stencil.
                DxDxu[IDX(5, j, k)] = (u[IDX(4, j, k)] - 2.0 * u[IDX(5, j, k)] +
                                       u[IDX(6, j, k)]) *
                                      idx_sqrd;

                // This is a centered fourth order stencil.
                DxDxu[IDX(6, j, k)] =
                    (-u[IDX(4, j, k)] + 16.0 * u[IDX(5, j, k)] -
                     30.0 * u[IDX(6, j, k)] + 16.0 * u[IDX(7, j, k)] -
                     u[IDX(8, j, k)]) *
                    idx_sqrd_by_12;

                // This is a centered sixth order stencil.
                DxDxu[IDX(7, j, k)] =
                    (2.0 * u[IDX(4, j, k)] - 27.0 * u[IDX(5, j, k)] +
                     270.0 * u[IDX(6, j, k)] - 490.0 * u[IDX(7, j, k)] +
                     270.0 * u[IDX(8, j, k)] - 27.0 * u[IDX(9, j, k)] +
                     2.0 * u[IDX(10, j, k)]) *
                    idx_sqrd_by_180;
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
                // This is a centered sixth order stencil.
                DxDxu[IDX(ie - 4, j, k)] =
                    (2.0 * u[IDX(ie - 7, j, k)] - 27.0 * u[IDX(ie - 6, j, k)] +
                     270.0 * u[IDX(ie - 5, j, k)] -
                     490.0 * u[IDX(ie - 4, j, k)] +
                     270.0 * u[IDX(ie - 3, j, k)] -
                     27.0 * u[IDX(ie - 2, j, k)] + 2.0 * u[IDX(ie - 1, j, k)]) *
                    idx_sqrd_by_180;

                // This is a centered fourth order stencil.
                DxDxu[IDX(ie - 3, j, k)] =
                    (-u[IDX(ie - 5, j, k)] + 16.0 * u[IDX(ie - 4, j, k)] -
                     30.0 * u[IDX(ie - 3, j, k)] + 16.0 * u[IDX(ie - 2, j, k)] -
                     u[IDX(ie - 1, j, k)]) *
                    idx_sqrd_by_12;

                // This is a centered second order stencil.
                DxDxu[IDX(ie - 2, j, k)] =
                    (u[IDX(ie - 3, j, k)] - 2.0 * u[IDX(ie - 2, j, k)] +
                     u[IDX(ie - 1, j, k)]) *
                    idx_sqrd;

                // This is a (totally) shifted second order stencil.
                DxDxu[IDX(ie - 1, j, k)] =
                    (-u[IDX(ie - 4, j, k)] + 4.0 * u[IDX(ie - 3, j, k)] -
                     5.0 * u[IDX(ie - 2, j, k)] + 2.0 * u[IDX(ie - 1, j, k)]) *
                    idx_sqrd;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DxDxu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv8666_yy(double *const DyDyu, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag) {
     const double idy_sqrd         = 1.0 / (dy * dy);
    const double idy_sqrd_by_12   = idy_sqrd / 12.0;
    const double idy_sqrd_by_180  = idy_sqrd / 180.0;
    const double idy_sqrd_by_5040 = idy_sqrd / 5040.0;

    const int nx                  = sz[0];
    const int ny                  = sz[1];
    const int nz                  = sz[2];
    const int ib                  = 4;
    const int jb                  = 4;
    const int kb                  = 4;
    const int ie                  = sz[0] - 4;
    const int je                  = sz[1] - 4;
    const int ke                  = sz[2] - 4;

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

                DyDyu[pp]    = (-9.0 * u[pp - 4 * nx] + 128.0 * u[pp - 3 * nx] -
                             1008.0 * u[pp - 2 * nx] + 8064.0 * u[pp - nx] -
                             14350.0 * u[pp] + 8064.0 * u[pp + nx] -
                             1008.0 * u[pp + 2 * nx] + 128.0 * u[pp + 3 * nx] -
                             9.0 * u[pp + 4 * nx]) *
                            idy_sqrd_by_5040;
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
                // This is a (totally) shifted second order stencil.
                DyDyu[IDX(i, 4, k)] =
                    (2.0 * u[IDX(i, 4, k)] - 5.0 * u[IDX(i, 5, k)] +
                     4.0 * u[IDX(i, 6, k)] - u[IDX(i, 7, k)]) *
                    idy_sqrd;

                // This is a centered second order stencil.
                DyDyu[IDX(i, 5, k)] = (u[IDX(i, 4, k)] - 2.0 * u[IDX(i, 5, k)] +
                                       u[IDX(i, 6, k)]) *
                                      idy_sqrd;

                // This is a centered fourth order stencil.
                DyDyu[IDX(i, 6, k)] =
                    (-u[IDX(i, 4, k)] + 16.0 * u[IDX(i, 5, k)] -
                     30.0 * u[IDX(i, 6, k)] + 16.0 * u[IDX(i, 7, k)] -
                     u[IDX(i, 8, k)]) *
                    idy_sqrd_by_12;

                // This is a centered sixth order stencil.
                DyDyu[IDX(i, 7, k)] =
                    (2.0 * u[IDX(i, 4, k)] - 27.0 * u[IDX(i, 5, k)] +
                     270.0 * u[IDX(i, 6, k)] - 490.0 * u[IDX(i, 7, k)] +
                     270.0 * u[IDX(i, 8, k)] - 27.0 * u[IDX(i, 9, k)] +
                     2.0 * u[IDX(i, 10, k)]) *
                    idy_sqrd_by_180;
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
                // This is a centered sixth order stencil.
                DyDyu[IDX(i, je - 4, k)] =
                    (2.0 * u[IDX(i, je - 7, k)] - 27.0 * u[IDX(i, je - 6, k)] +
                     270.0 * u[IDX(i, je - 5, k)] -
                     490.0 * u[IDX(i, je - 4, k)] +
                     270.0 * u[IDX(i, je - 3, k)] -
                     27.0 * u[IDX(i, je - 2, k)] + 2.0 * u[IDX(i, je - 1, k)]) *
                    idy_sqrd_by_180;

                // This is a centered fourth order stencil.
                DyDyu[IDX(i, je - 3, k)] =
                    (-u[IDX(i, je - 5, k)] + 16.0 * u[IDX(i, je - 4, k)] -
                     30.0 * u[IDX(i, je - 3, k)] + 16.0 * u[IDX(i, je - 2, k)] -
                     u[IDX(i, je - 1, k)]) *
                    idy_sqrd_by_12;

                // This is a centered second order stencil.
                DyDyu[IDX(i, je - 2, k)] =
                    (u[IDX(i, je - 3, k)] - 2.0 * u[IDX(i, je - 2, k)] +
                     u[IDX(i, je - 1, k)]) *
                    idy_sqrd;

                // This is a (totally) shifted second order stencil.
                DyDyu[IDX(i, je - 1, k)] =
                    (-u[IDX(i, je - 4, k)] + 4.0 * u[IDX(i, je - 3, k)] -
                     5.0 * u[IDX(i, je - 2, k)] + 2.0 * u[IDX(i, je - 1, k)]) *
                    idy_sqrd;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DyDyu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

template <unsigned int P>
void deriv8666_zz(double *const DzDzu, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag) {
   const double idz_sqrd         = 1.0 / (dz * dz);
    const double idz_sqrd_by_12   = idz_sqrd / 12.0;
    const double idz_sqrd_by_180  = idz_sqrd / 180.0;
    const double idz_sqrd_by_5040 = idz_sqrd / 5040.0;

    const int nx                  = sz[0];
    const int ny                  = sz[1];
    const int nz                  = sz[2];
    const int ib                  = 4;
    const int jb                  = 4;
    const int kb                  = 4;
    const int ie                  = sz[0] - 4;
    const int je                  = sz[1] - 4;
    const int ke                  = sz[2] - 4;

    const int n                   = nx * ny;

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

                DzDzu[pp]    = (-9.0 * u[pp - 4 * n] + 128.0 * u[pp - 3 * n] -
                             1008.0 * u[pp - 2 * n] + 8064.0 * u[pp - n] -
                             14350.0 * u[pp] + 8064.0 * u[pp + n] -
                             1008.0 * u[pp + 2 * n] + 128.0 * u[pp + 3 * n] -
                             9.0 * u[pp + 4 * n]) *
                            idz_sqrd_by_5040;
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
                // This is a (totally) shifted second order stencil.
                DzDzu[IDX(i, j, 4)] =
                    (2.0 * u[IDX(i, j, 4)] - 5.0 * u[IDX(i, j, 5)] +
                     4.0 * u[IDX(i, j, 6)] - u[IDX(i, j, 7)]) *
                    idz_sqrd;

                // This is a centered second order stencil.
                DzDzu[IDX(i, j, 5)] = (u[IDX(i, j, 4)] - 2.0 * u[IDX(i, j, 5)] +
                                       u[IDX(i, j, 6)]) *
                                      idz_sqrd;

                // This is a centered fourth order stencil.
                DzDzu[IDX(i, j, 6)] =
                    (-u[IDX(i, j, 4)] + 16.0 * u[IDX(i, j, 5)] -
                     30.0 * u[IDX(i, j, 6)] + 16.0 * u[IDX(i, j, 7)] -
                     u[IDX(i, j, 8)]) *
                    idz_sqrd_by_12;

                // This is a centered sixth order stencil.
                DzDzu[IDX(i, j, 7)] =
                    (2.0 * u[IDX(i, j, 4)] - 27.0 * u[IDX(i, j, 5)] +
                     270.0 * u[IDX(i, j, 6)] - 490.0 * u[IDX(i, j, 7)] +
                     270.0 * u[IDX(i, j, 8)] - 27.0 * u[IDX(i, j, 9)] +
                     2.0 * u[IDX(i, j, 10)]) *
                    idz_sqrd_by_180;
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
                // This is a centered sixth order stencil.
                DzDzu[IDX(i, j, ke - 4)] =
                    (2.0 * u[IDX(i, j, ke - 7)] - 27.0 * u[IDX(i, j, ke - 6)] +
                     270.0 * u[IDX(i, j, ke - 5)] -
                     490.0 * u[IDX(i, j, ke - 4)] +
                     270.0 * u[IDX(i, j, ke - 3)] -
                     27.0 * u[IDX(i, j, ke - 2)] + 2.0 * u[IDX(i, j, ke - 1)]) *
                    idz_sqrd_by_180;

                // This is a centered fourth order stencil.
                DzDzu[IDX(i, j, ke - 3)] =
                    (-u[IDX(i, j, ke - 5)] + 16.0 * u[IDX(i, j, ke - 4)] -
                     30.0 * u[IDX(i, j, ke - 3)] + 16.0 * u[IDX(i, j, ke - 2)] -
                     u[IDX(i, j, ke - 1)]) *
                    idz_sqrd_by_12;

                // This is a centered second order stencil.
                DzDzu[IDX(i, j, ke - 2)] =
                    (u[IDX(i, j, ke - 3)] - 2.0 * u[IDX(i, j, ke - 2)] +
                     u[IDX(i, j, ke - 1)]) *
                    idz_sqrd;

                // This is a (totally) shifted second order stencil.
                DzDzu[IDX(i, j, ke - 1)] =
                    (-u[IDX(i, j, ke - 4)] + 4.0 * u[IDX(i, j, ke - 3)] -
                     5.0 * u[IDX(i, j, ke - 2)] + 2.0 * u[IDX(i, j, ke - 1)]) *
                    idz_sqrd;
            }
        }
    }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                if (std::isnan(DzDzu[pp]))
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
            }
        }
    }
#endif
}

// Explicit instiation for P = 4 and 5 for deriv866_[dim]
template void deriv8666_xx<4>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);
template void deriv8666_xx<5>(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag);

template void deriv8666_yy<4>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);
template void deriv8666_yy<5>(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag);

template void deriv8666_zz<4>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);
template void deriv8666_zz<5>(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag);

}  // namespace dendroderivs
