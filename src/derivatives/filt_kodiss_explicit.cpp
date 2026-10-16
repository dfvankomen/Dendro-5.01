#include "derivatives/filt_kodiss_explicit.h"

namespace dendroderivs {

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
// ---------------------------------------------------------------------------
// Physical-face closures for KO2 and KO6 (rewritten 2026-08-29: the previous
// rows were anti-dissipative on the Nyquist mode, see testKOSign). `p0` is the
// first active point on the face, `s` the stride pointing INTO the domain
// (negative on the high face), `h` the spacing. Every row must be dissipative:
// testKOSign checks the Nyquist sign and the spectrum of the assembled 1-D
// operator with both faces closed.
// ---------------------------------------------------------------------------

// KO2: the summation-by-parts form -H^{-1} D2^T D2 / 16 with the second-order
// norm H = diag(1/2, 1, ...) — rows 0..2 one-sided, the centred 4th difference
// from row 3. Negative semi-definite in the H inner product by construction
// (<u, A u>_H = -|D2 u|^2 / 16).
static inline void ko2_face_rows(double *const Du, const double *const u,
                                 const long p0, const long s, const double h) {
    const double f  = 1.0 / (16.0 * h);
    const double u0 = u[p0], u1 = u[p0 + s], u2 = u[p0 + 2 * s], u3 = u[p0 + 3 * s], u4 = u[p0 + 4 * s];
    Du[p0]         = f * (-4.0 * u0 + 8.0 * u1 - 4.0 * u2);
    Du[p0 + s]     = f * (4.0 * u0 - 9.0 * u1 + 6.0 * u2 - u3);
    Du[p0 + 2 * s] = f * (-2.0 * u0 + 6.0 * u1 - 7.0 * u2 + 4.0 * u3 - u4);
}

// KO6: the layering ko_deriv8 uses — the production KO4 one-sided rows
// (ko_deriv42, unchanged) on the first three points, the centred KO4 stencil
// on the fourth, and KO6 proper from the fifth, where its 9-point stencil
// reads active points only.
static inline void ko6_face_rows(double *const Du, const double *const u,
                                 const long p0, const long s, const double h) {
    const double smr3 = 59.0 / 48.0 * 64 * h, smr2 = 43.0 / 48.0 * 64 * h, smr1 = 49.0 / 48.0 * 64 * h;
    const double u0 = u[p0], u1 = u[p0 + s], u2 = u[p0 + 2 * s], u3 = u[p0 + 3 * s];
    const double u4 = u[p0 + 4 * s], u5 = u[p0 + 5 * s], u6 = u[p0 + 6 * s];
    Du[p0]         = (u3 - 3.0 * u2 + 3.0 * u1 - u0) / smr3;
    Du[p0 + s]     = (u4 - 6.0 * u3 + 12.0 * u2 - 10.0 * u1 + 3.0 * u0) / smr2;
    Du[p0 + 2 * s] = (u5 - 6.0 * u4 + 15.0 * u3 - 19.0 * u2 + 12.0 * u1 - 3.0 * u0) / smr1;
    Du[p0 + 3 * s] = (-1.0 / 64.0 / h) * (-u0 + 6.0 * u1 - 15.0 * u2 + 20.0 * u3 - 15.0 * u4 + 6.0 * u5 - u6);
}

// KO8: ko6_face_rows, KO6 proper on the fifth point, KO8 from the sixth.
static inline void ko8_face_rows(double *const Du, const double *const u,
                                 const long p0, const long s, const double h) {
    ko6_face_rows(Du, u, p0, s, h);
    const long q = p0 + 4 * s;
    Du[q]        = (-1.0 / 256.0 / h) *
                   (u[q - 4 * s] - 8.0 * u[q - 3 * s] + 28.0 * u[q - 2 * s] -
                    56.0 * u[q - s] + 70.0 * u[q] - 56.0 * u[q + s] +
                    28.0 * u[q + 2 * s] - 8.0 * u[q + 3 * s] + u[q + 4 * s]);
}

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
        for (int k = kb; k < ke; k++)
            for (int j = jb; j < je; j++) ko2_face_rows(Du, u, IDX(ib, j, k), 1, dx);
    }
    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (int k = kb; k < ke; k++)
            for (int j = jb; j < je; j++) ko2_face_rows(Du, u, IDX(ie - 1, j, k), -1, dx);
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

    if (bflag & (1u << OCT_DIR_DOWN)) {
        for (int k = kb; k < ke; k++)
            for (int i = ib; i < ie; i++) ko2_face_rows(Du, u, IDX(i, jb, k), nx, dy);
    }
    if (bflag & (1u << OCT_DIR_UP)) {
        for (int k = kb; k < ke; k++)
            for (int i = ib; i < ie; i++) ko2_face_rows(Du, u, IDX(i, je - 1, k), -nx, dy);
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

    if (bflag & (1u << OCT_DIR_BACK)) {
        for (int j = jb; j < je; j++)
            for (int i = ib; i < ie; i++) ko2_face_rows(Du, u, IDX(i, j, kb), (long)nx * ny, dz);
    }
    if (bflag & (1u << OCT_DIR_FRONT)) {
        for (int j = jb; j < je; j++)
            for (int i = ib; i < ie; i++) ko2_face_rows(Du, u, IDX(i, j, ke - 1), -(long)nx * ny, dz);
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
        for (int k = kb; k < ke; k++)
            for (int j = jb; j < je; j++) ko6_face_rows(Du, u, IDX(ib, j, k), 1, dx);
    }
    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (int k = kb; k < ke; k++)
            for (int j = jb; j < je; j++) ko6_face_rows(Du, u, IDX(ie - 1, j, k), -1, dx);
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
        for (int k = kb; k < ke; k++)
            for (int i = ib; i < ie; i++) ko6_face_rows(Du, u, IDX(i, jb, k), nx, dy);
    }
    if (bflag & (1u << OCT_DIR_UP)) {
        for (int k = kb; k < ke; k++)
            for (int i = ib; i < ie; i++) ko6_face_rows(Du, u, IDX(i, je - 1, k), -nx, dy);
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
        for (int j = jb; j < je; j++)
            for (int i = ib; i < ie; i++) ko6_face_rows(Du, u, IDX(i, j, kb), (long)nx * ny, dz);
    }
    if (bflag & (1u << OCT_DIR_FRONT)) {
        for (int j = jb; j < je; j++)
            for (int i = ib; i < ie; i++) ko6_face_rows(Du, u, IDX(i, j, ke - 1), -(long)nx * ny, dz);
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

// KO8 interior (radius-5) + ko8_face_rows on physical faces
// Formatting mirrors the KO6 derivatives: ib/jb/kb from P, bflag gating.

template <unsigned int P>
void ko_deriv8_x(double *const Du, const double *const u, const double dx,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be between 4 and 5 (for now)!");

    // sign convention (2026-08-29): -u/h on the Nyquist mode like KO2/4/6;
    // delta^10 gives -1024 u there (gate: testKOSign)
    const double pref8      = +1.0 / (1024.0 * dx);  // KO8 radius-5 interior

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

    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (int k = kb; k < ke; k++)
            for (int j = jb; j < je; j++)
                ko8_face_rows(Du, u, IDX(ib, j, k), 1, dx);
    }
    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (int k = kb; k < ke; k++)
            for (int j = jb; j < je; j++)
                ko8_face_rows(Du, u, IDX(ie - 1, j, k), -1, dx);
    }
}

template <unsigned int P>
void ko_deriv8_y(double *const Du, const double *const u, const double dy,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be 4 or 5!");

    const double pref8 = +1.0 / (1024.0 * dy);

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

    if (bflag & (1u << OCT_DIR_DOWN)) {
        for (int k = kb; k < ke; k++)
            for (int i = ib; i < ie; i++)
                ko8_face_rows(Du, u, IDX(i, jb, k), nx, dy);
    }
    if (bflag & (1u << OCT_DIR_UP)) {
        for (int k = kb; k < ke; k++)
            for (int i = ib; i < ie; i++)
                ko8_face_rows(Du, u, IDX(i, je - 1, k), -nx, dy);
    }
}


template <unsigned int P>
void ko_deriv8_z(double *const Du, const double *const u, const double dz,
                 const unsigned int *sz, unsigned bflag) {
    static_assert(P >= 4 && P <= 5, "P must be 4 or 5!");

    const double pref8      = +1.0 / (1024.0 * dz);

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

    if (bflag & (1u << OCT_DIR_BACK)) {
        for (int j = jb; j < je; j++)
            for (int i = ib; i < ie; i++)
                ko8_face_rows(Du, u, IDX(i, j, kb), (long)nx * ny, dz);
    }
    if (bflag & (1u << OCT_DIR_FRONT)) {
        for (int j = jb; j < je; j++)
            for (int i = ib; i < ie; i++)
                ko8_face_rows(Du, u, IDX(i, j, ke - 1), -(long)nx * ny, dz);
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
