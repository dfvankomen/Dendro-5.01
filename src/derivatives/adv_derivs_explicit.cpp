/**
 * @file adv_derivs_explicit.cpp
 * @brief 4th-order upwinded (advective) explicit FD — branchless inner loop.
 *
 * Ported from BSSN_GR/src/derivs.cpp's `deriv42adv_{x,y,z}` so dendrolib
 * is the single source of truth for explicit derivative stencils. The
 * interior stencil is identical to the original; the inner loop is
 * rewritten to compute both the forward- and backward-biased stencils
 * and blend by `betaX[pp] > 0.0`. This replaces the original data-
 * dependent `if/else` that the compiler couldn't vectorize around.
 */

#include "derivatives/adv_derivs_explicit.h"

#include "dendro.h"  // OCT_DIR_LEFT, OCT_DIR_RIGHT, OCT_DIR_DOWN, OCT_DIR_UP,
                     // OCT_DIR_BACK, OCT_DIR_FRONT

namespace dendroderivs {

#define IDX(i, j, k) ((i) + nx * ((j) + ny * (k)))

// ----------------------------------------------------------------------
// adv_deriv42_x
// ----------------------------------------------------------------------
template <unsigned int P>
void adv_deriv42_x(double *const __restrict__ Dxu,
                   const double *const __restrict__ u, const double dx,
                   const unsigned int *sz,
                   const double *const __restrict__ betax, unsigned bflag) {
    const double idx       = 1.0 / dx;
    const double idx_by_2  = 0.50 * idx;
    const double idx_by_12 = idx / 12.0;

    const int nx           = static_cast<int>(sz[0]);
    const int ny           = static_cast<int>(sz[1]);
    const int nz           = static_cast<int>(sz[2]);
    const int ib           = static_cast<int>(P);
    const int jb           = static_cast<int>(P);
    const int kb           = static_cast<int>(P);
    const int ie           = static_cast<int>(sz[0]) - static_cast<int>(P);
    const int je           = static_cast<int>(sz[1]) - static_cast<int>(P);
    const int ke           = static_cast<int>(sz[2]) - static_cast<int>(P);

    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp = IDX(i, j, k);
                // forward-biased stencil (used when betax > 0)
                const double df =
                    (-3.0 * u[pp - 1] - 10.0 * u[pp] + 18.0 * u[pp + 1] -
                     6.0 * u[pp + 2] + u[pp + 3]) *
                    idx_by_12;
                // backward-biased stencil (used when betax <= 0)
                const double db =
                    (-u[pp - 3] + 6.0 * u[pp - 2] - 18.0 * u[pp - 1] +
                     10.0 * u[pp] + 3.0 * u[pp + 1]) *
                    idx_by_12;
                // branchless ternary; compiler emits VBLENDPD / CMOV so the
                // loop stays vectorizable even with data-dependent routing
                Dxu[pp] = (betax[pp] > 0.0) ? df : db;
            }
        }
    }

    // LEFT domain boundary: one-sided and reduced-order near i=3..5
    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (int k = kb; k < ke; k++) {
            for (int j = jb; j < je; j++) {
                Dxu[IDX(3, j, k)] =
                    (-3.0 * u[IDX(3, j, k)] + 4.0 * u[IDX(4, j, k)] -
                     u[IDX(5, j, k)]) *
                    idx_by_2;

                // i=4: upwind forward if beta>0 else central
                const double d4_f =
                    (-3.0 * u[IDX(4, j, k)] + 4.0 * u[IDX(5, j, k)] -
                     u[IDX(6, j, k)]) *
                    idx_by_2;
                const double d4_b =
                    (-u[IDX(3, j, k)] + u[IDX(5, j, k)]) * idx_by_2;
                Dxu[IDX(4, j, k)] =
                    (betax[IDX(4, j, k)] > 0.0) ? d4_f : d4_b;

                // i=5: full 5-point forward stencil if beta>0 else one-sided
                const double d5_f =
                    (-3.0 * u[IDX(4, j, k)] - 10.0 * u[IDX(5, j, k)] +
                     18.0 * u[IDX(6, j, k)] - 6.0 * u[IDX(7, j, k)] +
                     u[IDX(8, j, k)]) *
                    idx_by_12;
                const double d5_b =
                    (u[IDX(3, j, k)] - 4.0 * u[IDX(4, j, k)] +
                     3.0 * u[IDX(5, j, k)]) *
                    idx_by_2;
                Dxu[IDX(5, j, k)] =
                    (betax[IDX(5, j, k)] > 0.0) ? d5_f : d5_b;
            }
        }
    }

    // RIGHT domain boundary: mirror of LEFT
    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (int k = kb; k < ke; k++) {
            for (int j = jb; j < je; j++) {
                // i = ie-3: full 5-point backward stencil if beta<=0 else
                // one-sided forward
                const double dA_f =
                    (-3.0 * u[IDX(ie - 3, j, k)] +
                     4.0 * u[IDX(ie - 2, j, k)] - u[IDX(ie - 1, j, k)]) *
                    idx_by_2;
                const double dA_b =
                    (-u[IDX(ie - 6, j, k)] + 6.0 * u[IDX(ie - 5, j, k)] -
                     18.0 * u[IDX(ie - 4, j, k)] +
                     10.0 * u[IDX(ie - 3, j, k)] +
                     3.0 * u[IDX(ie - 2, j, k)]) *
                    idx_by_12;
                Dxu[IDX(ie - 3, j, k)] =
                    (betax[IDX(ie - 3, j, k)] > 0.0) ? dA_f : dA_b;

                // i = ie-2
                const double dB_f =
                    (-u[IDX(ie - 3, j, k)] + u[IDX(ie - 1, j, k)]) * idx_by_2;
                const double dB_b =
                    (u[IDX(ie - 4, j, k)] - 4.0 * u[IDX(ie - 3, j, k)] +
                     3.0 * u[IDX(ie - 2, j, k)]) *
                    idx_by_2;
                Dxu[IDX(ie - 2, j, k)] =
                    (betax[IDX(ie - 2, j, k)] > 0.0) ? dB_f : dB_b;

                // i = ie-1: one-sided backward regardless of beta sign
                Dxu[IDX(ie - 1, j, k)] =
                    (u[IDX(ie - 3, j, k)] - 4.0 * u[IDX(ie - 2, j, k)] +
                     3.0 * u[IDX(ie - 1, j, k)]) *
                    idx_by_2;
            }
        }
    }
}

// ----------------------------------------------------------------------
// adv_deriv42_y
// ----------------------------------------------------------------------
template <unsigned int P>
void adv_deriv42_y(double *const __restrict__ Dyu,
                   const double *const __restrict__ u, const double dy,
                   const unsigned int *sz,
                   const double *const __restrict__ betay, unsigned bflag) {
    const double idy       = 1.0 / dy;
    const double idy_by_2  = 0.50 * idy;
    const double idy_by_12 = idy / 12.0;

    const int nx           = static_cast<int>(sz[0]);
    const int ny           = static_cast<int>(sz[1]);
    const int nz           = static_cast<int>(sz[2]);
    const int ib           = static_cast<int>(P);
    const int jb           = static_cast<int>(P);
    const int kb           = static_cast<int>(P);
    const int ie           = static_cast<int>(sz[0]) - static_cast<int>(P);
    const int je           = static_cast<int>(sz[1]) - static_cast<int>(P);
    const int ke           = static_cast<int>(sz[2]) - static_cast<int>(P);
    const int stride_y     = nx;

    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp    = IDX(i, j, k);
                const double df = (-3.0 * u[pp - stride_y] - 10.0 * u[pp] +
                                   18.0 * u[pp + stride_y] -
                                   6.0 * u[pp + 2 * stride_y] +
                                   u[pp + 3 * stride_y]) *
                                  idy_by_12;
                const double db =
                    (-u[pp - 3 * stride_y] + 6.0 * u[pp - 2 * stride_y] -
                     18.0 * u[pp - stride_y] + 10.0 * u[pp] +
                     3.0 * u[pp + stride_y]) *
                    idy_by_12;
                Dyu[pp] = (betay[pp] > 0.0) ? df : db;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_DOWN)) {
        for (int k = kb; k < ke; k++) {
            for (int i = ib; i < ie; i++) {
                Dyu[IDX(i, 3, k)] =
                    (-3.0 * u[IDX(i, 3, k)] + 4.0 * u[IDX(i, 4, k)] -
                     u[IDX(i, 5, k)]) *
                    idy_by_2;

                const double d4_f =
                    (-3.0 * u[IDX(i, 4, k)] + 4.0 * u[IDX(i, 5, k)] -
                     u[IDX(i, 6, k)]) *
                    idy_by_2;
                const double d4_b =
                    (-u[IDX(i, 3, k)] + u[IDX(i, 5, k)]) * idy_by_2;
                Dyu[IDX(i, 4, k)] =
                    (betay[IDX(i, 4, k)] > 0.0) ? d4_f : d4_b;

                const double d5_f =
                    (-3.0 * u[IDX(i, 4, k)] - 10.0 * u[IDX(i, 5, k)] +
                     18.0 * u[IDX(i, 6, k)] - 6.0 * u[IDX(i, 7, k)] +
                     u[IDX(i, 8, k)]) *
                    idy_by_12;
                const double d5_b =
                    (u[IDX(i, 3, k)] - 4.0 * u[IDX(i, 4, k)] +
                     3.0 * u[IDX(i, 5, k)]) *
                    idy_by_2;
                Dyu[IDX(i, 5, k)] =
                    (betay[IDX(i, 5, k)] > 0.0) ? d5_f : d5_b;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_UP)) {
        for (int k = kb; k < ke; k++) {
            for (int i = ib; i < ie; i++) {
                const double dA_f =
                    (-3.0 * u[IDX(i, je - 3, k)] +
                     4.0 * u[IDX(i, je - 2, k)] - u[IDX(i, je - 1, k)]) *
                    idy_by_2;
                const double dA_b =
                    (-u[IDX(i, je - 6, k)] + 6.0 * u[IDX(i, je - 5, k)] -
                     18.0 * u[IDX(i, je - 4, k)] +
                     10.0 * u[IDX(i, je - 3, k)] +
                     3.0 * u[IDX(i, je - 2, k)]) *
                    idy_by_12;
                Dyu[IDX(i, je - 3, k)] =
                    (betay[IDX(i, je - 3, k)] > 0.0) ? dA_f : dA_b;

                const double dB_f =
                    (-u[IDX(i, je - 3, k)] + u[IDX(i, je - 1, k)]) * idy_by_2;
                const double dB_b =
                    (u[IDX(i, je - 4, k)] - 4.0 * u[IDX(i, je - 3, k)] +
                     3.0 * u[IDX(i, je - 2, k)]) *
                    idy_by_2;
                Dyu[IDX(i, je - 2, k)] =
                    (betay[IDX(i, je - 2, k)] > 0.0) ? dB_f : dB_b;

                Dyu[IDX(i, je - 1, k)] =
                    (u[IDX(i, je - 3, k)] - 4.0 * u[IDX(i, je - 2, k)] +
                     3.0 * u[IDX(i, je - 1, k)]) *
                    idy_by_2;
            }
        }
    }
}

// ----------------------------------------------------------------------
// adv_deriv42_z
// ----------------------------------------------------------------------
template <unsigned int P>
void adv_deriv42_z(double *const __restrict__ Dzu,
                   const double *const __restrict__ u, const double dz,
                   const unsigned int *sz,
                   const double *const __restrict__ betaz, unsigned bflag) {
    const double idz       = 1.0 / dz;
    const double idz_by_2  = 0.50 * idz;
    const double idz_by_12 = idz / 12.0;

    const int nx           = static_cast<int>(sz[0]);
    const int ny           = static_cast<int>(sz[1]);
    const int nz           = static_cast<int>(sz[2]);
    const int ib           = static_cast<int>(P);
    const int jb           = static_cast<int>(P);
    const int kb           = static_cast<int>(P);
    const int ie           = static_cast<int>(sz[0]) - static_cast<int>(P);
    const int je           = static_cast<int>(sz[1]) - static_cast<int>(P);
    const int ke           = static_cast<int>(sz[2]) - static_cast<int>(P);
    const int stride_z     = nx * ny;

    for (int k = kb; k < ke; k++) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const int pp    = IDX(i, j, k);
                const double df = (-3.0 * u[pp - stride_z] - 10.0 * u[pp] +
                                   18.0 * u[pp + stride_z] -
                                   6.0 * u[pp + 2 * stride_z] +
                                   u[pp + 3 * stride_z]) *
                                  idz_by_12;
                const double db =
                    (-u[pp - 3 * stride_z] + 6.0 * u[pp - 2 * stride_z] -
                     18.0 * u[pp - stride_z] + 10.0 * u[pp] +
                     3.0 * u[pp + stride_z]) *
                    idz_by_12;
                Dzu[pp] = (betaz[pp] > 0.0) ? df : db;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_BACK)) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                Dzu[IDX(i, j, 3)] =
                    (-3.0 * u[IDX(i, j, 3)] + 4.0 * u[IDX(i, j, 4)] -
                     u[IDX(i, j, 5)]) *
                    idz_by_2;

                const double d4_f =
                    (-3.0 * u[IDX(i, j, 4)] + 4.0 * u[IDX(i, j, 5)] -
                     u[IDX(i, j, 6)]) *
                    idz_by_2;
                const double d4_b =
                    (-u[IDX(i, j, 3)] + u[IDX(i, j, 5)]) * idz_by_2;
                Dzu[IDX(i, j, 4)] =
                    (betaz[IDX(i, j, 4)] > 0.0) ? d4_f : d4_b;

                const double d5_f =
                    (-3.0 * u[IDX(i, j, 4)] - 10.0 * u[IDX(i, j, 5)] +
                     18.0 * u[IDX(i, j, 6)] - 6.0 * u[IDX(i, j, 7)] +
                     u[IDX(i, j, 8)]) *
                    idz_by_12;
                const double d5_b =
                    (u[IDX(i, j, 3)] - 4.0 * u[IDX(i, j, 4)] +
                     3.0 * u[IDX(i, j, 5)]) *
                    idz_by_2;
                Dzu[IDX(i, j, 5)] =
                    (betaz[IDX(i, j, 5)] > 0.0) ? d5_f : d5_b;
            }
        }
    }

    if (bflag & (1u << OCT_DIR_FRONT)) {
        for (int j = jb; j < je; j++) {
            for (int i = ib; i < ie; i++) {
                const double dA_f =
                    (-3.0 * u[IDX(i, j, ke - 3)] +
                     4.0 * u[IDX(i, j, ke - 2)] - u[IDX(i, j, ke - 1)]) *
                    idz_by_2;
                const double dA_b =
                    (-u[IDX(i, j, ke - 6)] + 6.0 * u[IDX(i, j, ke - 5)] -
                     18.0 * u[IDX(i, j, ke - 4)] +
                     10.0 * u[IDX(i, j, ke - 3)] +
                     3.0 * u[IDX(i, j, ke - 2)]) *
                    idz_by_12;
                Dzu[IDX(i, j, ke - 3)] =
                    (betaz[IDX(i, j, ke - 3)] > 0.0) ? dA_f : dA_b;

                const double dB_f =
                    (-u[IDX(i, j, ke - 3)] + u[IDX(i, j, ke - 1)]) * idz_by_2;
                const double dB_b =
                    (u[IDX(i, j, ke - 4)] - 4.0 * u[IDX(i, j, ke - 3)] +
                     3.0 * u[IDX(i, j, ke - 2)]) *
                    idz_by_2;
                Dzu[IDX(i, j, ke - 2)] =
                    (betaz[IDX(i, j, ke - 2)] > 0.0) ? dB_f : dB_b;

                Dzu[IDX(i, j, ke - 1)] =
                    (u[IDX(i, j, ke - 3)] - 4.0 * u[IDX(i, j, ke - 2)] +
                     3.0 * u[IDX(i, j, ke - 1)]) *
                    idz_by_2;
            }
        }
    }
}

#undef IDX

// --- explicit instantiations ---------------------------------------------
// BSSN runs with element order 6 (pad-width 3). Add more instantiations
// when another pad-width is needed.

template void adv_deriv42_x<3>(double *const, const double *const, const double,
                               const unsigned int *, const double *const,
                               unsigned);
template void adv_deriv42_y<3>(double *const, const double *const, const double,
                               const unsigned int *, const double *const,
                               unsigned);
template void adv_deriv42_z<3>(double *const, const double *const, const double,
                               const unsigned int *, const double *const,
                               unsigned);

template void adv_deriv42_x<4>(double *const, const double *const, const double,
                               const unsigned int *, const double *const,
                               unsigned);
template void adv_deriv42_y<4>(double *const, const double *const, const double,
                               const unsigned int *, const double *const,
                               unsigned);
template void adv_deriv42_z<4>(double *const, const double *const, const double,
                               const unsigned int *, const double *const,
                               unsigned);

}  // namespace dendroderivs
