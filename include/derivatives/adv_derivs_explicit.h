/**
 * @file adv_derivs_explicit.h
 * @brief Upwinded (advective) explicit finite-difference derivatives.
 *
 * These are 4th-order biased stencils used by formulations like BSSN's
 * moving-puncture gauge advection terms. The upwind direction is chosen
 * per-cell based on the local sign of a shift-vector component (`betax`,
 * `betay`, or `betaz`). The rest of dendrolib's explicit derivatives live
 * in derivs_explicit.h; this header adds the `beta`-dependent variants.
 *
 * Implementation note: the inner loop computes *both* the forward- and
 * backward-biased stencils and selects by a branchless ternary. This
 * lets GCC/Clang emit a VBLENDPD / CMOV and keep the loop auto-
 * vectorizable even with data-dependent per-cell routing. On AVX2 this
 * is typically ~2x faster than the naive if/else form because the branch
 * mispredicts frequently when beta oscillates in the interior.
 *
 * Template parameter P is the padding width. BSSN today runs with P = 3
 * (element order 6, half-width 3) and that's the only instantiation we
 * currently provide — the interior bounds generalize cleanly to higher
 * P but the boundary handling hard-codes the 3-cell bias at the domain
 * boundary (matching the original BSSN implementation).
 */
#pragma once

#include "derivatives.h"

namespace dendroderivs {

// raw function pointer type for advective stencils (one extra `betax`
// pointer compared to the centered-FD StencilFn).
using AdvStencilFn = void (*)(double *const, const double *const,
                              const double, const unsigned int *,
                              const double *const, unsigned int);

template <unsigned int P>
void adv_deriv42_x(double *const Dxu, const double *const u, const double dx,
                   const unsigned int *sz, const double *const betax,
                   unsigned bflag);

template <unsigned int P>
void adv_deriv42_y(double *const Dyu, const double *const u, const double dy,
                   const unsigned int *sz, const double *const betay,
                   unsigned bflag);

template <unsigned int P>
void adv_deriv42_z(double *const Dzu, const double *const u, const double dz,
                   const unsigned int *sz, const double *const betaz,
                   unsigned bflag);

}  // namespace dendroderivs
