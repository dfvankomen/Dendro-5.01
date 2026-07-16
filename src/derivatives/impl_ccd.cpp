#include "derivatives/impl_ccd.h"

#include <vector>

namespace dendroderivs {

/**
 * @brief Classical 3-point combined compact difference, 6th order.
 *
 * Chu & Fan (1998). The interior coefficients were re-derived from scratch by
 * exact rational Taylor matching (scripts/ccd_operators.py) rather than
 * transcribed, following the NOVA convention, and they reproduce the published
 * values:
 *
 *   eq1 (f'):   a1 = 7/16,  b1 = 1/16,  c1 = 15/16
 *   eq2 (f''):  a2 = -9/8,  b2 = -1/8,  c2 = 3
 *
 * so, at unit spacing, with g = h f' and w = h^2 f'':
 *
 *   eq1:  a1(g_{i-1}+g_{i+1}) + g_i + b1(w_{i-1}-w_{i+1})
 *             = c1(f_{i+1}-f_{i-1})
 *   eq2:  a2(g_{i-1}-g_{i+1}) + b2(w_{i-1}+w_{i+1}) + w_i
 *             = c2(f_{i-1}-2f_i+f_{i+1})
 *
 * Both interior equations are 6th order. The boundary closures below are
 * **8th** order (f stencil over offsets 0..6) — deliberately two orders above
 * the interior, which is not the obvious choice and is worth explaining.
 *
 * A 6th-order closure (f offsets 0..4) is formally "matched" to the interior
 * and still measures ~5.0 for f'' on the project's convergence test, against a
 * formal 6 — outside the ~0.5 gate. The reason is that D = A^-1 B is *dense*,
 * so a closure with a large truncation constant contaminates every point in
 * the block, decaying inward but never vanishing; at the block sizes we
 * actually run (n_active = 7..43) the interior never gets far enough from the
 * closure for that to die off. Raising the closure order shrinks that constant.
 * Measured f'' order over the sin(2*pi*x) sweep, interior region:
 *
 *     closure order 5 (f 0..3):  4.98 ... 5.54     cond(A) 6.1e4
 *     closure order 6 (f 0..4):  4.98              cond(A) 1.0e6
 *     closure order 8 (f 0..6):  5.91  <- this     cond(A) 2.8e4
 *
 * so the 8th-order closure is better on every axis at once: measured order,
 * conditioning, and absolute error (~4x lower at the finest h). The lesson
 * generalizes to the collaborators' schemes: for CCD, match the closure to the
 * *error constant* you need, not to the interior's formal order.
 *
 * The f stencil reaches offset 6, i.e. 7 points, which is exactly n_active at
 * the tightest configuration we support (n = 13, pw = 3, both faces bounded ->
 * n_active = 7). It fits, but with nothing to spare: a wider closure, or a
 * larger pw at n = 13, will throw out of createMatrix rather than silently
 * misbehave.
 *
 * @warning The boundary f' equation deliberately has NO w_0 term (its w row
 * leads with an explicit 0.0). This is not an oversight and must not be
 * "fixed". Deriving both closures from the same term set at maximal order gives
 * a one-dimensional solution space, so both equations collapse onto the same
 * relation -- the f'' closure comes out as exactly 49/6 times the f' one. The
 * two rows are then linearly dependent, the assembled system is singular
 * (rcond ~ 1e-18), and the solver returns garbage instead of failing. Omitting
 * w_0 from the f' equation makes the pair independent by construction: this row
 * has w_0 = 0 and the f'' row has w_0 = 1, so neither can be a multiple of the
 * other. createCCDMatrixSystemForSingleSize guards this, but the guard is a
 * backstop, not a licence to reintroduce the term.
 */
CCDDiagonalEntries* createCCD6Diagonals() {
    CCDBlocks blk;

    // Interior: one entry per off-diagonal, so a 3-point scheme is one each.
    // These are the block coefficients with dx already cancelled by the scaled
    // unknowns -- the same numbers you would write in A_k / B_k.
    //
    // NOTE the leading minus in A_k = [-b1 ...]: the f'_{i±1} coefficient of
    // this scheme is +7/16, so b1 is -7/16. Same for b2. See CCDBlocks.
    blk.b1 = {-7.0 / 16.0};  // g family, eq1
    blk.c1 = {1.0 / 16.0};   // w family, eq1
    blk.a1 = {15.0 / 16.0};  // f family, eq1 (rhs)

    blk.b2 = {-9.0 / 8.0};   // g family, eq2
    blk.c2 = {1.0 / 8.0};    // w family, eq2
    blk.a2 = {3.0};          // f family, eq2 (rhs); center -2*3 = -6 is derived

    // One-sided 8th-order closure at the boundary node. Rows are dense lists
    // indexed from column 0, so an omitted term is an explicit 0.0.
    //   eq1:  g_0 - (17/10) g_1 - 3 w_1                = sum_k f_k coefficients
    //   eq2:  (89/10) g_0 + (43/5) g_1 + w_0 - 6 w_1   = sum_k f_k coefficients
    blk.gBoundary1 = {{1.0, -17.0 / 10.0}};
    blk.wBoundary1 = {{0.0, -3.0}};  // leading 0.0: no w_0 — see the warning
    blk.fBoundary1 = {{-89.0 / 20.0, 6379.0 / 600.0, -15.0 / 2.0, 5.0 / 3.0,
                       -5.0 / 12.0, 3.0 / 40.0, -1.0 / 150.0}};

    blk.gBoundary2 = {{89.0 / 10.0, 43.0 / 5.0}};
    blk.wBoundary2 = {{1.0, -6.0}};
    blk.fBoundary2 = {{-41929.0 / 1800.0, 8959.0 / 300.0, -15.0 / 2.0,
                       10.0 / 9.0, -5.0 / 24.0, 3.0 / 100.0, -1.0 / 450.0}};

    // ends are symmetric, so the lower closure mirrors the upper: the ctor
    // copies the rows and the builder applies the per-family parity when it
    // places them.
    return ccd_from_blocks(blk);
}

}  // namespace dendroderivs
