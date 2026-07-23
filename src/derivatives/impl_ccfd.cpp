#include "derivatives/impl_ccfd.h"

#include <vector>

namespace dendroderivs {

/**
 * @brief Classical 3-point combined compact finite difference, 6th order.
 *
 * Chu & Fan (1998), written here in Sengupta's six-matrix notation (the same
 * form the collaborators' Mathematica notebook emits; see
 * findings/CCFD_Matrix_Implementation.pdf). The interior coefficients were
 * re-derived from scratch by exact rational Taylor matching
 * (scripts/ccfd_operators.py) rather than transcribed, and they reproduce the
 * published values. In the block/whiteboard convention:
 *
 *   eq1 (f'):   a1 = 7/16,  b1 = 1/16,  c1 = 15/16
 *   eq2 (f''):  a2 = -9/8,  b2 = -1/8,  c2 = 3
 *
 * so, at unit spacing, with g = h f' and w = h^2 f'':
 *
 *   eq1:  a1(g_{i-1}+g_{i+1}) + g_i + b1(w_{i-1}-w_{i+1})
 *             = c1(f_{i+1}-f_{i-1})            (== A1 g + B1 w = C1 f)
 *   eq2:  a2(g_{i-1}-g_{i+1}) + b2(w_{i-1}+w_{i+1}) + w_i
 *             = c2(f_{i-1}-2f_i+f_{i+1})       (== A2 g + B2 w = C2 f)
 *
 * The six Sengupta matrices' interior rows are just those equations written
 * out, centered, with each matrix's symmetry made explicit (A1,B2,C2 symmetric;
 * A2,B1,C1 antisymmetric) and C2's center forced to s = -2*sum(a2_k) = -6 so the
 * f'' row annihilates a constant. This is a hand-maintained reference scheme, so
 * it is written directly rather than through the full septadiagonal template; a
 * generated header fills the template's named-coefficient slots instead, but
 * lands on the same CCFDDiagonalEntries.
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
 * generalizes to the collaborators' schemes: for CCFD, match the closure to the
 * *error constant* you need, not to the interior's formal order.
 *
 * The f stencil reaches offset 6, i.e. 7 points, which is exactly n_active at
 * the tightest configuration we support (n = 13, pw = 3, both faces bounded ->
 * n_active = 7). It fits, but with nothing to spare: a wider closure, or a
 * larger pw at n = 13, will throw out of createMatrix rather than silently
 * misbehave.
 *
 * @warning The boundary f' equation deliberately has NO w_0 term (B1Boundary's
 * first column is an explicit 0.0). This is not an oversight and must not be
 * "fixed". Deriving both closures from the same term set at maximal order gives
 * a one-dimensional solution space, so both equations collapse onto the same
 * relation -- the f'' closure comes out as exactly 49/6 times the f' one. The
 * two rows are then linearly dependent, the assembled system is singular
 * (rcond ~ 1e-18), and the solver returns garbage instead of failing. Omitting
 * w_0 from the f' equation makes the pair independent by construction: this row
 * has w_0 = 0 and the f'' row has w_0 = 1, so neither can be a multiple of the
 * other. createCCFDMatrixSystemForSingleSize guards this, but the guard is a
 * backstop, not a licence to reintroduce the term.
 */
CCFDDiagonalEntries* createCCFD6Diagonals() {
    // --- interior rows, centered (length 2m+1 = 3 for a 3-point scheme) -----
    // A: f' coefficients (alpha), B: f'' coefficients (beta), C: f coefficients
    // (a). Symmetry and C2's forced center are baked in, exactly as a generated
    // header would emit them.
    std::vector<double> A1Interior{7.0 / 16.0, 1.0, 7.0 / 16.0};
    std::vector<double> A2Interior{-9.0 / 8.0, 0.0, 9.0 / 8.0};
    std::vector<double> B1Interior{1.0 / 16.0, 0.0, -1.0 / 16.0};
    std::vector<double> B2Interior{-1.0 / 8.0, 1.0, -1.0 / 8.0};
    std::vector<double> C1Interior{-15.0 / 16.0, 0.0, 15.0 / 16.0};
    std::vector<double> C2Interior{3.0, -6.0, 3.0};

    // --- one-sided 8th-order closure at the boundary node -------------------
    // One dense row per near-boundary node, indexed from column 0 (an omitted
    // term is an explicit 0.0). Here there is one closure row per end.
    //   eq1:  g_0 - (17/10) g_1 - 3 w_1                = sum_k C1 f_k
    //   eq2:  (89/10) g_0 + (43/5) g_1 + w_0 - 6 w_1   = sum_k C2 f_k
    std::vector<std::vector<double>> A1Boundary{{1.0, -17.0 / 10.0}};
    std::vector<std::vector<double>> A2Boundary{{89.0 / 10.0, 43.0 / 5.0}};
    std::vector<std::vector<double>> B1Boundary{
        {0.0, -3.0}};  // leading 0.0: no w_0 — see the warning
    std::vector<std::vector<double>> B2Boundary{{1.0, -6.0}};
    std::vector<std::vector<double>> C1Boundary{
        {-89.0 / 20.0, 6379.0 / 600.0, -15.0 / 2.0, 5.0 / 3.0, -5.0 / 12.0,
         3.0 / 40.0, -1.0 / 150.0}};
    std::vector<std::vector<double>> C2Boundary{
        {-41929.0 / 1800.0, 8959.0 / 300.0, -15.0 / 2.0, 10.0 / 9.0,
         -5.0 / 24.0, 3.0 / 100.0, -1.0 / 450.0}};

    // ends are symmetric, so the lower closure mirrors the upper: the ctor
    // copies the rows and the builder applies the per-matrix parity when it
    // places them. Constructor argument order matches the generated-header
    // template: the six interior vectors then the six boundary blocks.
    return new CCFDDiagonalEntries(A1Interior, A2Interior, B1Interior,
                                   B2Interior, C1Interior, C2Interior,
                                   A1Boundary, A2Boundary, B1Boundary,
                                   B2Boundary, C1Boundary, C2Boundary);
}

}  // namespace dendroderivs
