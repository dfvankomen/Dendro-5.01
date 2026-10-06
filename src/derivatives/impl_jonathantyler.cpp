
#include "derivatives/impl_jonathantyler.h"

#include "derivatives/derivs_compact.h"

namespace dendroderivs {

MatrixDiagonalEntries* createJTT4DiagonalsFirstOrder() {
    // D_x coeffs
    // row 1
    constexpr double alpha1_x = 3.0;
    constexpr double a1_x     = -17.0 / 6.0;
    constexpr double b1_x     = 3.0 / 2.0;
    constexpr double c1_x     = 3.0 / 2.0;
    constexpr double d1_x     = -1.0 / 6.0;
    // interior
    constexpr double alpha_x  = 1.0 / 4.0;
    constexpr double a_x      = 3.0 / 2.0;

    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{{1.0, alpha1_x}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{alpha_x, 1.0, alpha_x};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{{a1_x, b1_x, c1_x, d1_x}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{-a_x / 2.0, 0.0, a_x / 2.0};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createJTT4DiagonalsSecondOrder() {
    // 4th-order tridiagonal compact 2nd-derivative scheme.
    //
    // Interior (4th-order Padé, standard):
    //   (1/10) f''_{i-1} + f''_i + (1/10) f''_{i+1}
    //     = (1/h^2) ((6/5) f_{i-1} - (12/5) f_i + (6/5) f_{i+1})
    //   Derived from Taylor matching: alpha = 1/10, a = 6/5 cancel
    //   through f^(4); leading error h^4 f^(6).
    //
    // Node-0 boundary closure (5-point one-sided, 4th-order):
    //   f''_0 + alpha1 f''_1
    //     = (1/h^2) (q_0 f_0 + q_1 f_1 + q_2 f_2 + q_3 f_3 + q_4 f_4)
    //   with alpha1 = 10 and q = {145/12, -76/3, 29/2, -4/3, 1/12};
    //   6 unknowns matched against 2 consistency + 4 Taylor conditions
    //   (m=0..3). Leading error h^4 f^(6) * (-7/180).
    //
    // PRIOR BUG (now fixed): the boundary P row used to be
    //   {alpha1, 1, alpha1}  → placed P[0,0]=10, P[0,1]=1, P[0,2]=10,
    //   which is a nonsensical row. The correct encoding for the
    //   intended equation f''_0 + alpha1 f''_1 is {1, alpha1} placed
    //   at cols 0 and 1.
    // See scripts/derive_jtt4_2nd_boundary.py for the derivation.
    constexpr double alpha1_xx = 10.0;
    constexpr double a1_xx     = 145.0 / 12.0;
    constexpr double b1_xx     = -76.0 / 3.0;
    constexpr double c1_xx     = 29.0 / 2.0;
    constexpr double d1_xx     = -4.0 / 3.0;
    constexpr double e1_xx     = 1.0 / 12.0;
    constexpr double alpha_xx  = 1.0 / 10.0;
    constexpr double a_xx      = 6.0 / 5.0;

    // Node-0 P boundary row: 2 entries placed at cols 0, 1 → P[0,0]=1
    // (diagonal), P[0,1]=alpha1.
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, alpha1_xx}};

    std::vector<double> P2DiagInterior{alpha_xx, 1.0, alpha_xx};
    std::vector<std::vector<double>> Q2DiagBoundary{
        {a1_xx, b1_xx, c1_xx, d1_xx, e1_xx}};
    std::vector<double> Q2DiagInterior{a_xx, -2.0 * a_xx, a_xx};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}

// penta diagonal first order
MatrixDiagonalEntries* createJTP6DiagonalsFirstOrder() {
    // D_x coeffs
    // row 1
    const double alpha = 17.0 / 57.0;
    const double beta  = -1.0 / 114.0;

    // NOTE: the JT thesis doesn't have boundary of idx 1 (node #2) for this
    // scheme, we opted to use JTT6's second rows

    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        // {1.0, 8.0, 6.0},
        {1.0, 5.0},
        {1.0 / 8.0, 1.0, 3.0 / 4.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{beta, alpha, 1.0, alpha, beta};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        // {a1, b1, c1, d1, e1},
        // {-43.0 / 12.0, -20.0 / 3.0, 9.0, 4.0 / 3.0, -1.0 / 12.0},
        {-197.0 / 60.0, -5.0 / 12.0, 5.0, -5.0 / 3.0, 5.0 / 12.0, -1.0 / 20.0},
        {-43.0 / 96.0, -5.0 / 6.0, 9.0 / 8.0, 1.0 / 6.0, -1.0 / 96.0, 0.0}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{-15.0 / 19.0, 0.0, 15.0 / 19.0};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createJTP6DiagonalsSecondOrder() {
    // pentadiagonal 6th-order compact second derivative
    // interior: alpha=12/97, beta=-1/194, a=120/97
    // these satisfy the 6th-order taylor expansion constraints for P f'' = Q f / h^2
    const double alpha = 12.0 / 97.0;
    const double beta  = -1.0 / 194.0;
    const double a     = 120.0 / 97.0;

    // boundary closures: row 0 from thesis Table 26 (P6 at node 1).
    // Row 1 (node-1 closure) is derived: Tyler's Table 28 has no P6 entry,
    // so we follow the JTT6 pattern of an asymmetric 3-entry P at the
    // boundary row (alpha1 = interior alpha for smooth transition,
    // alpha2 free). This gives 6th-order accuracy with a 7-point Q row,
    // which fits within n_active >= 7 (required for LEFTRIGHT_BOUNDARY
    // matrices at eleorder=6).
    const double alpha1_row1 = alpha;                    // = 12/97
    const double alpha2_row1 = -445.0 / 194.0;           // derived
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, 11.0 / 2.0, -131.0 / 4.0},
        {alpha1_row1, 1.0, alpha2_row1}};

    std::vector<double> P2DiagInterior{beta, alpha, 1.0, alpha, beta};

    // Q boundary row 1 derived via Taylor matching together with alpha2_row1.
    // See scripts/derive_jtp6_2nd_boundary.py. Q offsets: k = -1..5 from
    // node 1. Conditions: 2 consistency (sum_Q = sum_Q*k = 0) + Taylor
    // matching m = 0..5 = 8 equations in 8 unknowns (7 q's + alpha2_row1).
    // Leading error: h^6 * f^(8) * (-7.015e-3).
    std::vector<std::vector<double>> Q2DiagBoundary{
        {177.0 / 16.0, -507.0 / 8.0, 783.0 / 8.0, -201.0 / 4.0,
         81.0 / 16.0, -3.0 / 8.0},
        {51851.0 / 34920.0, -11397.0 / 1940.0, 2931.0 / 388.0, -5987.0 / 1746.0,
         207.0 / 776.0, 3.0 / 1940.0, -31.0 / 8730.0}};

    // Q interior: standard second-derivative stencil {a, -2a, a}
    std::vector<double> Q2DiagInterior{a, -2.0 * a, a};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createJTT6DiagonalsFirstOrder() {
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, 5.0, 0.0}, {1.0 / 8.0, 1.0, 3.0 / 4.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0 / 3.0, 1.0, 1.0 / 3.0};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {-197.0 / 60.0, -5.0 / 12.0, 5.0, -5.0 / 3.0, 5.0 / 12.0, -1.0 / 20.0},
        {-43.0 / 96.0, -5.0 / 6.0, 9.0 / 8.0, 1.0 / 6.0, -1.0 / 96.0, 0.0}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{-1.0 / 36.0, -7.0 / 9.0, 0.0, 7.0 / 9.0,
                                       1.0 / 36.0};
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createJTT6DiagonalsSecondOrder() {
    // boundary elements for P matrix for 2nd derivative
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, 126.0 / 11.0}, {2.0 / 11.0, 1.0, -131.0 / 22.0}};

    // diagonal elements for P matrix for 2nd derivative.
    std::vector<double> P2DiagInterior{2.0 / 11.0, 1.0, 2.0 / 11.0};
    // boundary elements for Q matrix for 2nd derivative
    std::vector<std::vector<double>> Q2DiagBoundary{
        {13097.0 / 990.0, -2943.0 / 110.0, 573.0 / 44.0, 167.0 / 99.0,
         -18.0 / 11.0, 57.0 / 110.0, -131.0 / 1980.0},
        {177.0 / 88.0, -507.0 / 44.0, 783.0 / 44.0, -201.0 / 22.0, 81.0 / 88.0,
         -3.0 / 44.0}};
    // boundary elements for Q matrix for 2nd derivative
    constexpr double Q2DI_a = 12.0 / 11.0;  // a parameter
    constexpr double Q2DI_b = 3.0 / 11.0;   // b parameter
    std::vector<double> Q2DiagInterior{Q2DI_b / 4.0, Q2DI_a,
                                       -2.0 * (Q2DI_a + (Q2DI_b / 4.0)), Q2DI_a,
                                       Q2DI_b / 4.0};
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}

}  // namespace dendroderivs
