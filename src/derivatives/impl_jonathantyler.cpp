
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
    // NOTE: SOMETHING IS WRONG WITH THESE VALUES?
    // TODO: this needs to be fixed!
    // D_xx coeffs
    // row 1
    constexpr double alpha1_xx = 10.0;
    constexpr double a1_xx     = 145.0 / 12.0;
    constexpr double b1_xx     = -76.0 / 3.0;
    constexpr double c1_xx     = 29.0 / 2.0;
    constexpr double d1_xx     = -4.0 / 3.0;
    constexpr double e1_xx     = 1.0 / 12.0;
    // interior
    constexpr double alpha_xx  = 1.0 / 10.0;
    constexpr double a_xx      = 6.0 / 5.0;

    // boundary elements for P matrix for 2nd derivative
    std::vector<std::vector<double>> P2DiagBoundary{
        {alpha1_xx, 1.0, alpha1_xx}};

    // diagonal elements for P matrix for 2nd derivative.
    std::vector<double> P2DiagInterior{alpha_xx, 1.0, alpha_xx};
    // boundary elements for Q matrix for 2nd derivative
    std::vector<std::vector<double>> Q2DiagBoundary{
        {a1_xx, b1_xx, c1_xx, d1_xx, e1_xx}};
    // boundary elements for Q matrix for 2nd derivative
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
    // D_x coeffs
    // row 1
    const double alpha = 12.0 / 97.0;
    const double beta  = -1.0 / 194.0;
    const double a     = 120.0 / 97.0;

    // Q boundary terms, slot 1
    const double a1    = 177.0 / 16.0;
    const double b1    = -507.0 / 8.0;
    const double c1    = 783.0 / 8.0;
    const double d1    = -201.0 / 4.0;
    const double e1    = 81.0 / 16.0;
    const double f1    = -3.0 / 8.0;

    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{{11.0 / 2.0, -131.0 / 4.0},
                                                    {alpha, 1.0, alpha, beta}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{beta, alpha, 1.0, alpha, beta};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{{a1, b1, c1, d1, e1, f1}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{a / 2.0, 0.0, a / 2.0};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

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
