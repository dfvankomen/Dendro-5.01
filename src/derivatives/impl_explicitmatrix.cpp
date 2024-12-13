
#include "derivatives/impl_explicitmatrix.h"

#include "derivatives/derivs_compact.h"

namespace dendroderivs {

MatrixDiagonalEntries* createE4DiagonalsFirstOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{{1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{{-3.0, 4.0, -1.0},
                                                    {-1.0, 0.0, 5.0}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0,
                                       -1.0 / 12.0};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE4DiagonalsSecondOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{{1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{{2.0, -5.0, 4.0, -1.0},
                                                    {1.0, -2.0, 1.0, 0.0}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{-1.0 / 12.0, 16.0 / 12.0, -30.0 / 12.0,
                                       16.0 / 12.0, -1.0 / 12.0};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE6DiagonalsFirstOrder() {
    constexpr double d   = 60.0;
    constexpr double d_b = 12.0;
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{{1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {-25.0 / d_b, 48.0 / d_b, -36.0 / d_b, 16.0 / d_b, -3.0 / d_b},
        {-3.0 / d_b, -10.0 / d_b, 18.0 / d_b, -6.0 / d_b, 1.0 / d_b},
        {1.0 / d_b, -8.0 / d_b, 0.0, 8.0 / d_b, -1.0 / d_b}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{-1.0 / d, 9.0 / d,  -45.0 / d, 0.0,
                                       45.0 / d, -9.0 / d, 1.0 / d};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE6DiagonalsSecondOrder() {
    constexpr double d   = 180.0;
    constexpr double d_b = 12.0;
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{{1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {45.0 / d_b, -154.0 / d_b, 214.0 / d_b, -156.0 / d_b, 61.0 / d_b,
         -10.0 / d_b},
        {10.0 / d_b, -15.0 / d_b, -4.0 / d_b, 14.0 / d_b, -6.0 / d_b,
         1.0 / d_b},
        {-1.0 / d_b, 16.0 / d_b, -30.0 / d_b, 16.0 / d_b, -1.0 / d_b}};
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{2.0 / d,    -27.0 / d, 270.0 / d,
                                       -490.0 / d, 270.0 / d, -27.0 / d,
                                       2.0 / d};
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

}  // namespace dendroderivs
