
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

}  // namespace dendroderivs
