
#include "derivatives/impl_explicitmatrix.h"

#include "derivatives/derivs_compact.h"

namespace dendroderivs {

MatrixDiagonalEntries* createE4DiagonalsFirstOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative

    std::vector<std::vector<double>> P1DiagBoundary{{1.0, 0.0}, {0.0, 1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};

    // boundary elements for Q matrix for 1st derivative
    // NOTE: this is if we use up to four points
    // constexpr double d_b = 2.0;
    // std::vector<std::vector<double>> Q1DiagBoundary{
    //     {-3.0 / d_b, 4.0 / d_b, -1.0 / d_b},
    //     {-1.0 / d_b, 0.0 / d_b, 1.0 / d_b}};

    // NOTE: using up to 5 points to match the interior stencil
    constexpr double d_b = 12.0;
    std::vector<std::vector<double>> Q1DiagBoundary{
        {-25.0 / d_b, 48.0 / d_b, -36.0 / d_b, 16.0 / d_b, -3.0 / d_b},
        {-3.0 / d_b, -10.0 / d_b, 18.0 / d_b, -6.0 / d_b, 1.0 / d_b}};

    // diagonal elements for Q matrix for 1st derivative
    constexpr double d = 12.0;
    std::vector<double> Q1DiagInterior{1.0 / d, -8.0 / d, 0.0, 8.0 / d,
                                       -1.0 / d};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE4DiagonalsSecondOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{{1.0, 0.0}, {0.0, 1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};

    // boundary elements for Q matrix for 1st derivative
    // NOTE: this is if we use up to four points
    // constexpr double d_b = 1.0;
    // std::vector<std::vector<double>> Q1DiagBoundary{
    //     {2.0 / d_b, -5.0 / d_b, 4.0 / d_b, -1.0 / d_b},
    //     {1.0 / d_b, -2.0 / d_b, 1.0 / d_b, 0.0 / d_b}};

    // NOTE: using up to 5 points to match the interior stencil
    constexpr double d_b = 12.0;
    std::vector<std::vector<double>> Q1DiagBoundary{
        {35.0 / d_b, -104.0 / d_b, 114.0 / d_b, -56.0 / d_b, 11.0 / d_b},
        {11.0 / d_b, -20.0 / d_b, 6.0 / d_b, 4.0 / d_b, -1.0 / d_b}};

    // diagonal elements for Q matrix for 1st derivative
    constexpr double d = 12.0;
    std::vector<double> Q1DiagInterior{-1.0 / d, 16.0 / d, -30.0 / d, 16.0 / d,
                                       -1.0 / d};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE6DiagonalsFirstOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};

    // boundary elements for Q matrix for 1st derivative
    // NOTE: this is with 5 points
    // constexpr double d_b = 12.0;
    // std::vector<std::vector<double>> Q1DiagBoundary{
    //     {-25.0 / d_b, 48.0 / d_b, -36.0 / d_b, 16.0 / d_b, -3.0 / d_b},
    //     {-3.0 / d_b, -10.0 / d_b, 18.0 / d_b, -6.0 / d_b, 1.0 / d_b},
    //     {1.0 / d_b, -8.0 / d_b, 0.0, 8.0 / d_b, -1.0 / d_b}};

    // NOTE: this is with 7 points, to match the interior points
    constexpr double d_b = 60.0;
    std::vector<std::vector<double>> Q1DiagBoundary{
        {-147.0 / d_b, 360.0 / d_b, -450.0 / d_b, 400.0 / d_b, -225.0 / d_b,
         72.0 / d_b, -10.0 / d_b},
        {-10.0 / d_b, -77.0 / d_b, 150.0 / d_b, -100.0 / d_b, 50.0 / d_b,
         -15.0 / d_b, 2.0 / d_b},
        {2.0 / d_b, -24.0 / d_b, -35.0 / d_b, 80.0 / d_b, -30.0 / d_b,
         8.0 / d_b, -1.0 / d_b}};

    // diagonal elements for Q matrix for 1st derivative
    constexpr double d = 60.0;
    std::vector<double> Q1DiagInterior{-1.0 / d, 9.0 / d,  -45.0 / d, 0.0,
                                       45.0 / d, -9.0 / d, 1.0 / d};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE6DiagonalsSecondOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0}, {0.0, 1.0}, {0.0, 0.0, 1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};

    // boundary elements for Q matrix for 1st derivative
    // NOTE:: these are for 6 points
    constexpr double d_b = 12.0;
    std::vector<std::vector<double>> Q1DiagBoundary{
        {45.0 / d_b, -154.0 / d_b, 214.0 / d_b, -156.0 / d_b, 61.0 / d_b,
         -10.0 / d_b},
        {10.0 / d_b, -15.0 / d_b, -4.0 / d_b, 14.0 / d_b, -6.0 / d_b,
         1.0 / d_b},
        {-1.0 / d_b, 16.0 / d_b, -30.0 / d_b, 16.0 / d_b, -1.0 / d_b}};

    // NOTE: these are for 8 points, this might be too many
    // constexpr double d_b = 180.0;
    // std::vector<std::vector<double>> Q1DiagBoundary{
    //     {938.0 / d_b, -4014.0 / d_b, 7911.0 / d_b, -9490.0 / d_b, 7380.0 /
    //     d_b,
    //      -3618.0 / d_b, 1019.0 / d_b, -126.0 / d_b},
    //     {
    //         126.0 / d_b,
    //         -70.0 / d_b,
    //         -486.0 / d_b,
    //         855.0 / d_b,
    //         -670.0 / d_b,
    //         324.0 / d_b,
    //         -90.0 / d_b,
    //         11.0 / d_b,
    //     },
    //     {
    //         -11.0 / d_b,
    //         214.0 / d_b,
    //         -378.0 / d_b,
    //         130.0 / d_b,
    //         85.0 / d_b,
    //         -54.0 / d_b,
    //         16.0 / d_b,
    //         -2.0 / d_b,
    //     }};

    // diagonal elements for Q matrix for 2nd derivative
    constexpr double d = 180.0;
    std::vector<double> Q1DiagInterior{2.0 / d,    -27.0 / d, 270.0 / d,
                                       -490.0 / d, 270.0 / d, -27.0 / d,
                                       2.0 / d};
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE8DiagonalsFirstOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};

    // boundary elements for Q matrix for 1st derivative
    // NOTE: this is with 7 points
    constexpr double d_b = 60.0;
    std::vector<std::vector<double>> Q1DiagBoundary{
        {-147.0 / d_b, 360.0 / d_b, -450.0 / d_b, 400.0 / d_b, -225.0 / d_b,
         72.0 / d_b, -10.0 / d_b},
        {-10.0 / d_b, -77.0 / d_b, 150.0 / d_b, -100.0 / d_b, 50.0 / d_b,
         -15.0 / d_b, 2.0 / d_b},
        {2.0 / d_b, -24.0 / d_b, -35.0 / d_b, 80.0 / d_b, -30.0 / d_b,
         8.0 / d_b, -1.0 / d_b},
        {-1.0 / d_b, 9.0 / d_b, -45.0 / d_b, 0.0, 45.0 / d_b, -9.0 / d_b,
         1.0 / d_b}};

    // diagonal elements for Q matrix for 1st derivative
    constexpr double d = 2520.0;
    std::vector<double> Q1DiagInterior{9.0 / d,     -96.0 / d, 504.0 / d,
                                       -2016.0 / d, 0.0 / d,   2016.0 / d,
                                       -504.0 / d,  96.0 / d,  -9.0 / d};

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* createE8DiagonalsSecondOrder() {
    // D_x coeffs
    // row 1
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0}, {0.0, 1.0}, {0.0, 0.0, 1.0}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{1.0};

    // boundary elements for Q matrix for 1st derivative
    // NOTE: these are for 8 points
    constexpr double d_b = 180.0;
    std::vector<std::vector<double>> Q1DiagBoundary{
        {938.0 / d_b, -4014.0 / d_b, 7911.0 / d_b, -9490.0 / d_b, 7380.0 / d_b,
         -3618.0 / d_b, 1019.0 / d_b, -126.0 / d_b},
        {126.0 / d_b, -70.0 / d_b, -486.0 / d_b, 855.0 / d_b, -670.0 / d_b,
         324.0 / d_b, -90.0 / d_b, 11.0 / d_b},
        {-11.0 / d_b, 214.0 / d_b, -378.0 / d_b, 130.0 / d_b, 85.0 / d_b,
         -54.0 / d_b, 16.0 / d_b, -2.0 / d_b},
        {2.0 / d_b, -27.0 / d_b, 270.0 / d_b, -490.0 / d_b, 270.0 / d_b,
         -27.0 / d_b, 2.0 / d_b}

    };

    // diagonal elements for Q matrix for 2nd derivative
    constexpr double d = 5040.0;
    std::vector<double> Q1DiagInterior{-9.0 / d,    128.0 / d,    -1008.0 / d,
                                       8064.0 / d,  -14350.0 / d, 8064.0 / d,
                                       -1008.0 / d, 128.0 / d,    -9.0 / d};
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

}  // namespace dendroderivs
