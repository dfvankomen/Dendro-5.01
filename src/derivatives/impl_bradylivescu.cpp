#include "derivatives/impl_bradylivescu.h"

#include "derivatives.h"
#include "derivatives/derivs_compact.h"

namespace dendroderivs {

void fill_alpha_beta_bl_6th(std::vector<std::vector<double>> &alpha,
                            std::vector<std::vector<double>> &beta,
                            unsigned int matrixID) {
    std::vector<double> w(4, 0.0);

    std::cout << "Attempting to fill alpha/beta" << std::endl;

    // fill alpha based on the imported generated code

    switch (matrixID) {
        case 1:
#include "generated/implicit_6th_schemes_nbs1.cpp"
            break;
        case 2:
#include "generated/implicit_6th_schemes_nbs2.cpp"
            break;
        case 3:
#include "generated/implicit_6th_schemes_nbs3.cpp"
            break;
        case 4:
#include "generated/implicit_6th_schemes_nbs4.cpp"
            break;
        case 5:
#include "generated/implicit_6th_schemes_nbs5.cpp"
            break;
        case 6:
#include "generated/implicit_6th_schemes_nbs6.cpp"
            break;
        case 7:
#include "generated/implicit_6th_schemes_nbs7.cpp"
            break;
        case 8:
#include "generated/implicit_6th_schemes_nbs8.cpp"
            break;
        case 9:
#include "generated/implicit_6th_schemes_nbs9.cpp"
            break;
        case 10:
#include "generated/implicit_6th_schemes_nbs10.cpp"
            break;
        case 11:
#include "generated/implicit_6th_schemes_nbs11.cpp"
            break;
        case 12:
#include "generated/implicit_6th_schemes_nbs12.cpp"
            break;
        case 13:
#include "generated/implicit_6th_schemes_nbs13.cpp"
            break;
        case 14:
#include "generated/implicit_6th_schemes_nbs14.cpp"
            break;
        case 15:
#include "generated/implicit_6th_schemes_nbs15.cpp"
            break;
        default:
            throw DendroDerivsNotImplemented(
                "The matrix number for BL's 6th order appears to be out of "
                "bounds! We can't "
                "initialize the matrix!");
            break;
    }
}

MatrixDiagonalEntries *createBL6thDiagonalsFirstOrder(unsigned int matrixID) {
    constexpr unsigned int NR = 4;
    constexpr unsigned int NT = 6;
    std::vector<std::vector<double>> alpha(NR, std::vector<double>(NT, 0.0));
    std::vector<std::vector<double>> beta(NR, std::vector<double>(3, 0.0));
    // fill alpha and beta
    fill_alpha_beta_bl_6th(alpha, beta, matrixID);

    constexpr double delta1 = 1.0 / 3.0;

    // boundary elements for P matrix for 1st derivative
    // P boundaries requires **4** rows!

    // std::vector<std::vector<double>> P1DiagBoundary{
    //     {1.0, beta[0][2]},
    //     {beta[1][0], 1.0, beta[1][2]},
    //     {0.0, beta[1][0], 1.0, beta[1][2]},
    //     {0.0, 0.0, beta[1][0], 1.0, beta[1][2]},
    //     {0.0, 0.0, 0.0, beta[1][0], 1.0, beta[1][2]}};

    // more robust way to fill the matrix, more like original code
    std::vector<std::vector<double>> P1DiagBoundary;
    P1DiagBoundary.push_back({1.0, beta[0][2]});

    for (unsigned int i = 1; i < NR; i++) {
        unsigned int n_cols = 2 + i;  // 3 + i - 1
        std::vector<double> temp(n_cols, 0.0);
        temp[i - 1] = beta[i][0];
        temp[i]     = 1.0;
        temp[i + 1] = beta[i][2];

        P1DiagBoundary.push_back(temp);
        std::cout << "temp: ";
        for (auto &x : temp) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{delta1, 1.0, delta1};

    // boundary elements for Q matrix for 1st derivative
    // NOTE: the boundaries are just alpha for this scheme!
    std::vector<std::vector<double>> Q1DiagBoundary = alpha;
    // diagonal elements for Q matrix for 1st derivative
    constexpr double gamma2                         = 1.0 / 36.0;
    constexpr double gamma1                         = 7.0 / 9.0;
    std::vector<double> Q1DiagInterior{-gamma2, -gamma1, 0.0, gamma1, gamma2};

    MatrixDiagonalEntries *diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

}  // namespace dendroderivs
