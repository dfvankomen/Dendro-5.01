#include "derivatives/impl_boris.h"

#include <stdexcept>

#include "derivatives.h"
#include "derivatives/derivs_compact.h"

namespace dendroderivs {

MatrixDiagonalEntries *createBoris4thDiagonalsFirstOrder(
    unsigned int boundary_type, unsigned int pw) {
    std::vector<std::vector<double>> P1DiagBoundary;
    std::vector<std::vector<double>> Q1DiagBoundary;

    if (boundary_type == 1) {
        // DIRICHLET boundary order 4
        P1DiagBoundary = {{1.0, 3.0}};

        Q1DiagBoundary = {{-17.0 / 6.0, 3.0 / 2.0, 3.0 / 2.0, -1.0 / 6.0}};
    } else if (boundary_type == 2) {
        // O4_L4_CLOSURE
        // fill in the ghosts with the closure type:
        for (unsigned int i = 0; i < pw; i++) {
            std::vector<double> temp(i + 1, 0.0);
            temp[i] = 1.0;
            P1DiagBoundary.push_back(temp);
            Q1DiagBoundary.push_back(temp);
        }

        // then build up the next vector
        std::vector<double> temp(pw, 0.0);
        std::vector<double> boundRow1{1.0};

        // row 1
        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        P1DiagBoundary.push_back(temp);

        // row 1
        constexpr double t3 = 1.0 / 12.0;
        temp                = std::vector<double>(pw - 1, 0.0);
        boundRow1           = {-3.0 * t3, -10.0 * t3, 18 * t3, -6.0 * t3, t3};

        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        Q1DiagBoundary.push_back(temp);
    } else if (boundary_type == 3) {
        // P1_O4_CLOSURE
        if (pw < 3) {
            throw std::invalid_argument(
                "Not enough padding points for P6_04 Closure!");
        }

        for (unsigned int i = 0; i < pw; i++) {
            std::vector<double> temp(i + 1, 0.0);
            temp[i] = 1.0;
            P1DiagBoundary.push_back(temp);
            Q1DiagBoundary.push_back(temp);
        }

        // then build up the next vector
        std::vector<double> temp(pw, 0.0);
        std::vector<double> boundRow1{1.0};

        // row 1
        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        P1DiagBoundary.push_back(temp);

        // row 1
        constexpr double t1 = 1.0 / 72.0;
        // offset by 3 back
        temp                = std::vector<double>(pw - 3, 0.0);
        boundRow1           = {-t1,       10.0 * t1,  -53.0 * t1, 0.0,
                               53.0 * t1, -10.0 * t1, t1};

        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        Q1DiagBoundary.push_back(temp);
    } else {
        throw std::invalid_argument(
            "Invalid boundary_type (check MatrixID value input in "
            "constructor!)");
    }

    // -0.75, 0, 0.75
    std::vector<double> P1DiagInterior{1.0 / 4.0, 1.0, 1.0 / 4.0};
    std::vector<double> Q1DiagInterior{-3.0 / 4.0, 0.0, 3.0 / 4.0};

    MatrixDiagonalEntries *diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    // set the Q1DiagBoundaryLower padding rows to -1 to avoid the "parity"
    // issue if using closure
    if (boundary_type > 1) {
        for (unsigned int i = 0; i < pw; i++) {
            diagEntries->QDiagBoundaryLower[i][i] = -1.0;
        }
    }

    return diagEntries;
}

MatrixDiagonalEntries *createBoris6thDiagonalsFirstOrder(
    unsigned int boundary_type, unsigned int pw) {
    std::vector<std::vector<double>> P1DiagBoundary;
    std::vector<std::vector<double>> Q1DiagBoundary;

    if (boundary_type == 1) {
        // DIRICHLET boundary order 6
        P1DiagBoundary = {{1.0, 5.0}, {2.0 / 11.0, 1.0, 2.0 / 11.0}};

        Q1DiagBoundary = {{-197.0 / 60.0, -5.0 / 12.0, 5.0, -5.0 / 3.0,
                           5.0 / 12.0, -1.0 / 20.0},

                          {-20.0 / 33.0, -35.0 / 132.0, 34.0 / 33.0,
                           -7.0 / 33.0, 2.0 / 33.0, -1.0 / 132.0}};
    } else if (boundary_type == 2) {
        // O6_L4_CLOSURE
        // fill in the ghosts 1's'
        for (unsigned int i = 0; i < pw; i++) {
            std::vector<double> temp(i + 1, 0.0);
            temp[i] = 1.0;
            P1DiagBoundary.push_back(temp);
            Q1DiagBoundary.push_back(temp);
        }

        // then build up the next vector
        std::vector<double> temp(pw, 0.0);
        std::vector<double> boundRow1{1.0};

        // row 1
        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        P1DiagBoundary.push_back(temp);

        // row 1
        constexpr double t4 = 1.0 / 60.0;
        temp                = std::vector<double>(pw - 2, 0.0);
        boundRow1           = {2.0 * t4,   -24.0 * t4, -35.0 * t4, 80.0 * t4,
                               -30.0 * t4, 8.0 * t4,   -1.0 * t4};

        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        Q1DiagBoundary.push_back(temp);
    } else if (boundary_type == 3) {
        // P1_O6_CLOSURE
        if (pw < 4) {
            throw std::invalid_argument(
                "Not enough padding points for P6_06 Closure!");
        }

        for (unsigned int i = 0; i < pw; i++) {
            std::vector<double> temp(i + 1, 0.0);
            temp[i] = 1.0;
            P1DiagBoundary.push_back(temp);
            Q1DiagBoundary.push_back(temp);
        }

        // then build up the next vector
        std::vector<double> temp(pw, 0.0);
        std::vector<double> boundRow1{1.0};

        // row 1
        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        P1DiagBoundary.push_back(temp);

        // row 1
        constexpr double t2 = 1.0 / 300.0;
        // offset by 4 back
        temp                = std::vector<double>(pw - 4, 0.0);
        boundRow1 = {t2,         -11.0 * t2, 59.0 * t2, -239.0 * t2, 0.0,
                     239.0 * t2, -59.0 * t2, 11.0 * t2, -t2};

        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        Q1DiagBoundary.push_back(temp);
    } else {
        throw std::invalid_argument(
            "Invalid boundary_type (check MatrixID value input in "
            "constructor!)");
    }

    // TODO:
    std::vector<double> P1DiagInterior{1.0 / 3.0, 1.0, 1.0 / 3.0};
    constexpr double t1 = 1.0 / 36.0;
    std::vector<double> Q1DiagInterior{-t1, -28.0 * t1, 0.0, 28.0 * t1, t1};

    MatrixDiagonalEntries *diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    // set the Q1DiagBoundaryLower first row to -1 to avoid the "parity"
    // issue
    if (boundary_type > 1) {
        for (unsigned int i = 0; i < pw; i++) {
            diagEntries->QDiagBoundaryLower[i][i] = -1.0;
        }
    }

    return diagEntries;
}

MatrixDiagonalEntries *createBoris6thEtaDiagonalsFirstOrder(
    unsigned int boundary_type, unsigned int pw) {
    std::vector<std::vector<double>> P1DiagBoundary;
    std::vector<std::vector<double>> Q1DiagBoundary;

    if (boundary_type == 1) {
        // DIRICHLET boundary order 6
        P1DiagBoundary = {{1.0, 5.0}, {2.0 / 11.0, 1.0, 2.0 / 11.0}};

        Q1DiagBoundary = {{-197.0 / 60.0, -5.0 / 12.0, 5.0, -5.0 / 3.0,
                           5.0 / 12.0, -1.0 / 20.0},

                          {-20.0 / 33.0, -35.0 / 132.0, 34.0 / 33.0,
                           -7.0 / 33.0, 2.0 / 33.0, -1.0 / 132.0}};
    } else if (boundary_type == 2) {
        // O6_L4_CLOSURE
        // fill in the ghosts 1's'
        for (unsigned int i = 0; i < pw; i++) {
            std::vector<double> temp(i + 1, 0.0);
            temp[i] = 1.0;
            P1DiagBoundary.push_back(temp);
            Q1DiagBoundary.push_back(temp);
        }

        // then build up the next vector
        std::vector<double> temp(pw, 0.0);
        std::vector<double> boundRow1{1.0};

        // row 1
        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        P1DiagBoundary.push_back(temp);

        // row 1
        constexpr double t4 = 1.0 / 60.0;
        temp                = std::vector<double>(pw - 2, 0.0);
        boundRow1           = {2.0 * t4,   -24.0 * t4, -35.0 * t4, 80.0 * t4,
                               -30.0 * t4, 8.0 * t4,   -1.0 * t4};

        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        Q1DiagBoundary.push_back(temp);
    } else if (boundary_type == 3) {
        // P1_O6__ETA_CLOSURE
        if (pw < 4) {
            throw std::invalid_argument(
                "Not enough padding points for P6_06_ETA Closure!");
        }

        for (unsigned int i = 0; i < pw; i++) {
            std::vector<double> temp(i + 1, 0.0);
            temp[i] = 1.0;
            P1DiagBoundary.push_back(temp);
            Q1DiagBoundary.push_back(temp);
        }

        // then build up the next vector
        std::vector<double> temp(pw, 0.0);
        std::vector<double> boundRow1{1.0};

        // row 1
        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        P1DiagBoundary.push_back(temp);

        // row 1
        constexpr double t2 = 1.0 / 300.0;
        // offset by 4 back
        temp                = std::vector<double>(pw - 4, 0.0);
        boundRow1 = {0.0035978349, -0.038253676, 0.20036969,  -0.80036969,  0.0,
                     0.80036969,   -0.20036969,  0.038253676, -0.0035978349};

        temp.insert(temp.end(), boundRow1.begin(), boundRow1.end());
        Q1DiagBoundary.push_back(temp);
    } else {
        throw std::invalid_argument(
            "Invalid boundary_type (check MatrixID value input in "
            "constructor!)");
    }

    // TODO:
    std::vector<double> P1DiagInterior{0.37987923, 1.0, 0.37987923};
    std::vector<double> Q1DiagInterior{0.0023272948, -0.052602255, -0.78165660,
                                       0.0,          0.78165660,   0.052602255,
                                       -0.0023272948};

    MatrixDiagonalEntries *diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    // set the Q1DiagBoundaryLower first row to -1 to avoid the "parity"
    // issue
    if (boundary_type > 1) {
        for (unsigned int i = 0; i < pw; i++) {
            diagEntries->QDiagBoundaryLower[i][i] = -1.0;
        }
    }

    return diagEntries;
}

}  // namespace dendroderivs
