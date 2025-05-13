#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/filt_inmat.h"

namespace dendroderivs {

inline MatrixDiagonalEntries* create_BYUT6_filter_diagonals(
    const std::vector<double>& F_coeffs) {
    double alpha = F_coeffs[0];

    double a0    = (11.0 + 10.0 * alpha) / 16.0;
    double a1    = (15.0 + 34.0 * alpha) / 32.0;
    double a2    = (3.0 * (-1.0 + 2.0 * alpha)) / 16.0;
    double a3    = (1.0 - 2.0 * alpha) / 32.0;
    double a00   = (63.0 + alpha) / 64.0;
    double a01   = (3.0 + 29.0 * alpha) / 32.0;
    double a02   = (15.0 * (-1.0 + alpha)) / 64.0;
    double a03   = (-5.0 * (-1.0 + alpha)) / 16.0;
    double a04   = (15.0 * (-1.0 + alpha)) / 64.0;
    double a05   = (-3.0 * (-1.0 + alpha)) / 32.0;
    double a06   = (-1.0 + alpha) / 64.0;
    double a10   = (1.0 + 62.0 * alpha) / 64.0;
    double a11   = (29.0 + 6.0 * alpha) / 32.0;
    double a12   = (15.0 + 34.0 * alpha) / 64.0;
    double a13   = (5.0 * (-1.0 + 2.0 * alpha)) / 16.0;
    double a14   = (-15.0 * (-1.0 + 2.0 * alpha)) / 64.0;
    double a15   = (3.0 * (-1.0 + 2.0 * alpha)) / 32.0;
    double a16   = (1.0 - 2.0 * alpha) / 64.0;
    double a20   = (-1.0 + 2.0 * alpha) / 64.0;
    double a21   = (3.0 + 26.0 * alpha) / 32.0;
    double a22   = (49.0 + 30.0 * alpha) / 64.0;
    double a23   = (5.0 + 6.0 * alpha) / 16.0;
    double a24   = (15.0 * (-1.0 + 2.0 * alpha)) / 64.0;
    double a25   = (-3.0 * (-1.0 + 2.0 * alpha)) / 32.0;
    double a26   = (-1.0 + 2.0 * alpha) / 64.0;

    // diagonal elements for R matrix for 1st derivative
    std::vector<double> RDiagInterior{alpha, 1.0, alpha};
    // boundary elements for S matrix for 1st derivative
    std::vector<std::vector<double>> RDiagBoundary{
        {1, alpha, 0.0, 0.0}, {alpha, 1, alpha, 0.0}, {0.0, alpha, 1, alpha}};
    // boundary elements for S matrix for 1st derivative
    std::vector<std::vector<double>> SDiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06},
        {a10, a11, a12, a13, a14, a15, a16},
        {a20, a21, a22, a23, a24, a25, a26}};

    // diagonal elements for S matrix for 1st derivative
    std::vector<double> SDiagInterior{a3 / 2.0, a2 / 2.0, a1 / 2.0, a0,
                                      a1 / 2.0, a2 / 2.0, a3 / 2.0};

    return new MatrixDiagonalEntries{RDiagInterior, RDiagBoundary,
                                     SDiagInterior, SDiagBoundary};
}

class BYUT6Filter_InMatrix : public InMatrixFilter {
   public:
    BYUT6Filter_InMatrix(const std::vector<double>& input_coeffs)
        : InMatrixFilter(input_coeffs) {
        diagEntries = create_BYUT6_filter_diagonals(input_coeffs);
    }

    ~BYUT6Filter_InMatrix() = default;

    InMatFilterType get_filter_type() const override {
        return InMatFilterType::IMFT_BYUT6;
    }
};

}  // namespace dendroderivs
