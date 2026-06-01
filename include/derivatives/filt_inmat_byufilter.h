#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/filt_inmat.h"

namespace dendroderivs {

inline MatrixDiagonalEntries* create_BYUT4_filter_diagonals(
    const std::vector<double>& F_coeffs) {
    double alpha = F_coeffs[0];

    double a0  = (5.0 + 6.0 * alpha) / 8.0;
    double a1  = (1.0 + 2.0 * alpha) / 2.0;
    double a2  = (-1.0 + 2.0 * alpha) / 8.0;

    double a00 = (15.0 + alpha) / 16.0;
    double a01 = (1.0 + 3.0 * alpha) / 4.0;
    double a02 = (3.0 * (-1.0 + alpha)) / 8.0;
    double a03 = (1.0 - alpha) / 4.0;
    double a04 = (-1.0 + alpha) / 16.0;

    double a10 = (1.0 + 14.0 * alpha) / 16.0;
    double a11 = (3.0 + 2.0 * alpha) / 4.0;
    double a12 = (3.0 + 2.0 * alpha) / 8.0;
    double a13 = (-1.0 + 2.0 * alpha) / 4.0;
    double a14 = (1.0 - 2.0 * alpha) / 16.0;

    // P (your R) matrix: tridiagonal interior, simple boundary rows
    std::vector<double> RDiagInterior{alpha, 1.0, alpha};
    std::vector<std::vector<double>> RDiagBoundary{
        {1.0, alpha, 0.0},     // i = 0
        {alpha, 1.0, alpha},   // i = 1
        {0.0, alpha, 1.0}      // i = 2  (keeps shape consistent with BYUT6/8)
    };

    // Q (your S) matrix: 5-wide interior
    std::vector<double> SDiagInterior{a2 / 2.0, a1 / 2.0, a0, a1 / 2.0, a2 / 2.0};
    std::vector<std::vector<double>> SDiagBoundary{
        {a00, a01, a02, a03, a04},  // i = 0
        {a10, a11, a12, a13, a14}   // i = 1
        // (5-wide usually needs 2 boundary rows; add a 3rd if your factory expects it)
    };

    // ctor order: P-int, P-bdry, Q-int, Q-bdry
    return new MatrixDiagonalEntries{
        RDiagInterior, RDiagBoundary, SDiagInterior, SDiagBoundary
    };
}

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
inline MatrixDiagonalEntries* create_BYUT8_filter_diagonals(
	const std::vector<double>& F_coeffs) {
		
		double alpha = F_coeffs[0];

		double a0 = (93.0 + 70.0 * alpha) / 128.0;
		double a1 = (7.0 + 18.0 * alpha) / 16.0;
		double a2 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
		double a3 = (1.0 - 2.0 * alpha) / 16.0;
		double a4 = ( - 1.0 + 2.0 * alpha) / 128.0;
		double a00 = (255.0 + alpha) / 256.0;
		double a01 = (1.0 + 31.0 * alpha) / 32.0;
		double a02 = (7.0 * ( - 1.0 + alpha)) / 64.0;
		double a03 = ( - 7.0 * ( - 1.0 + alpha)) / 32.0;
		double a04 = (35.0 * ( - 1.0 + alpha)) / 128.0;
		double a05 = ( - 7.0 * ( - 1.0 + alpha)) / 32.0;
		double a06 = (7.0 * ( - 1.0 + alpha)) / 64.0;
		double a07 = (1.0 - alpha) / 32.0;
		double a08 = ( - 1.0 + alpha) / 256.0;
		double a10 = (1.0 + 254.0 * alpha) / 256.0;
		double a11 = (31.0 + 2.0 * alpha) / 32.0;
		double a12 = (7.0 + 50.0 * alpha) / 64.0;
		double a13 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
		double a14 = ( - 35.0 * ( - 1.0 + 2.0 * alpha)) / 128.0;
		double a15 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
		double a16 = ( - 7.0 * ( - 1.0 + 2.0 * alpha)) / 64.0;
		double a17 = ( - 1.0 + 2.0 * alpha) / 32.0;
		double a18 = (1.0 - 2.0 * alpha) / 256.0;
		double a20 = ( - 1.0 + 2.0 * alpha) / 256.0;
		double a21 = (1.0 + 30.0 * alpha) / 32.0;
		double a22 = (57.0 + 14.0 * alpha) / 64.0;
		double a23 = (7.0 + 18.0 * alpha) / 32.0;
		double a24 = (35.0 * ( - 1.0 + 2.0 * alpha)) / 128.0;
		double a25 = ( - 7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
		double a26 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 64.0;
		double a27 = (1.0 - 2.0 * alpha) / 32.0;
		double a28 = ( - 1.0 + 2.0 * alpha) / 256.0;

		// diagonal elements for R matrix for 1st derivative
		std::vector<double> RDiagInterior{
			alpha, 1.0, alpha
		};
            std::vector<std::vector<double>> RDiagBoundary{
        {1, alpha, 0.0, 0.0}, {alpha, 1, alpha, 0.0}, {0.0, alpha, 1, alpha}};

		// boundary elements for S matrix for 1st derivative
		std::vector<std::vector<double>> SDiagBoundary
			{{a00, a01, a02, a03, a04, a05, a06, a07, a08}, {a10, a11, a12, a13, a14, a15, a16, a17, a18}, {a20, a21, a22, a23, a24, a25, a26, a27, a28}}
		;

		// diagonal elements for S matrix for 1st derivative
		std::vector<double> SDiagInterior{
			a4/2.0, a3/2.0, a2/2.0, a1/2.0, a0, a1/2.0, a2/2.0, a3/2.0, a4/2.0
		};

		// store the entries for matrix creation
    return new MatrixDiagonalEntries{RDiagInterior, RDiagBoundary,
                                     SDiagInterior, SDiagBoundary};
	}
class BYUT4Filter_InMatrix : public InMatrixFilter {
public:
    explicit BYUT4Filter_InMatrix(const std::vector<double>& input_coeffs)
        : InMatrixFilter(input_coeffs) {
        diagEntries = create_BYUT4_filter_diagonals(input_coeffs);
    }

    BYUT4Filter_InMatrix(std::initializer_list<double> coeffs)
        : BYUT4Filter_InMatrix(std::vector<double>(coeffs)) {}

    ~BYUT4Filter_InMatrix() = default;  // <- remove 'override'

    InMatFilterType get_filter_type() const override {
        return InMatFilterType::IMFT_BYUT4;
    }
};

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

    class BYUT8Filter_InMatrix : public InMatrixFilter {
   public:
    BYUT8Filter_InMatrix(const std::vector<double>& input_coeffs)
        : InMatrixFilter(input_coeffs) {
        diagEntries = create_BYUT8_filter_diagonals(input_coeffs);
    }

    ~BYUT8Filter_InMatrix() = default;

    InMatFilterType get_filter_type() const override {
        return InMatFilterType::IMFT_BYUT8;
    }
};
}  // namespace dendroderivs
