#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/filt_inmat.h"

namespace dendroderivs {

inline void kim_filter_cal_coeff(double* c, double kc) {
    double AF = 30.0 - 5.0 * cos(kc) + 10 * cos(2.0 * kc) - 3.0 * cos(3.0 * kc);
    double alphaF = -(30.0 * cos(kc) + 2.0 * cos(3.0 * kc)) / AF;
    double betaF =
        (18.0 + 9.0 * cos(kc) + 6.0 * cos(2.0 * kc) - cos(3.0 * kc)) /
        (2.0 * AF);
    c[0] = AF;
    c[1] = alphaF;
    c[2] = betaF;
}

inline MatrixDiagonalEntries* create_Kim_filter_diagonals(
    const std::vector<double>& F_coeffs) {
    double pi  = 3.141592654;
    double kc  = 0.50 * pi;
    double eps = 0.25;
    double c0[3];
    double cd[3];
    double cdd[3];
    double cddd[3];

    double t2    = sin(M_PI / 2.0);
    double t3    = sin(M_PI / 3.0);
    double t6    = sin(M_PI / 6.0);

    double kcd   = kc * (1.0 - eps * t6 * t6);
    double kcdd  = kc * (1.0 - eps * t3 * t3);
    double kcddd = kc * (1.0 - eps * t2 * t2);

    kim_filter_cal_coeff(c0, kc);
    kim_filter_cal_coeff(cd, kcd);
    kim_filter_cal_coeff(cdd, kcdd);
    kim_filter_cal_coeff(cddd, kcddd);

    const double t1        = cos(0.5 * kc);
    const double aF1       = 30.0 * t1 * t1 * t1 * t1 / c0[0];
    const double aF2       = -2.0 * aF1 / 5.0;
    const double aF3       = aF1 / 15.0;
    const double aF0       = -2.0 * (aF1 + aF2 + aF3);

    const double alphaF    = c0[1];
    const double betaF     = c0[2];

    const double alphaFd   = cd[1];
    const double betaFd    = cd[2];
    const double alphaFdd  = cdd[1];
    const double betaFdd   = cdd[2];
    const double alphaFddd = cddd[1];
    const double betaFddd  = cddd[2];

    const double t1d       = cos(0.5 * kcd);
    const double aF1d      = 30.0 * t1d * t1d * t1d * t1d / cd[0];
    const double aF2d      = -2.0 * aF1d / 5.0;
    const double aF3d      = aF1d / 15.0;

    const double BF =
        (1.0 - betaFdd) * (1.0 + 6.0 * betaFdd + 60.0 * betaFdd * betaFdd) +
        (5.0 + 35 * betaFdd - 29.0 * betaFdd * betaFdd) * alphaFdd +
        (9.0 - 5.0 * betaFdd) * alphaFdd * alphaFdd;
    const double CF =
        1.0 + betaFddd * (5.0 + 4.0 * betaFddd + 60.0 * betaFddd * betaFddd) +
        5.0 * (1.0 + 3.0 * betaFddd + 10.0 * betaFddd * betaFddd) * alphaFddd +
        2.0 * (4.0 + 11.0 * betaFddd) * alphaFddd * alphaFddd +
        5.0 * alphaFddd * alphaFddd * alphaFddd;

    const double yF00 = 0.0;
    const double yF10 =
        (10.0 * betaFdd * betaFdd * (8.0 * betaFdd - 1.0) +
         (1.0 + 4.0 * betaFdd + 81.0 * betaFdd * betaFdd) * alphaFdd +
         5.0 * (1.0 + 8.0 * betaFdd) * alphaFdd * alphaFdd +
         9.0 * alphaFdd * alphaFdd * alphaFdd) /
        BF;
    const double yF20 = betaFd;
    const double yF01 =
        (alphaFddd * (1.0 + alphaFddd) * (1.0 + 4.0 * alphaFddd) +
         2.0 * alphaFddd * (7.0 + 3.0 * alphaFddd) * betaFddd +
         24.0 * (1.0 - alphaFddd) * betaFddd * betaFddd -
         80.0 * betaFddd * betaFddd * betaFddd) /
        CF;
    const double yF11 = 0.0;
    const double yF21 = alphaFd;
    const double yF02 =
        (alphaFddd * alphaFddd * alphaFddd +
         (1.0 + 3.0 * alphaFddd + 14.0 * alphaFddd * alphaFddd) * betaFddd +
         46.0 * alphaFddd * betaFddd * betaFddd +
         60.0 * betaFddd * betaFddd * betaFddd) /
        CF;
    const double yF12 =
        (alphaFdd * (1.0 + 5.0 * alphaFdd + 9.0 * alphaFdd * alphaFdd) +
         alphaFdd * (5.0 + 36.0 * alphaFdd) * betaFdd +
         (55.0 * alphaFdd - 1.0) * betaFdd * betaFdd +
         10.0 * betaFdd * betaFdd * betaFdd) /
        BF;
    const double yF22 = 0.0;
    const double yF03 = 0.0;
    const double yF13 =
        betaFdd *
        (1.0 + 5.0 * alphaFdd + 9.0 * alphaFdd * alphaFdd +
         5.0 * (1.0 + 7.0 * alphaFdd) * betaFdd + 50.0 * betaFdd * betaFdd) /
        BF;
    const double yF23 = alphaFd;
    const double yF04 = 0.0;
    const double yF14 = 0.0;
    const double yF24 = betaFd;

    const double bF20 = aF2d + 5.0 * aF3d;
    const double bF21 = aF1d - 10.0 * aF3d;
    const double bF23 = aF1d - 5.0 * aF3d;
    const double bF24 = aF2d + aF3d;
    const double bF25 = aF3d;
    const double bF22 = -(bF20 + bF21 + bF23 + bF24 + bF25);

    // diagonal elements for R matrix for 1st derivative
    std::vector<double> RDiagInterior{betaF, alphaF, 1.0, alphaF, betaF};
    std::vector<std::vector<double>> RDiagBoundary = {
        {1.0, yF01, yF02},
        {yF10, 1.0, yF12, yF13},
        {yF20, yF21, 1.0, yF23, yF24}};

    std::vector<double> SDiagInterior{
        aF3, aF2, aF1, aF0, aF1, aF2, aF3,
    };
    // boundary elements for S matrix for 1st derivative
    std::vector<std::vector<double>> SDiagBoundary{
        {0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0},
        {bF20, bF21, bF22, bF23, bF24, bF25}};
    // MatrixDiagonalEntries
    //  MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsFirstOrderFiltered(
    //  const std::vector<double>& F_coeffs) {

    // 	double alpha = F_coeffs[0];

    // 	double a0 = (93.0 + 70.0 * alpha) / 128.0;
    // 	double a1 = (7.0 + 18.0 * alpha) / 16.0;
    // 	double a2 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
    // 	double a3 = (1.0 - 2.0 * alpha) / 16.0;
    // 	double a4 = ( - 1.0 + 2.0 * alpha) / 128.0;
    // 	double a00 = (255.0 + alpha) / 256.0;
    // 	double a01 = (1.0 + 31.0 * alpha) / 32.0;
    // 	double a02 = (7.0 * ( - 1.0 + alpha)) / 64.0;
    // 	double a03 = ( - 7.0 * ( - 1.0 + alpha)) / 32.0;
    // 	double a04 = (35.0 * ( - 1.0 + alpha)) / 128.0;
    // 	double a05 = ( - 7.0 * ( - 1.0 + alpha)) / 32.0;
    // 	double a06 = (7.0 * ( - 1.0 + alpha)) / 64.0;
    // 	double a07 = (1.0 - alpha) / 32.0;
    // 	double a08 = ( - 1.0 + alpha) / 256.0;
    // 	double a10 = (1.0 + 254.0 * alpha) / 256.0;
    // 	double a11 = (31.0 + 2.0 * alpha) / 32.0;
    // 	double a12 = (7.0 + 50.0 * alpha) / 64.0;
    // 	double a13 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
    // 	double a14 = ( - 35.0 * ( - 1.0 + 2.0 * alpha)) / 128.0;
    // 	double a15 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
    // 	double a16 = ( - 7.0 * ( - 1.0 + 2.0 * alpha)) / 64.0;
    // 	double a17 = ( - 1.0 + 2.0 * alpha) / 32.0;
    // 	double a18 = (1.0 - 2.0 * alpha) / 256.0;
    // 	double a20 = ( - 1.0 + 2.0 * alpha) / 256.0;
    // 	double a21 = (1.0 + 30.0 * alpha) / 32.0;
    // 	double a22 = (57.0 + 14.0 * alpha) / 64.0;
    // 	double a23 = (7.0 + 18.0 * alpha) / 32.0;
    // 	double a24 = (35.0 * ( - 1.0 + 2.0 * alpha)) / 128.0;
    // 	double a25 = ( - 7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
    // 	double a26 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 64.0;
    // 	double a27 = (1.0 - 2.0 * alpha) / 32.0;
    // 	double a28 = ( - 1.0 + 2.0 * alpha) / 256.0;
    // 	double a30 = (1.0 - 2.0 * alpha) / 256.0;
    // 	double a31 = ( - 1.0 + 2.0 * alpha) / 32.0;
    // 	double a32 = (7.0 + 50.0 * alpha) / 64.0;
    // 	double a33 = (25.0 + 14.0 * alpha) / 32.0;
    // 	double a34 = (35.0 + 58.0 * alpha) / 128.0;
    // 	double a35 = (7.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
    // 	double a36 = ( - 7.0 * ( - 1.0 + 2.0 * alpha)) / 64.0;
    // 	double a37 = ( - 1.0 + 2.0 * alpha) / 32.0;
    // 	double a38 = (1.0 - 2.0 * alpha) / 256.0;

    // 	// diagonal elements for R matrix for 1st derivative
    // 	std::vector<double> RDiagInterior{
    // 		alpha, 1.0, alpha
    // 	};
    //   std::vector<std::vector<double>> RDiagBoundary = {
    //       {1.0, alpha},{alpha, 1.0, alpha},{0,alpha, 1.0, alpha}
    //   };

    // 	// boundary elements for S matrix for 1st derivative
    // 	std::vector<std::vector<double>> SDiagBoundary
    // 		{{a00, a01, a02, a03, a04, a05, a06, a07, a08}, {a10, a11, a12,
    // a13, a14, a15, a16, a17, a18}, {a20, a21, a22, a23, a24, a25, a26, a27,
    // a28}, {a30, a31, a32, a33, a34, a35, a36, a37, a38}}
    // 	;

    // 	// diagonal elements for S matrix for 1st derivative
    // 	std::vector<double> SDiagInterior{
    // 		a4/2.0, a3/2.0, a2/2.0, a1/2.0, a0, a1/2.0, a2/2.0, a3/2.0,
    // a4/2.0
    // 	};

    // MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsFirstOrderFiltered(const
    // std::vector<double>& F_coeffs) {
    //     double alpha = F_coeffs[0];

    // double a0 = (11.0 + 10.0 * alpha) / 16.0;
    // double a1 = (15.0 + 34.0 * alpha) / 32.0;
    // double a2 = (3.0 * ( - 1.0 + 2.0 * alpha)) / 16.0;
    // double a3 = (1.0 - 2.0 * alpha) / 32.0;
    // double a00 = (63.0 + alpha) / 64.0;
    // double a01 = (3.0 + 29.0 * alpha) / 32.0;
    // double a02 = (15.0 * ( - 1.0 + alpha)) / 64.0;
    // double a03 = ( - 5.0 * ( - 1.0 + alpha)) / 16.0;
    // double a04 = (15.0 * ( - 1.0 + alpha)) / 64.0;
    // double a05 = ( - 3.0 * ( - 1.0 + alpha)) / 32.0;
    // double a06 = ( - 1.0 + alpha) / 64.0;
    // double a10 = (1.0 + 62.0 * alpha) / 64.0;
    // double a11 = (29.0 + 6.0 * alpha) / 32.0;
    // double a12 = (15.0 + 34.0 * alpha) / 64.0;
    // double a13 = (5.0 * ( - 1.0 + 2.0 * alpha)) / 16.0;
    // double a14 = ( - 15.0 * ( - 1.0 + 2.0 * alpha)) / 64.0;
    // double a15 = (3.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
    // double a16 = (1.0 - 2.0 * alpha) / 64.0;
    // double a20 = ( - 1.0 + 2.0 * alpha) / 64.0;
    // double a21 = (3.0 + 26.0 * alpha) / 32.0;
    // double a22 = (49.0 + 30.0 * alpha) / 64.0;
    // double a23 = (5.0 + 6.0 * alpha) / 16.0;
    // double a24 = (15.0 * ( - 1.0 + 2.0 * alpha)) / 64.0;
    // double a25 = ( - 3.0 * ( - 1.0 + 2.0 * alpha)) / 32.0;
    // double a26 = ( - 1.0 + 2.0 * alpha) / 64.0;

    // 		// diagonal elements for R matrix for 1st derivative
    // std::vector<double> RDiagInterior{ 0
    // };

    //         std::vector<std::vector<double>> RDiagBoundary = {{0}

    // };

    // // boundary elements for S matrix for 1st derivative
    // std::vector<std::vector<double>> SDiagBoundary
    // 	{{0}}
    // ;

    // // diagonal elements for S matrix for 1st derivative
    // std::vector<double> SDiagInterior{0
    // };

    // // diagonal elements for R matrix for 1st derivative
    // std::vector<double> RDiagInterior{
    // 	alpha, 1.0, alpha
    // };

    //         std::vector<std::vector<double>> RDiagBoundary = {
    //     {1.0, alpha},{alpha, 1.0, alpha},{0,alpha, 1.0, alpha}
    // };

    // // boundary elements for S matrix for 1st derivative
    // std::vector<std::vector<double>> SDiagBoundary
    // 	{{a00, a01, a02, a03, a04, a05, a06}, {a10, a11, a12, a13, a14, a15,
    // a16}, {a20, a21, a22, a23, a24, a25, a26}}
    // ;

    // // diagonal elements for S matrix for 1st derivative
    // std::vector<double> SDiagInterior{
    // 	a3/2.0, a2/2.0, a1/2.0, a0, a1/2.0, a2/2.0, a3/2.0
    // };

    // double alpha = F_coeffs[0];

    // double a0 = (93.0 + 70.0 * alpha) / 128.0;
    // double a1 = (7.0 + 18.0 * alpha) / 16.0;
    // double a2 = (7.0 * (-1.0 + 2.0 * alpha)) / 32.0;
    // double a3 = (1.0 - 2.0 * alpha) / 16.0;
    // double a4 = (-1.0 + 2.0 * alpha) / 128.0;

    // double a00 = (63.0 + alpha) / 64.0;
    // double a01 = (3.0 + 29.0 * alpha) / 32.0;
    // double a02 = (15.0 * (-1.0 + alpha)) / 64.0;
    // double a03 = (-5.0 * (-1.0 + alpha)) / 16.0;
    // double a04 = (15.0 * (-1.0 + alpha)) / 64.0;
    // double a05 = (-3.0 * (-1.0 + alpha)) / 32.0;
    // double a06 = (-1.0 + alpha) / 64.0;

    // double a10 = (1.0 + 62.0 * alpha) / 64.0;
    // double a11 = (29.0 + 6.0 * alpha) / 32.0;
    // double a12 = (15.0 + 34.0 * alpha) / 64.0;
    // double a13 = (5.0 * (-1.0 + 2.0 * alpha)) / 16.0;
    // double a14 = (-15.0 * (-1.0 + 2.0 * alpha)) / 64.0;
    // double a15 = (3.0 * (-1.0 + 2.0 * alpha)) / 32.0;
    // double a16 = (1.0 - 2.0 * alpha) / 64.0;

    // double a20 = (-1.0 + 2.0 * alpha) / 64.0;
    // double a21 = (3.0 + 26.0 * alpha) / 32.0;
    // double a22 = (49.0 + 30.0 * alpha) / 64.0;
    // double a23 = (5.0 + 6.0 * alpha) / 16.0;
    // double a24 = (15.0 * (-1.0 + 2.0 * alpha)) / 64.0;
    // double a25 = (-3.0 * (-1.0 + 2.0 * alpha)) / 32.0;
    // double a26 = (-1.0 + 2.0 * alpha) / 64.0;

    // std::vector<double> RDiagInterior{ {alpha, 1.0, alpha} };
    // std::vector<std::vector<double>> RDiagBoundary = {
    //     {alpha}, {1.0}, {alpha}
    // };
    // std::vector<double> SDiagInterior{ a4/2.0, a3/2.0, a2/2.0, a1/2.0, a0,
    // a1/2.0, a2/2.0, a3/2.0, a4/2.0 };

    // std::vector<std::vector<double>> SDiagBoundary = {
    //     {a00, a01, a02, a03, a04, a05, a06},
    //     {a10, a11, a12, a13, a14, a15, a16},
    //     {a20, a21, a22, a23, a24, a25, a26}
    // };

    return new MatrixDiagonalEntries{RDiagInterior, RDiagBoundary,
                                     SDiagInterior, SDiagBoundary};
}

class KimFilter_InMatrix : public InMatrixFilter {
   public:
    KimFilter_InMatrix(const std::vector<double>& input_coeffs)
        : InMatrixFilter(input_coeffs) {
        diagEntries = create_Kim_filter_diagonals(input_coeffs);
    }

    ~KimFilter_InMatrix() = default;

    InMatFilterType get_filter_type() const override {
        return InMatFilterType::IMFT_KIM;
    }
};
}  // namespace dendroderivs
