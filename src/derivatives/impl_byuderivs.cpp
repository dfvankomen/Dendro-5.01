#include "derivatives/impl_byuderivs.h"

#include "derivatives/derivs_compact.h"

namespace dendroderivs {
    MatrixDiagonalEntries* BYUDerivsT8R3DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 3.0 / 8.0;
		double alpha = alpha0 + D_coeffs[0];
		double a08 = D_coeffs[1];
		double a18 = D_coeffs[2];
		double a28 = D_coeffs[3];

		double a1 = 25.0 / 16.0;
		double a2 = 1.0 / 5.0;
		double a3 =  - 1.0 / 80.0;
		double a00 = ( - 503.0 - 980.0 * a08) / 140.0;
		double a01 = ( - 63.0 - 1784.0 * a08) / 20.0;
		double a02 = (7.0 * (3.0 + 56.0 * a08)) / 2.0;
		double a03 = ( - 7.0 * (5.0 + 168.0 * a08)) / 6.0;
		double a04 = (35.0 * (1.0 + 56.0 * a08)) / 12.0;
		double a05 = ( - 7.0 * (3.0 + 280.0 * a08)) / 20.0;
		double a06 = (7.0 * (1.0 + 168.0 * a08)) / 30.0;
		double a07 = ( - 1.0 - 392.0 * a08) / 42.0;
		double a10 = ( - 3.0 * (191.0 + 50456.0 * a18)) / 1960.0;
		double a11 = ( - 29.0 + 7680.0 * a18) / 20.0;
		double a12 = (43.0 + 12096.0 * a18) / 40.0;
		double a13 = (5.0 - 5376.0 * a18) / 6.0;
		double a14 = (5.0 * ( - 1.0 + 2016.0 * a18)) / 24.0;
		double a15 = (1.0 - 3584.0 * a18) / 20.0;
		double a16 = ( - 1.0 + 6720.0 * a18) / 120.0;
		double a17 = (1.0 - 16128.0 * a18) / 1470.0;
		double a20 = ( - 1.0 + 1470.0 * a28) / 126.0;
		double a21 = ( - 59.0 + 35160.0 * a28) / 120.0;
		double a22 = ( - 47.0 - 58800.0 * a28) / 60.0;
		double a23 = (25.0 + 1176.0 * a28) / 24.0;
		double a24 = (5.0 * (1.0 + 2940.0 * a28)) / 18.0;
		double a25 = ( - 1.0 - 5880.0 * a28) / 24.0;
		double a26 = (1.0 + 11760.0 * a28) / 180.0;
		double a27 = ( - 1.0 - 29400.0 * a28) / 2520.0;
		double gamma01 = 7.0 * (1.0 + 8.0 * a08);
		double gamma10 = (1.0 + 336.0 * a18) / 14.0;
		double gamma12 = ( - 3.0 * ( - 1.0 + 448.0 * a18)) / 2.0;
		double gamma21 = (1.0 - 840.0 * a28) / 6.0;
		double gamma23 = (5.0 * (1.0 + 1176.0 * a28)) / 6.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12},
			{0.0, gamma21, 1.0, gamma23}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3/6.0, -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0, a3/6.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsT64R3DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 1.0 / 3.0;
		double alpha = alpha0 + D_coeffs[0];
		double a04 = D_coeffs[1];
		double a14 = D_coeffs[2];
		double a24 = D_coeffs[3];

		double a1 = 14.0 / 9.0;
		double a2 = 1.0 / 9.0;
		double a00 = ( - 17.0 - 18.0 * a04) / 6.0;
		double a01 = (3.0 - 20.0 * a04) / 2.0;
		double a02 = (3.0 * (1.0 + 12.0 * a04)) / 2.0;
		double a03 = ( - 1.0 - 36.0 * a04) / 6.0;
		double a10 = ( - 5.0 - 93.0 * a14) / 9.0;
		double a11 = ( - 1.0 + 64.0 * a14) / 2.0;
		double a12 = 1.0 - 12.0 * a14;
		double a13 = (1.0 - 192.0 * a14) / 18.0;
		double a20 = ( - 1.0 + 18.0 * a24) / 18.0;
		double a21 =  - 1.0 + 8.0 * a24;
		double a22 = (1.0 - 36.0 * a24) / 2.0;
		double a23 = (5.0 + 72.0 * a24) / 9.0;
		double gamma01 = 3.0 * (1.0 + 4.0 * a04);
		double gamma10 = (1.0 + 24.0 * a14) / 6.0;
		double gamma12 = (1.0 - 48.0 * a14) / 2.0;
		double gamma21 = (1.0 - 12.0 * a24) / 2.0;
		double gamma23 = (1.0 + 36.0 * a24) / 6.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12},
			{0.0, gamma21, 1.0, gamma23}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14},
			{a20, a21, a22, a23, a24}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}

MatrixDiagonalEntries* BYUDerivsT6R2DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 1.0 / 3.0;
		double alpha = alpha0 + D_coeffs[0];
		double a06 = D_coeffs[1];
		double a16 = D_coeffs[2];

		double a1 = 14.0 / 9.0;
		double a2 = 1.0 / 9.0;
		double a00 = ( - 197.0 - 300.0 * a06) / 60.0;
		double a01 = ( - 5.0 - 462.0 * a06) / 12.0;
		double a02 = 5.0 * (1.0 + 15.0 * a06);
		double a03 = ( - 5.0 * (1.0 + 30.0 * a06)) / 3.0;
		double a04 = (5.0 * (1.0 + 60.0 * a06)) / 12.0;
		double a05 = ( - 1.0 - 150.0 * a06) / 20.0;
		double a10 = ( - 227.0 - 21240.0 * a16) / 600.0;
		double a11 = ( - 13.0 + 1728.0 * a16) / 12.0;
		double a12 = (7.0 + 90.0 * a16) / 6.0;
		double a13 = (1.0 - 480.0 * a16) / 3.0;
		double a14 = ( - 1.0 + 1080.0 * a16) / 24.0;
		double a15 = (1.0 - 2880.0 * a16) / 300.0;
		double gamma01 = 5.0 * (1.0 + 6.0 * a06);
		double gamma10 = (1.0 + 120.0 * a16) / 10.0;
		double gamma12 = 1.0 - 180.0 * a16;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}

MatrixDiagonalEntries* BYUDerivsT4R2DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 1.0 / 4.0;
		double alpha = alpha0 + D_coeffs[0];
		double a04 = D_coeffs[1];
		double a14 = D_coeffs[2];

		double a1 = 3.0 / 2.0;
		double a00 = ( - 17.0 - 18.0 * a04) / 6.0;
		double a01 = (3.0 - 20.0 * a04) / 2.0;
		double a02 = (3.0 * (1.0 + 12.0 * a04)) / 2.0;
		double a03 = ( - 1.0 - 36.0 * a04) / 6.0;
		double a10 = ( - 5.0 - 93.0 * a14) / 9.0;
		double a11 = ( - 1.0 + 64.0 * a14) / 2.0;
		double a12 = 1.0 - 12.0 * a14;
		double a13 = (1.0 - 192.0 * a14) / 18.0;
		double gamma01 = 3.0 * (1.0 + 4.0 * a04);
		double gamma10 = (1.0 + 24.0 * a14) / 6.0;
		double gamma12 = (1.0 - 48.0 * a14) / 2.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a1/2.0, 0.0, a1/2.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsT4R1DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 1.0 / 4.0;
		double alpha = alpha0 + D_coeffs[0];
		double a04 = D_coeffs[1];

		double a1 = 3.0 / 2.0;
		double a00 = ( - 17.0 - 18.0 * a04) / 6.0;
		double a01 = (3.0 - 20.0 * a04) / 2.0;
		double a02 = (3.0 * (1.0 + 12.0 * a04)) / 2.0;
		double a03 = ( - 1.0 - 36.0 * a04) / 6.0;
		double gamma01 = 3.0 * (1.0 + 4.0 * a04);

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a1/2.0, 0.0, a1/2.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs) {
    double alpha0  = 1.0 / 4.0;
    double alpha   = alpha0 + D_coeffs[0];
    double a05     = D_coeffs[1];
    double a15     = D_coeffs[2];
    double a25     = D_coeffs[3];

    double a1      = (2.0 * (2.0 + alpha)) / 3.0;
    double a2      = (-1.0 + 4.0 * alpha) / 3.0;
    double a00     = (-37.0 + 48.0 * a05) / 12.0;
    double a01     = (2.0 + 65.0 * a05) / 3.0;
    double a02     = 3.0 - 40.0 * a05;
    double a03     = (2.0 * (-1.0 + 30.0 * a05)) / 3.0;
    double a04     = (1.0 - 80.0 * a05) / 12.0;
    double a10     = (-43.0 + 2004.0 * a15) / 96.0;
    double a11     = (-5.0 * (1.0 + 90.0 * a15)) / 6.0;
    double a12     = (9.0 + 100.0 * a15) / 8.0;
    double a13     = (1.0 + 300.0 * a15) / 6.0;
    double a14     = (-1.0 - 900.0 * a15) / 96.0;
    double a20     = (-1.0 - 96.0 * a25) / 36.0;
    double a21     = (-7.0 - 285.0 * a25) / 9.0;
    double a22     = 80.0 * a25;
    double a23     = (7.0 - 300.0 * a25) / 9.0;
    double a24     = (1.0 - 480.0 * a25) / 36.0;
    double gamma01 = -4.0 * (-1.0 + 5.0 * a05);
    double gamma10 = (1.0 - 60.0 * a15) / 8.0;
    double gamma12 = (3.0 * (1.0 + 100.0 * a15)) / 4.0;
    double gamma21 = (1.0 + 60.0 * a25) / 3.0;
    double gamma23 = (1.0 - 120.0 * a25) / 3.0;

    // clang-format off
    // boundary elements for P matrix for 1st derivative

    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, gamma01}, 
        {gamma10, 1.0, gamma12}, 
        {0.0, gamma21, 1.0, gamma23}
    };

    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{
        alpha, 1.0, alpha
    };

    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {a00, a01, a02, a03, a04, a05},
        {a10, a11, a12, a13, a14, a15},
        {a20, a21, a22, a23, a24, a25}
    };

    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{
        -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P1DiagBoundary);
    check_end_of_boundaries(Q1DiagBoundary);

    // now we can build up the values

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs) {
    double alpha0  = 1.0 / 10.0;  // default value for alpha
    double alpha   = alpha0 + D_coeffs[0];
    double a06     = D_coeffs[1];
    double a16     = D_coeffs[2];
    double a26     = D_coeffs[3];

    double a1      = (-4.0 * (-1.0 + alpha)) / 3.0;
    double a2      = (-1.0 + 10.0 * alpha) / 3.0;
    double a00     = (1955.0 - 1644.0 * a06) / 156.0;
    double a01     = (-4057.0 + 1764.0 * a06) / 156.0;
    double a02     = (1117.0 + 1530.0 * a06) / 78.0;
    double a03     = (-5.0 * (11.0 + 564.0 * a06)) / 78.0;
    double a04     = (-29.0 + 3420.0 * a06) / 156.0;
    double a05     = (7.0 - 1116.0 * a06) / 156.0;
    double a10     = (177.0 + 13048.0 * a16) / 88.0;
    double a11     = (-3.0 * (169.0 + 23328.0 * a16)) / 44.0;
    double a12     = (27.0 * (29.0 + 4700.0 * a16)) / 44.0;
    double a13     = (-201.0 - 35360.0 * a16) / 22.0;
    double a14     = (81.0 * (1.0 + 200.0 * a16)) / 88.0;
    double a15     = (-3.0 * (1.0 + 288.0 * a16)) / 44.0;
    double a20     = (3.0 + 44.0 * a26) / 44.0;
    double a21     = (-3.0 * (-8.0 + 99.0 * a26)) / 22.0;
    double a22     = (3.0 * (-17.0 + 990.0 * a26)) / 22.0;
    double a23     = (12.0 - 2695.0 * a26) / 11.0;
    double a24     = (3.0 * (1.0 + 1980.0 * a26)) / 44.0;
    double a25     = (-27.0 * a26) / 2.0;
    double gamma01 = (137.0 - 180.0 * a06) / 13.0;
    double gamma10 = (2.0 * (1.0 + 90.0 * a16)) / 11.0;
    double gamma12 = (-131.0 - 22680.0 * a16) / 22.0;
    double gamma21 = 2.0 / 11.0;
    double gamma23 = (2.0 * (1.0 + 495.0 * a26)) / 11.0;

    // clang-format off
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, gamma01}, 
        {gamma10, 1.0, gamma12}, 
        {0.0, gamma21, 1.0, gamma23}
    };

    // diagonal elements for P matrix for 2nd derivative.
    std::vector<double> P2DiagInterior{
        alpha, 1.0, alpha
    };

    // boundary elements for Q matrix for 2nd derivative
    std::vector<std::vector<double>> Q2DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06},
        {a10, a11, a12, a13, a14, a15, a16},
        {a20, a21, a22, a23, a24, a25, a26}
    };

    // interior elements for Q matrix for 2nd derivative
    double t1 = -2.0 * (a1 + a2/4.0);
    std::vector<double> Q2DiagInterior{
        a2/4.0, a1, t1, a1, a2/4.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P2DiagBoundary);
    check_end_of_boundaries(Q2DiagBoundary);

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs) {
    double alpha0  = 1.0 / 3.0;
    double alpha   = alpha0 + D_coeffs[0];
    double a07     = D_coeffs[1];
    double a17     = D_coeffs[2];
    double a27     = D_coeffs[3];

    double a1      = (9.0 + alpha) / 6.0;
    double a2      = (-9.0 + 32.0 * alpha) / 15.0;
    double a3      = (1.0 - 3.0 * alpha) / 10.0;
    double a00     = (3.0 * (-23.0 + 40.0 * a07)) / 20.0;
    double a01     = (-17.0 + 609.0 * a07) / 10.0;
    double a02     = (-3.0 * (-5.0 + 84.0 * a07)) / 2.0;
    double a03     = (5.0 * (-2.0 + 63.0 * a07)) / 3.0;
    double a04     = (-5.0 * (-1.0 + 56.0 * a07)) / 4.0;
    double a05     = (3.0 * (-1.0 + 105.0 * a07)) / 10.0;
    double a06     = (1.0 - 252.0 * a07) / 30.0;
    double a10     = (-79.0 + 12990.0 * a17) / 240.0;
    double a11     = (-7.0 * (11.0 + 2100.0 * a17)) / 60.0;
    double a12     = (55.0 - 4998.0 * a17) / 48.0;
    double a13     = (5.0 * (1.0 + 735.0 * a17)) / 9.0;
    double a14     = (-5.0 * (1.0 + 1470.0 * a17)) / 48.0;
    double a15     = (1.0 + 2940.0 * a17) / 60.0;
    double a16     = (-1.0 - 7350.0 * a17) / 720.0;
    double a20     = (-1.0 - 720.0 * a27) / 90.0;
    double a21     = (-167.0 - 49140.0 * a27) / 300.0;
    double a22     = (7.0 * (-1.0 + 864.0 * a27)) / 12.0;
    double a23     = 1.0 - 105.0 * a27;
    double a24     = (1.0 - 1680.0 * a27) / 6.0;
    double a25     = (-1.0 + 3780.0 * a27) / 60.0;
    double a26     = (1.0 - 10080.0 * a27) / 900.0;
    double gamma01 = -6.0 * (-1.0 + 7.0 * a07);
    double gamma10 = (1.0 - 210.0 * a17) / 12.0;
    double gamma12 = (5.0 * (1.0 + 294.0 * a17)) / 4.0;
    double gamma21 = (1.0 + 420.0 * a27) / 5.0;
    double gamma23 = (-2.0 * (-1.0 + 630.0 * a27)) / 3.0;

    // clang-format off
    // boundary elements for P matrix for 1st derivative

    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, gamma01}, 
        {gamma10, 1.0, gamma12}, 
        {0.0, gamma21, 1.0, gamma23}
    };

    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{
        alpha, 1.0, alpha
    };

    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06, a07},
        {a10, a11, a12, a13, a14, a15, a16, a17},
        {a20, a21, a22, a23, a24, a25, a26, a27}
    };

    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{
        -a3/6.0, -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0, a3/6.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P1DiagBoundary);
    check_end_of_boundaries(Q1DiagBoundary);

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs) {
    double alpha0  = 2.0 / 11.0;
    double alpha   = alpha0 + D_coeffs[0];
    double a08     = D_coeffs[1];
    double a18     = D_coeffs[2];
    double a28     = D_coeffs[3];

    double a1      = (-3.0 * (-2.0 + 3.0 * alpha)) / 4.0;
    double a2      = (3.0 * (-1.0 + 8.0 * alpha)) / 5.0;
    double a3      = (2.0 - 11.0 * alpha) / 20.0;
    double a00     = (3647.0 - 3267.0 * a08) / 261.0;
    double a01     = (-2.0 * (17727.0 + 320.0 * a08)) / 1305.0;
    double a02     = (5889.0 + 46480.0 * a08) / 580.0;
    double a03     = (7031.0 - 154224.0 * a08) / 1044.0;
    double a04     = (7.0 * (-417.0 + 10580.0 * a08)) / 522.0;
    double a05     = (-47.0 * (-3.0 + 112.0 * a08)) / 58.0;
    double a06     = (-3119.0 + 196560.0 * a08) / 5220.0;
    double a07     = (339.0 - 47920.0 * a08) / 5220.0;
    double a10     = (2186893.0 - 344999160.0 * a18) / 3405960.0;
    double a11     = (526369.0 + 201232512.0 * a18) / 170298.0;
    double a12     = (-7.0 * (470931.0 + 114586240.0 * a18)) / 378440.0;
    double a13     = (1940803.0 + 362121984.0 * a18) / 340596.0;
    double a14     = (-583529.0 + 25809840.0 * a18) / 681192.0;
    double a15     = (7401.0 - 4531520.0 * a18) / 47305.0;
    double a16     = (-14839.0 + 29135232.0 * a18) / 681192.0;
    double a17     = (2659.0 - 16899840.0 * a18) / 1702980.0;
    double a20     = (2659.0 - 1702980.0 * a28) / 30780.0;
    double a21     = (9191.0 - 3073680.0 * a28) / 6840.0;
    double a22     = (-2969.0 + 3029880.0 * a28) / 570.0;
    double a23     = (-7.0 * (-11131.0 + 16941456.0 * a28)) / 12312.0;
    double a24     = (-1969.0 + 3757320.0 * a28) / 684.0;
    double a25     = (851.0 - 1795920.0 * a28) / 2280.0;
    double a26     = (4.0 * (-92.0 + 251685.0 * a28)) / 7695.0;
    double a27     = (23.0 - 110160.0 * a28) / 6840.0;
    double gamma01 = (363.0 - 560.0 * a08) / 29.0;
    double gamma10 = (563.0 - 191520.0 * a18) / 18922.0;
    double gamma12 = (9.0 * (7327.0 + 1704640.0 * a18)) / 18922.0;
    double gamma21 = (-9.0 * (-1.0 + 560.0 * a28)) / 38.0;
    double gamma23 = (-563.0 + 1123920.0 * a28) / 342.0;

    // clang-format off
    // boundary elements for P matrix for 2nd derivative
    // gamma01 = 3.0/10.0*(-15.0+4.0*a00);  // 5th order

    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, gamma01}, 
        {gamma10, 1.0, gamma12}, 
        {0.0, gamma21, 1.0, gamma23}
    };

    // diagonal elements for P matrix for 2nd derivative.
    std::vector<double> P2DiagInterior{
        alpha, 1.0, alpha
    };

    // boundary elements for Q matrix for 2nd derivative

    std::vector<std::vector<double>> Q2DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06, a07, a08},
        {a10, a11, a12, a13, a14, a15, a16, a17, a18},
        {a20, a21, a22, a23, a24, a25, a26, a27, a28}
    };

    // interior elements for Q matrix for 2nd derivative
    double t1 = -2.0 * (a1 + a2/4.0 + a3/9.0);
    std::vector<double> Q2DiagInterior{
        a3/9.0, a2/4.0, a1, t1, a1, a2/4.0, a3/9.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P2DiagBoundary);
    check_end_of_boundaries(Q2DiagBoundary);

    // store the entries for matrix creation
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 1.0 / 3.0;
		double alpha = alpha0 + D_coeffs[0];
		double a06 = D_coeffs[1];
		double a16 = D_coeffs[2];
		double a26 = D_coeffs[3];
		double a36 = D_coeffs[4];

		double a1 = 14.0 / 9.0;
		double a2 = 1.0 / 9.0;
		double a00 = ( - 197.0 - 300.0 * a06) / 60.0;
		double a01 = ( - 5.0 - 462.0 * a06) / 12.0;
		double a02 = 5.0 * (1.0 + 15.0 * a06);
		double a03 = ( - 5.0 * (1.0 + 30.0 * a06)) / 3.0;
		double a04 = (5.0 * (1.0 + 60.0 * a06)) / 12.0;
		double a05 = ( - 1.0 - 150.0 * a06) / 20.0;
		double a10 = ( - 227.0 - 21240.0 * a16) / 600.0;
		double a11 = ( - 13.0 + 1728.0 * a16) / 12.0;
		double a12 = (7.0 + 90.0 * a16) / 6.0;
		double a13 = (1.0 - 480.0 * a16) / 3.0;
		double a14 = ( - 1.0 + 1080.0 * a16) / 24.0;
		double a15 = (1.0 - 2880.0 * a16) / 300.0;
		double a20 = ( - 1.0 + 300.0 * a26) / 60.0;
		double a21 = ( - 31.0 + 3852.0 * a26) / 48.0;
		double a22 = ( - 1.0 - 675.0 * a26) / 3.0;
		double a23 = (11.0 + 900.0 * a26) / 12.0;
		double a24 = (1.0 + 900.0 * a26) / 12.0;
		double a25 = ( - 1.0 - 2700.0 * a26) / 240.0;
		double a30 = (1.0 + 240.0 * a36) / 240.0;
		double a31 = ( - 1.0 - 192.0 * a36) / 12.0;
		double a32 = ( - 11.0 - 780.0 * a36) / 12.0;
		double a33 = (1.0 + 480.0 * a36) / 3.0;
		double a34 = (31.0 - 3120.0 * a36) / 48.0;
		double a35 = (1.0 - 960.0 * a36) / 60.0;
		double gamma01 = 5.0 * (1.0 + 6.0 * a06);
		double gamma10 = (1.0 + 120.0 * a16) / 10.0;
		double gamma12 = 1.0 - 180.0 * a16;
		double gamma21 = (1.0 - 180.0 * a26) / 4.0;
		double gamma23 = (1.0 + 300.0 * a26) / 2.0;
		double gamma32 = (1.0 + 120.0 * a36) / 2.0;
		double gamma34 = (1.0 - 240.0 * a36) / 4.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12},
			{0.0, gamma21, 1.0, gamma23},
			{0.0, 0.0, gamma32, 1.0, gamma34}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26},
			{a30, a31, a32, a33, a34, a35, a36}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs) {
    double alpha0  = 2.0 / 11.0;
    double alpha   = alpha0 + D_coeffs[0];
    double a08     = D_coeffs[1];
    double a18     = D_coeffs[2];
    double a28     = D_coeffs[3];
    double a38     = D_coeffs[4];

    double a1      = (-3.0 * (-2.0 + 3.0 * alpha)) / 4.0;
    double a2      = (3.0 * (-1.0 + 8.0 * alpha)) / 5.0;
    double a3      = (2.0 - 11.0 * alpha) / 20.0;
    double a00     = (3647.0 - 3267.0 * a08) / 261.0;
    double a01     = (-2.0 * (17727.0 + 320.0 * a08)) / 1305.0;
    double a02     = (5889.0 + 46480.0 * a08) / 580.0;
    double a03     = (7031.0 - 154224.0 * a08) / 1044.0;
    double a04     = (7.0 * (-417.0 + 10580.0 * a08)) / 522.0;
    double a05     = (-47.0 * (-3.0 + 112.0 * a08)) / 58.0;
    double a06     = (-3119.0 + 196560.0 * a08) / 5220.0;
    double a07     = (339.0 - 47920.0 * a08) / 5220.0;
    double a10     = (2186893.0 - 344999160.0 * a18) / 3405960.0;
    double a11     = (526369.0 + 201232512.0 * a18) / 170298.0;
    double a12     = (-7.0 * (470931.0 + 114586240.0 * a18)) / 378440.0;
    double a13     = (1940803.0 + 362121984.0 * a18) / 340596.0;
    double a14     = (-583529.0 + 25809840.0 * a18) / 681192.0;
    double a15     = (7401.0 - 4531520.0 * a18) / 47305.0;
    double a16     = (-14839.0 + 29135232.0 * a18) / 681192.0;
    double a17     = (2659.0 - 16899840.0 * a18) / 1702980.0;
    double a20     = (2659.0 - 1702980.0 * a28) / 30780.0;
    double a21     = (9191.0 - 3073680.0 * a28) / 6840.0;
    double a22     = (-2969.0 + 3029880.0 * a28) / 570.0;
    double a23     = (-7.0 * (-11131.0 + 16941456.0 * a28)) / 12312.0;
    double a24     = (-1969.0 + 3757320.0 * a28) / 684.0;
    double a25     = (851.0 - 1795920.0 * a28) / 2280.0;
    double a26     = (4.0 * (-92.0 + 251685.0 * a28)) / 7695.0;
    double a27     = (23.0 - 110160.0 * a28) / 6840.0;
    double a30     = (-23.0 + 6840.0 * a38) / 6840.0;
    double a31     = (459.0 - 48640.0 * a38) / 3420.0;
    double a32     = (7.0 * (21.0 + 2432.0 * a38)) / 152.0;
    double a33     = (-751.0 - 306432.0 * a38) / 342.0;
    double a34     = (7.0 * (189.0 + 311600.0 * a38)) / 1368.0;
    double a35     = (51.0 - 340480.0 * a38) / 380.0;
    double a36     = (-23.0 + 766080.0 * a38) / 6840.0;
    double a37     = (-128.0 * a38) / 9.0;
    double gamma01 = (363.0 - 560.0 * a08) / 29.0;
    double gamma10 = (563.0 - 191520.0 * a18) / 18922.0;
    double gamma12 = (9.0 * (7327.0 + 1704640.0 * a18)) / 18922.0;
    double gamma21 = (-9.0 * (-1.0 + 560.0 * a28)) / 38.0;
    double gamma23 = (-563.0 + 1123920.0 * a28) / 342.0;
    double gamma32 = 9.0 / 38.0;
    double gamma34 = (9.0 - 21280.0 * a38) / 38.0;

    // clang-format off
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, gamma01}, 
        {gamma10, 1.0, gamma12}, 
        {0.0, gamma21, 1.0, gamma23},
        {0.0, 0.0, gamma32, 1.0, gamma34}
    };

    // diagonal elements for P matrix for 2nd derivative.
    std::vector<double> P2DiagInterior{
        alpha, 1.0, alpha
    };

    // boundary elements for Q matrix for 2nd derivative

    std::vector<std::vector<double>> Q2DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06, a07, a08},
        {a10, a11, a12, a13, a14, a15, a16, a17, a18},
        {a20, a21, a22, a23, a24, a25, a26, a27, a28},
        {a30, a31, a32, a33, a34, a35, a36, a37, a38}
    };

    // interior elements for Q matrix for 2nd derivative
    double t1 = -2.0 * (a1 + a2/4.0 + a3/9.0);
    std::vector<double> Q2DiagInterior{
        a3/9.0, a2/4.0, a1, t1, a1, a2/4.0, a3/9.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P2DiagBoundary);
    check_end_of_boundaries(Q2DiagBoundary);

    // store the entries for matrix creation
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs) {
    double beta0   = -1.0 / 114.0;
    double beta    = beta0 + D_coeffs[0];
    double a06     = D_coeffs[1];
    double a17     = D_coeffs[2];
    double a27     = D_coeffs[3];
    double a07     = D_coeffs[4];
    double a11     = D_coeffs[5];
    double a22     = D_coeffs[6];

    double a1      = (-2.0 * (-7.0 + 12.0 * beta)) / 9.0;
    double a2      = (1.0 + 114.0 * beta) / 9.0;
    double alpha   = (1.0 + 12.0 * beta) / 3.0;
    double a00     = (-227.0 + 600.0 * a06 + 5400.0 * a07) / 60.0;
    double a01     = (-65.0 + 1644.0 * a06 + 14175.0 * a07) / 6.0;
    double a02     = (35.0 - 375.0 * a06 - 3528.0 * a07) / 3.0;
    double a03     = (-5.0 * (-2.0 + 120.0 * a06 + 945.0 * a07)) / 3.0;
    double a04     = (5.0 * (-1.0 + 120.0 * a06 + 840.0 * a07)) / 12.0;
    double a05     = (1.0 - 300.0 * a06 - 1575.0 * a07) / 30.0;
    double a10     = (-36499.0 - 11820.0 * a11 + 611400.0 * a17) / 64800.0;
    double a12     = (1331.0 + 780.0 * a11 + 161112.0 * a17) / 288.0;
    double a13     = (-263.0 - 240.0 * a11 - 25725.0 * a17) / 81.0;
    double a14     = (-29.0 - 20.0 * a11 - 9800.0 * a17) / 32.0;
    double a15     = (23.0 + 15.0 * a11 + 14700.0 * a17) / 225.0;
    double a16     = (-19.0 - 12.0 * a11 - 29400.0 * a17) / 2592.0;
    double a20     = (-1663.0 - 1452.0 * a22 - 1000512.0 * a27) / 16200.0;
    double a21     = (-1993.0 - 804.0 * a22 - 89964.0 * a27) / 2700.0;
    double a23     = (67.0 + 12.0 * a22 - 67473.0 * a27) / 81.0;
    double a24     = (5.0 - 156.0 * a22 + 169344.0 * a27) / 216.0;
    double a25     = (-1.0 - 4.0 * a22 + 15876.0 * a27) / 100.0;
    double a26     = (1.0 + 3.0 * a22 - 31752.0 * a27) / 2025.0;
    double gamma01 = -10.0 * (-1.0 + 12.0 * a06 + 105.0 * a07);
    double gamma02 = -10.0 * (-1.0 + 30.0 * a06 + 252.0 * a07);
    double gamma10 = (167.0 + 60.0 * a11 - 4200.0 * a17) / 1080.0;
    double gamma12 = (-47.0 - 60.0 * a11 - 5880.0 * a17) / 24.0;
    double gamma13 = (-77.0 - 60.0 * a11 - 14700.0 * a17) / 27.0;
    double gamma20 = (13.0 + 12.0 * a22 + 9072.0 * a27) / 540.0;
    double gamma21 = (19.0 + 12.0 * a22 + 5292.0 * a27) / 45.0;
    double gamma23 = (-2.0 * (-5.0 + 12.0 * a22 - 7938.0 * a27)) / 27.0;
    double gamma24 = (-1.0 - 12.0 * a22 + 21168.0 * a27) / 36.0;

    // clang-format off
    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, gamma01, gamma02}, 
        {gamma10, 1.0, gamma12, gamma13}, 
        {gamma20, gamma21, 1.0, gamma23, gamma24}
    };

    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{
        beta, alpha, 1.0, alpha, beta
    };

    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06, a07 },
        {a10, a11, a12, a13, a14, a15, a16, a17 },
        {a20, a21, a22, a23, a24, a25, a26, a27 }
    };

    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{
        -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P1DiagBoundary);
    check_end_of_boundaries(Q1DiagBoundary);

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs) {
    double beta0 = -1.0 / 194.0;
    double beta  = beta0 + D_coeffs[0];
    double a07   = D_coeffs[1];
    double a17   = D_coeffs[2];
    double a27   = D_coeffs[3];
    double a08   = D_coeffs[4];
    double a18   = D_coeffs[5];
    double a28   = D_coeffs[6];

    double a1    = (-12.0 * (-1.0 + 26.0 * beta)) / 11.0;
    double a2    = (3.0 * (1.0 + 194.0 * beta)) / 11.0;
    double alpha = (2.0 * (1.0 + 62.0 * beta)) / 11.0;
    double a00   = (48241.0 - 549180.0 * a07 - 5052780.0 * a08) / 900.0;
    double a01   = (16389.0 - 262820.0 * a07 - 2412720.0 * a08) / 25.0;
    double a02   = (-31359.0 + 486000.0 * a07 + 4463120.0 * a08) / 20.0;
    double a03   = (43622.0 - 667035.0 * a07 - 6130080.0 * a08) / 45.0;
    double a04   = (-2529.0 + 37220.0 * a07 + 344520.0 * a08) / 20.0;
    double a05   = (429.0 - 5670.0 * a07 - 54320.0 * a08) / 25.0;
    double a06   = (-1169.0 + 9720.0 * a07 + 123120.0 * a08) / 900.0;
    double a10 = (753829.0 - 24461820.0 * a17 - 355648320.0 * a18) / 1114560.0;
    double a11 = (57209.0 + 4218420.0 * a17 + 66251520.0 * a18) / 20640.0;
    double a12 = (-58367.0 - 8677836.0 * a17 - 103614720.0 * a18) / 8256.0;
    double a13 = (172793.0 + 92712420.0 * a17 + 979299072.0 * a18) / 55728.0;
    double a14 = (4453.0 - 7381500.0 * a17 - 72938880.0 * a18) / 8256.0;
    double a15 = (-391.0 + 2318580.0 * a17 + 21031680.0 * a18) / 20640.0;
    double a16 = (529.0 - 15888780.0 * a17 - 110004480.0 * a18) / 1114560.0;
    double a20 = (3134.0 - 3468987.0 * a27 - 52441929.0 * a28) / 13527.0;
    double a21 = (4139.0 - 505079.0 * a27 - 8559936.0 * a28) / 4843.0;
    double a22 = (-3025.0 + 5519178.0 * a27 + 85280160.0 * a28) / 1002.0;
    double a23 =
        (964810.0 - 3539581335.0 * a27 - 55007284608.0 * a28) / 392283.0;
    double a24 = (5.0 * (-423.0 + 3093467.0 * a27 + 49645602.0 * a28)) / 4843.0;
    double a25 = (-1343.0 + 10316889.0 * a27 + 142690368.0 * a28) / 14529.0;
    double a26 = (1817.0 - 22934502.0 * a27 - 250456608.0 * a28) / 784566.0;
    double gamma01 = (-36.0 * (-17.0 + 235.0 * a07 + 2160.0 * a08)) / 5.0;
    double gamma02 = (-9.0 * (-339.0 + 5220.0 * a07 + 47920.0 * a08)) / 5.0;
    double gamma10 = (23.0 - 1620.0 * a17 - 23040.0 * a18) / 688.0;
    double gamma12 = (5.0 * (467.0 + 8028.0 * a17 + 191232.0 * a18)) / 688.0;
    double gamma13 = (2659.0 - 1702980.0 * a17 - 16899840.0 * a18) / 3096.0;
    double gamma20 = (989.0 - 1151820.0 * a27 - 17308800.0 * a28) / 87174.0;
    double gamma21 =
        (-4.0 * (-3698.0 + 3539295.0 * a27 + 53804160.0 * a28)) / 43587.0;
    double gamma23 =
        (36.0 * (-47.0 + 481475.0 * a27 + 7424320.0 * a28)) / 4843.0;
    double gamma24 =
        (-13231.0 + 91009980.0 * a27 + 1339693920.0 * a28) / 87174.0;

    // clang-format off
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, gamma01, gamma02}, 
        {gamma10, 1.0, gamma12, gamma13}, 
        {gamma20, gamma21, 1.0, gamma23, gamma24}
    };

    // diagonal elements for P matrix for 2nd derivative.
    std::vector<double> P2DiagInterior{
        beta, alpha, 1.0, alpha, beta
    };

    // boundary elements for Q matrix for 2nd derivative
    std::vector<std::vector<double>> Q2DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06, a07, a08},
        {a10, a11, a12, a13, a14, a15, a16, a17, a18},
        {a20, a21, a22, a23, a24, a25, a26, a27, a28},
    };

    // interior elements for Q matrix for 2nd derivative
    double t1 = -2.0*(a1 + a2/4.0);
    std::vector<double> Q2DiagInterior{
        a2/4.0, a1, t1, a1, a2/4.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P2DiagBoundary);
    check_end_of_boundaries(Q2DiagBoundary);

    // store the entries for matrix creation
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}

MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs) {
    double beta0 = -1.0 / 194.0;  // default value of beta
    double beta  = beta0 + D_coeffs[0];
    double a08   = D_coeffs[1];
    double a19   = D_coeffs[2];
    double a29   = D_coeffs[3];
    double a39   = D_coeffs[4];
    double a09   = D_coeffs[5];
    double a11   = D_coeffs[6];
    double a22   = D_coeffs[7];
    double a33   = D_coeffs[8];

    // clang-format off
    double a1 = ( - 5.0 * ( - 15.0 + 28.0 * beta)) / 48.0;
    double a2 = (3.0 + 142.0 * beta) / 15.0;
    double a3 = ( - 1.0 + 36.0 * beta) / 80.0;
    double alpha = (3.0 + 20.0 * beta) / 8.0;
    double a00 = ( - 573.0 + 2940.0 * a08 + 31360.0 * a09) / 140.0;
    double a01 = ( - 203.0 + 8712.0 * a08 + 90846.0 * a09) / 10.0;
    double a02 = (301.0 - 1176.0 * a08 - 17856.0 * a09) / 20.0;
    double a03 = ( - 7.0 * ( - 5.0 + 504.0 * a08 + 5040.0 * a09)) / 3.0;
    double a04 = (7.0 * ( - 5.0 + 840.0 * a08 + 8064.0 * a09)) / 12.0;
    double a05 = ( - 7.0 * ( - 1.0 + 280.0 * a08 + 2520.0 * a09)) / 10.0;
    double a06 = (7.0 * ( - 1.0 + 504.0 * a08 + 4032.0 * a09)) / 60.0;
    double a07 = (1.0 - 1176.0 * a08 - 7056.0 * a09) / 105.0;
    double a10 = ( - 1409239.0 - 417620.0 * a11 + 59173380.0 * a19) / 2822400.0;
    double a12 = (7237.0 + 4060.0 * a11 + 1764180.0 * a19) / 800.0;
    double a13 = ( - 9719.0 - 7420.0 * a11 - 1031940.0 * a19) / 1800.0;
    double a14 = ( - 739.0 - 420.0 * a11 - 428652.0 * a19) / 192.0;
    double a15 = (129.0 + 70.0 * a11 + 119070.0 * a19) / 150.0;
    double a16 = ( - 53.0 - 28.0 * a11 - 79380.0 * a19) / 288.0;
    double a17 = (809.0 + 420.0 * a11 + 2143260.0 * a19) / 29400.0;
    double a18 = ( - 39.0 - 20.0 * a11 - 238140.0 * a19) / 19200.0;
    double a20 = ( - 108977.0 - 72260.0 * a22 - 283939200.0 * a29) / 1411200.0;
    double a21 = ( - 31761.0 - 11980.0 * a22 - 11250900.0 * a29) / 44100.0;
    double a23 = (1163.0 + 740.0 * a22 - 6933600.0 * a29) / 900.0;
    double a24 = ( - 111.0 - 380.0 * a22 + 1648512.0 * a29) / 288.0;
    double a25 = ( - 37.0 - 60.0 * a22 + 874800.0 * a29) / 300.0;
    double a26 = (7.0 + 10.0 * a22 - 259200.0 * a29) / 450.0;
    double a27 = ( - 3.0 - 4.0 * a22 + 194400.0 * a29) / 1764.0;
    double a28 = (47.0 + 60.0 * a22 - 6998400.0 * a29) / 470400.0;
    double a30 = ( - 47.0 - 60.0 * a33 + 470400.0 * a39) / 33600.0;
    double a31 = ( - 13723.0 - 13340.0 * a33 + 69178200.0 * a39) / 88200.0;
    double a32 = ( - 2563.0 - 940.0 * a33 - 297600.0 * a39) / 3600.0;
    double a34 = (109.0 + 20.0 * a33 + 771456.0 * a39) / 144.0;
    double a35 = (211.0 - 1220.0 * a33 - 9055200.0 * a39) / 1800.0;
    double a36 = ( - 7.0 - 60.0 * a33 - 1411200.0 * a39) / 1200.0;
    double a37 = (1.0 + 5.0 * a33 + 235200.0 * a39) / 1575.0;
    double a38 = ( - 1.0 - 4.0 * a33 - 470400.0 * a39) / 28224.0;
    double gamma01 =  - 14.0 * ( - 1.0 + 24.0 * a08 + 252.0 * a09);
    double gamma02 =  - 21.0 * ( - 1.0 + 56.0 * a08 + 576.0 * a09);
    double gamma10 = (433.0 + 140.0 * a11 - 26460.0 * a19) / 3360.0;
    double gamma12 = ( - 153.0 - 140.0 * a11 - 34020.0 * a19) / 40.0;
    double gamma13 = ( - 223.0 - 140.0 * a11 - 79380.0 * a19) / 30.0;
    double gamma20 = (29.0 + 20.0 * a22 + 86400.0 * a29) / 1680.0;
    double gamma21 = (39.0 + 20.0 * a22 + 48600.0 * a29) / 105.0;
    double gamma23 = (1.0 - 20.0 * a22 + 64800.0 * a29) / 15.0;
    double gamma24 = ( - 9.0 - 20.0 * a22 + 155520.0 * a29) / 24.0;
    double gamma31 = (19.0 + 20.0 * a33 - 117600.0 * a39) / 420.0;
    double gamma32 = (29.0 + 20.0 * a33 - 67200.0 * a39) / 60.0;
    double gamma34 = (11.0 - 20.0 * a33 - 94080.0 * a39) / 24.0;
    double gamma35 = (1.0 - 20.0 * a33 - 235200.0 * a39) / 60.0;

    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, gamma01, gamma02}, 
        {gamma10, 1.0, gamma12, gamma13}, 
        {gamma20, gamma21, 1.0, gamma23, gamma24},
        {0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
    };

    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{
        beta, alpha, 1.0, alpha, beta
    };

    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06, a07, a08, a09 },
        {a10, a11, a12, a13, a14, a15, a16, a17, a18, a19 },
        {a20, a21, a22, a23, a24, a25, a26, a27, a28, a29 },
        {a30, a31, a32, a33, a34, a35, a36, a37, a38, a39 } 
    };

    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{
        -a3/6.0, -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0, a3/6.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P1DiagBoundary);
    check_end_of_boundaries(Q1DiagBoundary);

    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}
MatrixDiagonalEntries* BYUDerivsT4R42DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 1.0 / 4.0;
		double alpha = alpha0 + D_coeffs[0];
		double a34 = D_coeffs[1];
		double a23 = D_coeffs[2];
		double a12 = D_coeffs[3];
		double a01 = D_coeffs[4];

		double a1 = 3.0 / 2.0;
		double a00 = ( - 197.0 + 18.0 * a01) / 60.0;
		double a02 = ( - 3.0 * ( - 7.0 + 3.0 * a01)) / 5.0;
		double a03 = ( - 16.0 + 9.0 * a01) / 15.0;
		double a04 = (3.0 - 2.0 * a01) / 20.0;
		double a10 = ( - 51.0 + 31.0 * a12) / 36.0;
		double a11 = (13.0 - 16.0 * a12) / 6.0;
		double a13 = ( - 15.0 + 16.0 * a12) / 18.0;
		double a14 = (1.0 - a12) / 12.0;
		double a20 = ( - 1.0 + a23) / 8.0;
		double a21 = ( - 14.0 + 9.0 * a23) / 9.0;
		double a22 = (7.0 - 9.0 * a23) / 4.0;
		double a24 = ( - 5.0 + 9.0 * a23) / 72.0;
		double a30 = (5.0 - 9.0 * a34) / 93.0;
		double a31 = ( - 39.0 + 64.0 * a34) / 62.0;
		double a32 = (3.0 * ( - 17.0 + 12.0 * a34)) / 31.0;
		double a33 = (413.0 - 576.0 * a34) / 186.0;
		double gamma01 = ( - 6.0 * ( - 4.0 + a01)) / 5.0;
		double gamma10 = (3.0 - 2.0 * a12) / 6.0;
		double gamma12 = ( - 3.0 + 4.0 * a12) / 2.0;
		double gamma21 = (11.0 - 9.0 * a23) / 12.0;
		double gamma23 = ( - 1.0 + 3.0 * a23) / 4.0;
		double gamma32 = ( - 3.0 * ( - 37.0 + 48.0 * a34)) / 62.0;
		double gamma34 = (3.0 * ( - 1.0 + 8.0 * a34)) / 62.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12},
			{0.0, gamma21, 1.0, gamma23},
			{0.0, 0.0, gamma32, 1.0, gamma34}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14},
			{a20, a21, a22, a23, a24},
			{a30, a31, a32, a33, a34}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a1/2.0, 0.0, a1/2.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsT6R42DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 1.0 / 3.0;
		double alpha = alpha0 + D_coeffs[0];
		double a36 = D_coeffs[1];
		double a25 = D_coeffs[2];
		double a14 = D_coeffs[3];
		double a03 = D_coeffs[4];

		double a1 = 14.0 / 9.0;
		double a2 = 1.0 / 9.0;
		double a00 = ( - 187.0 + 6.0 * a03) / 60.0;
		double a01 = (260.0 + 231.0 * a03) / 300.0;
		double a02 = (5.0 - 3.0 * a03) / 2.0;
		double a04 = ( - 5.0 - 6.0 * a03) / 12.0;
		double a05 = (4.0 + 3.0 * a03) / 20.0;
		double a06 = ( - 5.0 - 3.0 * a03) / 150.0;
		double a10 = ( - 185.0 - 354.0 * a14) / 450.0;
		double a11 = ( - 19.0 + 64.0 * a14) / 20.0;
		double a12 = (85.0 + 24.0 * a14) / 72.0;
		double a13 = (5.0 - 96.0 * a14) / 27.0;
		double a15 = ( - 5.0 - 192.0 * a14) / 900.0;
		double a16 = (1.0 + 24.0 * a14) / 1080.0;
		double a20 = ( - 1.0 - 24.0 * a25) / 54.0;
		double a21 = ( - 152.0 - 1605.0 * a25) / 225.0;
		double a22 = ( - 1.0 + 80.0 * a25) / 4.0;
		double a23 = ( - 4.0 * ( - 2.0 + 15.0 * a25)) / 9.0;
		double a24 = (1.0 - 120.0 * a25) / 18.0;
		double a26 = ( - 1.0 - 240.0 * a25) / 2700.0;
		double a30 = (1.0 + 240.0 * a36) / 240.0;
		double a31 = ( - 1.0 - 192.0 * a36) / 12.0;
		double a32 = ( - 11.0 - 780.0 * a36) / 12.0;
		double a33 = (1.0 + 480.0 * a36) / 3.0;
		double a34 = (31.0 - 3120.0 * a36) / 48.0;
		double a35 = (1.0 - 960.0 * a36) / 60.0;
		double gamma01 = (20.0 - 3.0 * a03) / 5.0;
		double gamma10 = (5.0 + 12.0 * a14) / 45.0;
		double gamma12 = (5.0 - 24.0 * a14) / 6.0;
		double gamma21 = (4.0 * (1.0 + 15.0 * a25)) / 15.0;
		double gamma23 = ( - 4.0 * ( - 1.0 + 30.0 * a25)) / 9.0;
		double gamma32 = (1.0 + 120.0 * a36) / 2.0;
		double gamma34 = (1.0 - 240.0 * a36) / 4.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12},
			{0.0, gamma21, 1.0, gamma23},
			{0.0, 0.0, gamma32, 1.0, gamma34}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26},
			{a30, a31, a32, a33, a34, a35, a36}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs) {
    // redefine constants for the 2nd derivative
    double beta0 = 23.0 / 2358.0;  // default value of beta
    double beta  = beta0 + D_coeffs[0];
    double a09   = D_coeffs[1];
    double a19   = D_coeffs[2];
    double a29   = D_coeffs[3];
    double a39   = D_coeffs[4];
    double a010  = D_coeffs[5];
    double a110  = D_coeffs[6];
    double a210  = D_coeffs[7];
    double a310  = D_coeffs[8];

    double a1    = (-3.0 * (-49.0 + 794.0 * beta)) / 152.0;
    double a2    = (3.0 * (17.0 + 818.0 * beta)) / 95.0;
    double a3    = (-23.0 + 2358.0 * beta) / 760.0;
    double alpha = (9.0 + 214.0 * beta) / 38.0;
    double a00 =
        (50985901.0 - 2862052704.0 * a010 - 253215648.0 * a09) / 2456496.0;
    double a01 =
        (67682336.0 - 18998582400.0 * a010 - 1701076545.0 * a09) / 767655.0;
    double a02 =
        (-28183546.0 + 6061493025.0 * a010 + 541270440.0 * a09) / 109665.0;
    double a03 =
        (-2.0 * (-9343009.0 + 1696107600.0 * a010 + 149941260.0 * a09)) /
        109665.0;
    double a04 =
        (7.0 * (-676957.0 + 2728800.0 * a010 - 987552.0 * a09)) / 175464.0;
    double a05 =
        (2.0 * (60950.0 + 27895392.0 * a010 + 3011967.0 * a09)) / 21933.0;
    double a06 =
        (-2.0 * (53789.0 + 80273025.0 * a010 + 8954820.0 * a09)) / 109665.0;
    double a07 =
        (2.0 * (45511.0 + 173451600.0 * a010 + 21534660.0 * a09)) / 767655.0;
    double a08 =
        (-86431.0 - 833061600.0 * a010 - 137158560.0 * a09) / 12282480.0;
    double a10 = (432329129.0 + 702985993920.0 * a110 + 63506898000.0 * a19) /
                 650905920.0;
    double a11 = (112117043.0 - 1065139488000.0 * a110 - 91692030780.0 * a19) /
                 40681620.0;
    double a12 =
        (-5894981.0 - 80155988500.0 * a110 - 6361728720.0 * a19) / 645740.0;
    double a13 = (44166803.0 + 2017796544000.0 * a110 + 163326527280.0 * a19) /
                 5811660.0;
    double a14 = (-20531803.0 - 2119607784000.0 * a110 - 172683642096.0 * a19) /
                 9298656.0;
    double a15 =
        (242467.0 + 23411467520.0 * a110 + 1947149400.0 * a19) / 645740.0;
    double a16 =
        (-384379.0 - 42956991000.0 * a110 - 3749805360.0 * a19) / 5811660.0;
    double a17 =
        (342443.0 + 50416128000.0 * a110 + 4886224560.0 * a19) / 40681620.0;
    double a18 =
        (-39119.0 - 8785672000.0 * a110 - 1122984240.0 * a19) / 72322880.0;
    double a20 =
        (563136185.0 + 255534425839728.0 * a210 + 15657679128672.0 * a29) /
        14559668928.0;
    double a21 =
        (60043829.0 + 916594602000.0 * a210 + 56445873561.0 * a29) / 75831609.0;
    double a22 =
        (126814705.0 - 54898575284700.0 * a210 - 3356567390304.0 * a29) /
        129997044.0;
    double a23 =
        (-130584571.0 + 24314020333200.0 * a210 + 1520160886944.0 * a29) /
        32499261.0;
    double a24 =
        (597285835.0 - 128576830158000.0 * a210 - 8641470799584.0 * a29) /
        346658784.0;
    double a25 =
        (16799995.0 + 714454918128.0 * a210 + 88970270886.0 * a29) / 32499261.0;
    double a26 = (-3867259.0 - 1013718699000.0 * a210 - 90886846176.0 * a29) /
                 129997044.0;
    double a27 =
        (151115.0 + 104378425200.0 * a210 + 9856645344.0 * a29) / 75831609.0;
    double a28 = (-1194407.0 - 1928926515600.0 * a210 - 236433066912.0 * a29) /
                 14559668928.0;
    double a30 =
        (1194407.0 + 236433066912.0 * a310 + 14559668928.0 * a39) / 225150912.0;
    double a31 =
        (1139498.0 + 202412448000.0 * a310 + 12231684927.0 * a39) / 3517983.0;
    double a32 =
        (1169837.0 - 82020613500.0 * a310 - 5143018176.0 * a39) / 2010276.0;
    double a33 =
        (-2326727.0 - 680625504000.0 * a310 - 40315066848.0 * a39) / 1005138.0;
    double a34 =
        (24079135.0 + 18310172580000.0 * a310 + 1081225881792.0 * a39) /
        16082208.0;
    double a35 =
        (-13099.0 - 186529125888.0 * a310 - 10715079186.0 * a39) / 502569.0;
    double a36 =
        (-141991.0 - 228309921000.0 * a310 - 14398340544.0 * a39) / 2010276.0;
    double a37 =
        (22903.0 + 41047776000.0 * a310 + 2830667616.0 * a39) / 7035966.0;
    double a38 =
        (-26617.0 - 64818684000.0 * a310 - 5793738048.0 * a39) / 225150912.0;
    double gamma01 =
        (-8.0 * (-9509.0 + 1083600.0 * a010 + 96390.0 * a09)) / 2437.0;
    double gamma02 =
        (-2.0 * (-125603.0 + 27052200.0 * a010 + 2424240.0 * a09)) / 2437.0;
    double gamma10 = (4097.0 + 7963200.0 * a110 + 761040.0 * a19) / 129148.0;
    double gamma12 =
        (-9.0 * (-22481.0 + 278068000.0 * a110 + 23256240.0 * a19)) / 64574.0;
    double gamma13 =
        (-26437.0 - 4429152000.0 * a110 - 359437680.0 * a19) / 32287.0;
    double gamma20 =
        (9413.0 + 6313885200.0 * a210 + 386447040.0 * a29) / 7222058.0;
    double gamma21 =
        (4.0 * (87721.0 + 20712258000.0 * a210 + 1270091340.0 * a29)) /
        3611029.0;
    double gamma23 =
        (-4.0 * (-2036071.0 + 241211250000.0 * a210 + 14626374840.0 * a29)) /
        3611029.0;
    double gamma24 =
        (9648101.0 - 459592938000.0 * a210 - 16186474080.0 * a29) / 14444116.0;
    double gamma31 =
        (3.0 * (989.0 + 188160000.0 * a310 + 11464320.0 * a39)) / 111682.0;
    double gamma32 =
        (6.0 * (3841.0 + 594405000.0 * a310 + 35797440.0 * a39)) / 55841.0;
    double gamma34 =
        (-7.0 * (-347.0 + 14694300000.0 * a310 + 873046080.0 * a39)) / 223364.0;
    double gamma35 =
        (-9413.0 - 16755379200.0 * a310 - 1024097760.0 * a39) / 111682.0;

    // clang-format off
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, gamma01, gamma02}, 
        {gamma10, 1.0, gamma12, gamma13}, 
        {gamma20, gamma21, 1.0, gamma23, gamma24},
        {0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
    };

    // diagonal elements for P matrix for 2nd derivative.
    std::vector<double> P2DiagInterior{
        beta, alpha, 1.0, alpha, beta
    };

    // boundary elements for Q matrix for 2nd derivative
    std::vector<std::vector<double>> Q2DiagBoundary{
        {a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a010},
        {a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a110},
        {a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a210},
        {a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a310}
    };

    // interior elements for Q matrix for 2nd derivative
    double t1 = -2.0*(a1 + a2/4.0 + a3/9.0);
    std::vector<double> Q2DiagInterior{
        a3/9.0, a2/4.0, a1, t1, a1, a2/4.0, a3/9.0
    };
    // clang-format on

    // pass the diagonal entries throuh the check function that could
    // potentially remove them
    check_end_of_boundaries(P2DiagBoundary);
    check_end_of_boundaries(Q2DiagBoundary);

    // store the entries for matrix creation
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary};

    return diagEntries;
}
MatrixDiagonalEntries* BYUDerivsP6R32DiagonalsSecondOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 12.0 / 97.0;
		double alpha = alpha0 + D_coeffs[0];
		double a06 = D_coeffs[1];
		double a16 = D_coeffs[2];
		double a26 = D_coeffs[3];

		double a1 = 120.0 / 97.0;
		double beta =  - 1.0 / 194.0;
		double a00 = (177.0 - 524.0 * a06) / 16.0;
		double a01 = ( - 3.0 * (169.0 + 1476.0 * a06)) / 8.0;
		double a02 = (27.0 * (29.0 + 380.0 * a06)) / 8.0;
		double a03 = ( - 201.0 - 3140.0 * a06) / 4.0;
		double a04 = (81.0 * (1.0 + 20.0 * a06)) / 16.0;
		double a05 = ( - 3.0 * (1.0 + 36.0 * a06)) / 8.0;
		double a10 = (421.0 + 13592.0 * a16) / 632.0;
		double a11 = (5465.0 - 441936.0 * a16) / 1896.0;
		double a12 = (5.0 * ( - 1387.0 + 98172.0 * a16)) / 948.0;
		double a13 = ( - 5.0 * ( - 209.0 + 27472.0 * a16)) / 316.0;
		double a14 = (5.0 * (179.0 + 53784.0 * a16)) / 1896.0;
		double a15 = ( - 23.0 - 27216.0 * a16) / 1896.0;
		double a20 = (3565.0 + 269178.0 * a26) / 18078.0;
		double a21 = (2.0 * (3680.0 + 78813.0 * a26)) / 9039.0;
		double a22 = ( - 5.0 * (1219.0 + 259173.0 * a26)) / 3013.0;
		double a23 = (20.0 * (368.0 + 321063.0 * a26)) / 9039.0;
		double a24 = ( - 5.0 * ( - 713.0 + 989658.0 * a26)) / 18078.0;
		double a25 = ( - 918.0 * a26) / 23.0;
		double gamma01 = (11.0 - 180.0 * a06) / 2.0;
		double gamma02 = ( - 131.0 - 1980.0 * a06) / 4.0;
		double gamma10 = (23.0 + 1620.0 * a16) / 711.0;
		double gamma12 = (547.0 - 22680.0 * a16) / 158.0;
		double gamma13 = (1169.0 + 110160.0 * a16) / 1422.0;
		double gamma20 = (23.0 + 1620.0 * a26) / 2358.0;
		double gamma21 = (344.0 * (23.0 + 1620.0 * a26)) / 27117.0;
		double gamma23 = ( - 8.0 * ( - 989.0 + 938385.0 * a26)) / 27117.0;
		double gamma24 = (529.0 - 3782700.0 * a26) / 54234.0;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1/1.0);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			-a1/1.0, t1, a1/1.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsP6R32DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 17.0 / 57.0;
		double alpha = alpha0 + D_coeffs[0];
		double a05 = D_coeffs[1];
		double a15 = D_coeffs[2];
		double a25 = D_coeffs[3];

		double a1 = 30.0 / 19.0;
		double beta =  - 1.0 / 114.0;
		double a00 = ( - 43.0 - 72.0 * a05) / 12.0;
		double a01 = ( - 5.0 * (4.0 + 75.0 * a05)) / 3.0;
		double a02 = 9.0 + 80.0 * a05;
		double a03 = (4.0 * (1.0 + 45.0 * a05)) / 3.0;
		double a04 = ( - 1.0 - 120.0 * a05) / 12.0;
		double a10 = ( - 47.0 - 2244.0 * a15) / 144.0;
		double a11 = ( - 4.0 + 225.0 * a15) / 3.0;
		double a12 = (3.0 + 500.0 * a15) / 4.0;
		double a13 = ( - 4.0 * ( - 2.0 + 375.0 * a15)) / 9.0;
		double a14 = (1.0 - 900.0 * a15) / 48.0;
		double a20 = ( - 25.0 + 2784.0 * a25) / 216.0;
		double a21 = (5.0 * ( - 4.0 + 165.0 * a25)) / 27.0;
		double a22 =  - 100.0 * a25;
		double a23 = (20.0 * (1.0 + 15.0 * a25)) / 27.0;
		double a24 = (25.0 * (1.0 + 384.0 * a25)) / 216.0;
		double gamma01 = 4.0 * (2.0 + 15.0 * a05);
		double gamma02 = 6.0 * (1.0 + 20.0 * a05);
		double gamma10 = (1.0 + 60.0 * a15) / 12.0;
		double gamma12 = ( - 3.0 * ( - 1.0 + 100.0 * a15)) / 2.0;
		double gamma13 = (1.0 - 300.0 * a15) / 3.0;
		double gamma20 = (1.0 - 120.0 * a25) / 36.0;
		double gamma21 = ( - 4.0 * ( - 1.0 + 75.0 * a25)) / 9.0;
		double gamma23 = (4.0 * (1.0 + 150.0 * a25)) / 9.0;
		double gamma24 = (1.0 + 600.0 * a25) / 36.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05},
			{a10, a11, a12, a13, a14, a15},
			{a20, a21, a22, a23, a24, a25}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a1/2.0, 0.0, a1/2.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsP6R2DiagonalsFirstOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 17.0 / 57.0;
		double alpha = alpha0 + D_coeffs[0];
		double a05 = D_coeffs[1];
		double a15 = D_coeffs[2];

		double a1 = 30.0 / 19.0;
		double beta =  - 1.0 / 114.0;
		double a00 = ( - 43.0 - 72.0 * a05) / 12.0;
		double a01 = ( - 5.0 * (4.0 + 75.0 * a05)) / 3.0;
		double a02 = 9.0 + 80.0 * a05;
		double a03 = (4.0 * (1.0 + 45.0 * a05)) / 3.0;
		double a04 = ( - 1.0 - 120.0 * a05) / 12.0;
		double a10 = ( - 47.0 - 2244.0 * a15) / 144.0;
		double a11 = ( - 4.0 + 225.0 * a15) / 3.0;
		double a12 = (3.0 + 500.0 * a15) / 4.0;
		double a13 = ( - 4.0 * ( - 2.0 + 375.0 * a15)) / 9.0;
		double a14 = (1.0 - 900.0 * a15) / 48.0;
		double gamma01 = 4.0 * (2.0 + 15.0 * a05);
		double gamma02 = 6.0 * (1.0 + 20.0 * a05);
		double gamma10 = (1.0 + 60.0 * a15) / 12.0;
		double gamma12 = ( - 3.0 * ( - 1.0 + 100.0 * a15)) / 2.0;
		double gamma13 = (1.0 - 300.0 * a15) / 3.0;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05},
			{a10, a11, a12, a13, a14, a15}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a1/2.0, 0.0, a1/2.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsP6R2DiagonalsSecondOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 12.0 / 97.0;
		double alpha = alpha0 + D_coeffs[0];
		double a06 = D_coeffs[1];
		double a16 = D_coeffs[2];

		double a1 = 120.0 / 97.0;
		double beta =  - 1.0 / 194.0;
		double a00 = (177.0 - 524.0 * a06) / 16.0;
		double a01 = ( - 3.0 * (169.0 + 1476.0 * a06)) / 8.0;
		double a02 = (27.0 * (29.0 + 380.0 * a06)) / 8.0;
		double a03 = ( - 201.0 - 3140.0 * a06) / 4.0;
		double a04 = (81.0 * (1.0 + 20.0 * a06)) / 16.0;
		double a05 = ( - 3.0 * (1.0 + 36.0 * a06)) / 8.0;
		double a10 = (421.0 + 13592.0 * a16) / 632.0;
		double a11 = (5465.0 - 441936.0 * a16) / 1896.0;
		double a12 = (5.0 * ( - 1387.0 + 98172.0 * a16)) / 948.0;
		double a13 = ( - 5.0 * ( - 209.0 + 27472.0 * a16)) / 316.0;
		double a14 = (5.0 * (179.0 + 53784.0 * a16)) / 1896.0;
		double a15 = ( - 23.0 - 27216.0 * a16) / 1896.0;
		double gamma01 = (11.0 - 180.0 * a06) / 2.0;
		double gamma02 = ( - 131.0 - 1980.0 * a06) / 4.0;
		double gamma10 = (23.0 + 1620.0 * a16) / 711.0;
		double gamma12 = (547.0 - 22680.0 * a16) / 158.0;
		double gamma13 = (1169.0 + 110160.0 * a16) / 1422.0;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16}
		};

		double t1 = -2.0 * (a1/1.0);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			-a1/1.0, t1, a1/1.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* BYUDerivsT6R2DiagonalsSecondOrder(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 2.0 / 11.0;
		double alpha = alpha0 + D_coeffs[0];
		double a07 = D_coeffs[1];
		double a17 = D_coeffs[2];

		double a1 = 12.0 / 11.0;
		double a2 = 3.0 / 11.0;
		double a00 = (7.0 * (1871.0 + 1620.0 * a07)) / 990.0;
		double a01 = ( - 2943.0 - 700.0 * a07) / 110.0;
		double a02 = ( - 3.0 * ( - 191.0 + 648.0 * a07)) / 44.0;
		double a03 = (167.0 + 7695.0 * a07) / 99.0;
		double a04 = ( - 2.0 * (9.0 + 335.0 * a07)) / 11.0;
		double a05 = (3.0 * (19.0 + 1080.0 * a07)) / 110.0;
		double a06 = ( - 131.0 - 16200.0 * a07) / 1980.0;
		double a10 = (48241.0 + 14404140.0 * a17) / 110160.0;
		double a11 = (1821.0 - 493220.0 * a17) / 340.0;
		double a12 = ( - 10453.0 + 2142324.0 * a17) / 816.0;
		double a13 = (1283.0 - 230490.0 * a17) / 162.0;
		double a14 = ( - 281.0 + 30740.0 * a17) / 272.0;
		double a15 = (143.0 + 10620.0 * a17) / 1020.0;
		double a16 = ( - 1169.0 - 788220.0 * a17) / 110160.0;
		double gamma01 = (18.0 * (7.0 + 10.0 * a07)) / 11.0;
		double gamma10 = (5.0 * (1.0 + 1692.0 * a17)) / 612.0;
		double gamma12 = ( - 3.0 * ( - 113.0 + 21780.0 * a17)) / 68.0;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01},
			{gamma10, 1.0, gamma12}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			alpha, 1.0, alpha
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07},
			{a10, a11, a12, a13, a14, a15, a16, a17}
		};

		double t1 = -2.0 * (a1/1.0, a2/4.0);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			-a2/4.0, -a1/1.0, t1, a1/1.0, a2/4.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}

//FIRST DERIVATIVES

MatrixDiagonalEntries* createA4_1_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.820500840908007;
		double gamma02 = 10.282867796276006;
		double gamma10 = 0.08185064221837997;
		double gamma12 = 1.68906274817202;
		double gamma13 = 0.496147893139263;
		double gamma20 = 0.015218619469010378;
		double gamma21 = 0.313606951149977;
		double gamma23 = 0.7939030813731411;
		double gamma24 = 0.14913730019362592;
		double a00 =  - 3.7344525106358377;
		double a01 =  - 10.765902293945098;
		double a02 = 11.158777464424904;
		double a03 = 3.8932795799678406;
		double a04 =  - 0.6389839716987255;
		double a05 = 0.09587727382044378;
		double a06 =  - 0.008595531659208112;
		double a10 =  - 0.31850909623408924;
		double a11 =  - 1.396944235857124;
		double a12 = 0.536401605870878;
		double a13 = 1.122316892896269;
		double a14 = 0.05945060450387947;
		double a15 =  - 0.0027438380250889333;
		double a16 = 0.000028066844770735025;
		double a20 =  - 0.06702204160713257;
		double a21 =  - 0.6115799835980097;
		double a22 =  - 0.4353270505424934;
		double a23 = 0.7147300302515313;
		double a24 = 0.3855053316097469;
		double a25 = 0.014273663642987601;
		double a26 =  - 0.0005799497566284084;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3/6.0, -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0, a3/6.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_2_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.82050086297005;
		double gamma02 = 10.282867809058313;
		double gamma10 = 0.08185064221829087;
		double gamma12 = 1.6890627481714804;
		double gamma13 = 0.4961478931388701;
		double gamma20 = 0.017918887158094664;
		double gamma21 = 0.33500390219730425;
		double gamma23 = 0.7548768581531722;
		double gamma24 = 0.13667343083285768;
		double a00 =  - 3.73445251380582;
		double a01 =  - 10.765902315893966;
		double a02 = 11.158777481815811;
		double a03 = 3.8932795799881186;
		double a04 =  - 0.6389839738157034;
		double a05 = 0.09587727397420936;
		double a06 =  - 0.008595531676508965;
		double a10 =  - 0.31850909623427476;
		double a11 =  - 1.396944235854646;
		double a12 = 0.5364016058704685;
		double a13 = 1.1223168928950014;
		double a14 = 0.059450604503400145;
		double a15 =  - 0.0027438380250710795;
		double a16 = 0.00002806684476978311;
		double a20 =  - 0.07670359108573083;
		double a21 =  - 0.6273443641711278;
		double a22 =  - 0.37833992259630045;
		double a23 = 0.7128462614236629;
		double a24 = 0.3572245284141374;
		double a25 = 0.012842138314361781;
		double a26 =  - 0.0005250502990408262;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_3_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 11.720569036874554;
		double gamma02 = 15.554462579626016;
		double gamma10 = 0.1081072833795741;
		double gamma12 = 0.9879415284429277;
		double gamma13 =  - 0.019091049862053604;
		double gamma20 = 0.001594503696184088;
		double gamma21 = 0.11896661542190605;
		double gamma23 = 1.4712607824914257;
		double gamma24 = 0.39065672002582436;
		double a00 =  - 3.869256849728272;
		double a01 =  - 15.34137988312291;
		double a02 = 12.883637155318173;
		double a03 = 7.717258677339867;
		double a04 =  - 1.6841269756473085;
		double a05 = 0.32933821713218175;
		double a06 =  - 0.035470348869858045;
		double a10 =  - 0.3964964225533291;
		double a11 =  - 1.0420535477054953;
		double a12 = 1.1470798670545719;
		double a13 = 0.3494082667125378;
		double a14 =  - 0.06737997031063933;
		double a15 = 0.010504184266465403;
		double a16 =  - 0.0010623774453651048;
		double a20 =  - 0.008283084846223839;
		double a21 =  - 0.3750015455221021;
		double a22 =  - 1.2054305952790025;
		double a23 = 0.6249651589579047;
		double a24 = 0.9236932467174234;
		double a25 = 0.04166802585994469;
		double a26 =  - 0.0016112058874788778;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_4_Diagonals() {
			double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.092199664350433;
		double gamma02 = 10.362304786419537;
		double gamma10 = 0.10511747198457709;
		double gamma12 = 1.3999606300191687;
		double gamma13 = 0.36626783281004827;
		double gamma20 = 0.010832502509735786;
		double gamma21 = 0.23818680686599208;
		double gamma23 = 1.1282790769923212;
		double gamma24 = 0.28864969919854105;
		double a00 =  - 3.583310087673525;
		double a01 =  - 9.995923639727009;
		double a02 = 9.549519843633083;
		double a03 = 4.968473573688743;
		double a04 =  - 1.17661765410291;
		double a05 = 0.2747094330971768;
		double a06 =  - 0.03685144021286404;
		double a10 =  - 0.380753971849515;
		double a11 =  - 1.1730254249425538;
		double a12 = 0.6536944559326225;
		double a13 = 0.8627866179739128;
		double a14 = 0.03735935464342531;
		double a15 = 0.0004486340570757499;
		double a16 =  - 0.0005096658149371253;
		double a20 =  - 0.04678686551681484;
		double a21 =  - 0.5104482821604837;
		double a22 =  - 0.770033072247565;
		double a23 = 0.6228933619764672;
		double a24 = 0.67271251593522;
		double a25 = 0.033041689527390984;
		double a26 =  - 0.001379347514430855;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3/6.0, -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0, a3/6.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_5_Diagonals() {
			double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.82050086297005;
		double gamma02 = 10.282867809058313;
		double gamma10 = 0.08185064221829087;
		double gamma12 = 1.6890627481714804;
		double gamma13 = 0.4961478931388701;
		double gamma20 = 0.010832502509737819;
		double gamma21 = 0.2381868068661776;
		double gamma23 = 1.128279076992181;
		double gamma24 = 0.28864969919845446;
		double a00 =  - 3.73445251380582;
		double a01 =  - 10.765902315893966;
		double a02 = 11.158777481815811;
		double a03 = 3.8932795799881186;
		double a04 =  - 0.6389839738157034;
		double a05 = 0.09587727397420936;
		double a06 =  - 0.008595531676508965;
		double a10 =  - 0.31850909623427476;
		double a11 =  - 1.396944235854646;
		double a12 = 0.5364016058704685;
		double a13 = 1.1223168928950014;
		double a14 = 0.059450604503400145;
		double a15 =  - 0.0027438380250710795;
		double a16 = 0.00002806684476978311;
		double a20 =  - 0.046786865516800155;
		double a21 =  - 0.5104482821608083;
		double a22 =  - 0.7700330722473107;
		double a23 = 0.6228933619765842;
		double a24 = 0.6727125159356983;
		double a25 = 0.033041689527405986;
		double a26 =  - 0.0013793475144312974;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_6_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.82050086297005;
		double gamma02 = 10.282867809058313;
		double gamma10 = 0.36393960516685847;
		double gamma12 =  - 6.34581504273248;
		double gamma13 =  - 5.455121314714797;
		double gamma20 = 0.010832502509737819;
		double gamma21 = 0.2381868068661776;
		double gamma23 = 1.128279076992181;
		double gamma24 = 0.28864969919845446;
		double a00 =  - 3.73445251380582;
		double a01 =  - 10.765902315893966;
		double a02 = 11.158777481815811;
		double a03 = 3.8932795799881186;
		double a04 =  - 0.6389839738157034;
		double a05 = 0.09587727397420936;
		double a06 =  - 0.008595531676508965;
		double a10 =  - 1.1690129655580526;
		double a11 = 2.5693037088752675;
		double a12 = 7.670098232641336;
		double a13 =  - 7.8155049102691265;
		double a14 =  - 1.3854291627394484;
		double a15 = 0.1415361269516958;
		double a16 =  - 0.010991029895683714;
		double a20 =  - 0.046786865516800155;
		double a21 =  - 0.5104482821608083;
		double a22 =  - 0.7700330722473107;
		double a23 = 0.6228933619765842;
		double a24 = 0.6727125159356983;
		double a25 = 0.033041689527405986;
		double a26 =  - 0.0013793475144312974;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_7_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 11.72056913137962;
		double gamma02 = 15.554462594162011;
		double gamma10 = 0.36393960516685847;
		double gamma12 =  - 6.34581504273248;
		double gamma13 =  - 5.455121314714797;
		double gamma20 = 0.010832502509737819;
		double gamma21 = 0.2381868068661776;
		double gamma23 = 1.128279076992181;
		double gamma24 = 0.28864969919845446;
		double a00 =  - 3.8692568733227635;
		double a01 =  - 15.341379957576446;
		double a02 = 12.883637219907088;
		double a03 = 7.717258726068238;
		double a04 =  - 1.6841269855708267;
		double a05 = 0.3293382198137049;
		double a06 =  - 0.03547034916968481;
		double a10 =  - 1.1690129655580526;
		double a11 = 2.5693037088752675;
		double a12 = 7.670098232641336;
		double a13 =  - 7.8155049102691265;
		double a14 =  - 1.3854291627394484;
		double a15 = 0.1415361269516958;
		double a16 =  - 0.010991029895683714;
		double a20 =  - 0.046786865516800155;
		double a21 =  - 0.5104482821608083;
		double a22 =  - 0.7700330722473107;
		double a23 = 0.6228933619765842;
		double a24 = 0.6727125159356983;
		double a25 = 0.033041689527405986;
		double a26 =  - 0.0013793475144312974;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;

	}
MatrixDiagonalEntries* createA4_8_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 10.124423686900291;
		double gamma02 = 13.061603846624669;
		double gamma10 = 0.36393960516397067;
		double gamma12 =  - 6.345815043464687;
		double gamma13 =  - 5.455121315585303;
		double gamma20 = 0.010832502509735786;
		double gamma21 = 0.23818680686599208;
		double gamma23 = 1.1282790769923212;
		double gamma24 = 0.28864969919854105;
		double a00 =  - 3.6694533066008543;
		double a01 =  - 12.369980002354707;
		double a02 = 10.464970745197672;
		double a03 = 6.987379266251513;
		double a04 =  - 1.7858832667253233;
		double a05 = 0.4303380766868438;
		double a06 =  - 0.05737147795252053;
		double a10 =  - 1.1690129655423311;
		double a11 = 2.569303709628177;
		double a12 = 7.6700982331484715;
		double a13 =  - 7.815504911034343;
		double a14 =  - 1.3854291629690247;
		double a15 = 0.14153612692248557;
		double a16 =  - 0.010991029895809476;
		double a20 =  - 0.04678686551681484;
		double a21 =  - 0.5104482821604837;
		double a22 =  - 0.770033072247565;
		double a23 = 0.6228933619764672;
		double a24 = 0.67271251593522;
		double a25 = 0.033041689527390984;
		double a26 =  - 0.001379347514430855;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3/6.0, -a2/4.0, -a1/2.0, 0.0, a1/2.0, a2/4.0, a3/6.0
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_9_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.092199640716895;
		double gamma02 = 10.362304769374216;
		double gamma10 = 0.36393960516685847;
		double gamma12 =  - 6.34581504273248;
		double gamma13 =  - 5.455121314714797;
		double gamma20 = 0.010832502509737819;
		double gamma21 = 0.2381868068661776;
		double gamma23 = 1.128279076992181;
		double gamma24 = 0.28864969919845446;
		double a00 =  - 3.583310083032476;
		double a01 =  - 9.995923596790389;
		double a02 = 9.549519902457892;
		double a03 = 4.968473477723128;
		double a04 =  - 1.1766177021837405;
		double a05 = 0.2747094406267014;
		double a06 =  - 0.03685144035384007;
		double a10 =  - 1.1690129655580526;
		double a11 = 2.5693037088752675;
		double a12 = 7.670098232641336;
		double a13 =  - 7.8155049102691265;
		double a14 =  - 1.3854291627394484;
		double a15 = 0.1415361269516958;
		double a16 =  - 0.010991029895683714;
		double a20 =  - 0.046786865516800155;
		double a21 =  - 0.5104482821608083;
		double a22 =  - 0.7700330722473107;
		double a23 = 0.6228933619765842;
		double a24 = 0.6727125159356983;
		double a25 = 0.033041689527405986;
		double a26 =  - 0.0013793475144312974;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_10_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.82050086297005;
		double gamma02 = 10.282867809058313;
		double gamma10 = 0.06432504195032081;
		double gamma12 = 2.253048588922067;
		double gamma13 = 0.8947355749649457;
		double gamma20 = 0.010832502509737819;
		double gamma21 = 0.2381868068661776;
		double gamma23 = 1.128279076992181;
		double gamma24 = 0.28864969919845446;
		double a00 =  - 3.73445251380582;
		double a01 =  - 10.765902315893966;
		double a02 = 11.158777481815811;
		double a03 = 3.8932795799881186;
		double a04 =  - 0.6389839738157034;
		double a05 = 0.09587727397420936;
		double a06 =  - 0.008595531676508965;
		double a10 =  - 0.2633689121814849;
		double a11 =  - 1.6681586712239223;
		double a12 = 0.040493403700274426;
		double a13 = 1.7567567175694963;
		double a14 = 0.14259309069789727;
		double a15 =  - 0.008532325517430831;
		double a16 = 0.00021669695996460306;
		double a20 =  - 0.046786865516800155;
		double a21 =  - 0.5104482821608083;
		double a22 =  - 0.7700330722473107;
		double a23 = 0.6228933619765842;
		double a24 = 0.6727125159356983;
		double a25 = 0.033041689527405986;
		double a26 =  - 0.0013793475144312974;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_11_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 7.382379779647204;
		double gamma02 = 6.118391304701528;
		double gamma10 = 0.10511747198457366;
		double gamma12 = 1.3999606300191616;
		double gamma13 = 0.36626783281006475;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999154;
		double gamma23 = 1.110302482205355;
		double gamma24 = 0.3070536064807964;
		double a00 =  - 3.4387560411379563;
		double a01 =  - 6.120972374567166;
		double a02 = 7.819283685673539;
		double a03 = 2.0327058980606223;
		double a04 =  - 0.3578340204155545;
		double a05 = 0.07704263923889951;
		double a06 =  - 0.011469786857726549;
		double a10 =  - 0.3807539718494947;
		double a11 =  - 1.1730254249425167;
		double a12 = 0.6536944559325863;
		double a13 = 0.8627866179738892;
		double a14 = 0.03735935464344308;
		double a15 = 0.0004486340570785038;
		double a16 =  - 0.0005096658149371573;
		double a20 =  - 0.058823348930518396;
		double a21 =  - 0.5249167357886487;
		double a22 =  - 0.714498395784144;
		double a23 = 0.5832981939847537;
		double a24 = 0.6747270495592782;
		double a25 = 0.0425803899581001;
		double a26 =  - 0.0023671529988773697;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_12_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 10.124423686900291;
		double gamma02 = 13.061603846624669;
		double gamma10 = 0.10511747198457709;
		double gamma12 = 1.3999606300191687;
		double gamma13 = 0.36626783281004827;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999557;
		double gamma23 = 1.1103024822053873;
		double gamma24 = 0.3070536064807722;
		double a00 =  - 3.6694533066008543;
		double a01 =  - 12.369980002354707;
		double a02 = 10.464970745197672;
		double a03 = 6.987379266251513;
		double a04 =  - 1.7858832667253233;
		double a05 = 0.4303380766868438;
		double a06 =  - 0.05737147795252053;
		double a10 =  - 0.380753971849515;
		double a11 =  - 1.1730254249425538;
		double a12 = 0.6536944559326225;
		double a13 = 0.8627866179739128;
		double a14 = 0.03735935464342531;
		double a15 = 0.0004486340570757499;
		double a16 =  - 0.0005096658149371253;
		double a20 =  - 0.058823348930526466;
		double a21 =  - 0.5249167357886829;
		double a22 =  - 0.7144983957840794;
		double a23 = 0.5832981939848182;
		double a24 = 0.6747270495592943;
		double a25 = 0.04258038995809657;
		double a26 =  - 0.0023671529988773224;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_13_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 7.382379779647204;
		double gamma02 = 6.118391304701528;
		double gamma10 = 0.17427954622883998;
		double gamma12 = 0.16936101668592404;
		double gamma13 =  - 0.40635680897269083;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999154;
		double gamma23 = 1.110302482205355;
		double gamma24 = 0.3070536064807964;
		double a00 =  - 3.4387560411379563;
		double a01 =  - 6.120972374567166;
		double a02 = 7.819283685673539;
		double a03 = 2.0327058980606223;
		double a04 =  - 0.3578340204155545;
		double a05 = 0.07704263923889951;
		double a06 =  - 0.011469786857726549;
		double a10 =  - 0.5727257684097207;
		double a11 =  - 0.4112262980330496;
		double a12 = 1.4956215666978379;
		double a13 =  - 0.3873992442738074;
		double a14 =  - 0.14379120110929572;
		double a15 = 0.022496214935192167;
		double a16 =  - 0.002975269806106884;
		double a20 =  - 0.058823348930518396;
		double a21 =  - 0.5249167357886487;
		double a22 =  - 0.714498395784144;
		double a23 = 0.5832981939847537;
		double a24 = 0.6747270495592782;
		double a25 = 0.0425803899581001;
		double a26 =  - 0.0023671529988773697;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_14_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 11.72056913137962;
		double gamma02 = 15.554462594162011;
		double gamma10 = 0.1081072833783197;
		double gamma12 = 0.9879415284063021;
		double gamma13 =  - 0.01909104991398132;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999154;
		double gamma23 = 1.110302482205355;
		double gamma24 = 0.3070536064807964;
		double a00 =  - 3.8692568733227635;
		double a01 =  - 15.341379957576446;
		double a02 = 12.883637219907088;
		double a03 = 7.717258726068238;
		double a04 =  - 1.6841269855708267;
		double a05 = 0.3293382198137049;
		double a06 =  - 0.03547034916968481;
		double a10 =  - 0.3964964225581668;
		double a11 =  - 1.0420535476865542;
		double a12 = 1.1470798670311428;
		double a13 = 0.3494082667185844;
		double a14 =  - 0.06737997033390317;
		double a15 = 0.010504184268101917;
		double a16 =  - 0.0010623774454914835;
		double a20 =  - 0.058823348930518396;
		double a21 =  - 0.5249167357886487;
		double a22 =  - 0.714498395784144;
		double a23 = 0.5832981939847537;
		double a24 = 0.6747270495592782;
		double a25 = 0.0425803899581001;
		double a26 =  - 0.0023671529988773697;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_15_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 10.124421930341821;
		double gamma02 = 13.061601592420779;
		double gamma10 = 0.36393960516685847;
		double gamma12 =  - 6.34581504273248;
		double gamma13 =  - 5.455121314714797;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999154;
		double gamma23 = 1.110302482205355;
		double gamma24 = 0.3070536064807964;
		double a00 =  - 3.6694526780300785;
		double a01 =  - 12.36997789511369;
		double a02 = 10.464968940637677;
		double a03 = 6.9873780576381375;
		double a04 =  - 1.7858829589397163;
		double a05 = 0.4303380013201043;
		double a06 =  - 0.05737146796374551;
		double a10 =  - 1.1690129655580526;
		double a11 = 2.5693037088752675;
		double a12 = 7.670098232641336;
		double a13 =  - 7.8155049102691265;
		double a14 =  - 1.3854291627394484;
		double a15 = 0.1415361269516958;
		double a16 =  - 0.010991029895683714;
		double a20 =  - 0.058823348930518396;
		double a21 =  - 0.5249167357886487;
		double a22 =  - 0.714498395784144;
		double a23 = 0.5832981939847537;
		double a24 = 0.6747270495592782;
		double a25 = 0.0425803899581001;
		double a26 =  - 0.0023671529988773697;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_16_Diagonals() {
			double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 9.092199640716895;
		double gamma02 = 10.362304769374216;
		double gamma10 = 0.36393960516685847;
		double gamma12 =  - 6.34581504273248;
		double gamma13 =  - 5.455121314714797;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999154;
		double gamma23 = 1.110302482205355;
		double gamma24 = 0.3070536064807964;
		double a00 =  - 3.583310083032476;
		double a01 =  - 9.995923596790389;
		double a02 = 9.549519902457892;
		double a03 = 4.968473477723128;
		double a04 =  - 1.1766177021837405;
		double a05 = 0.2747094406267014;
		double a06 =  - 0.03685144035384007;
		double a10 =  - 1.1690129655580526;
		double a11 = 2.5693037088752675;
		double a12 = 7.670098232641336;
		double a13 =  - 7.8155049102691265;
		double a14 =  - 1.3854291627394484;
		double a15 = 0.1415361269516958;
		double a16 =  - 0.010991029895683714;
		double a20 =  - 0.058823348930518396;
		double a21 =  - 0.5249167357886487;
		double a22 =  - 0.714498395784144;
		double a23 = 0.5832981939847537;
		double a24 = 0.6747270495592782;
		double a25 = 0.0425803899581001;
		double a26 =  - 0.0023671529988773697;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_17_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 7.382379779615977;
		double gamma02 = 6.118391304628355;
		double gamma10 = 0.0643250419496078;
		double gamma12 = 2.2530485889042495;
		double gamma13 = 0.8947355749693718;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999557;
		double gamma23 = 1.1103024822053873;
		double gamma24 = 0.3070536064807722;
		double a00 =  - 3.438756041105928;
		double a01 =  - 6.120972374525143;
		double a02 = 7.819283685619206;
		double a03 = 2.0327058980757524;
		double a04 =  - 0.3578340204185375;
		double a05 = 0.07704263923912963;
		double a06 =  - 0.011469786857738811;
		double a10 =  - 0.26336891217791125;
		double a11 =  - 1.6681586712175933;
		double a12 = 0.04049340369078189;
		double a13 = 1.7567567175439678;
		double a14 = 0.142593090701886;
		double a15 =  - 0.00853232551642941;
		double a16 = 0.0002166969599412292;
		double a20 =  - 0.058823348930526466;
		double a21 =  - 0.5249167357886829;
		double a22 =  - 0.7144983957840794;
		double a23 = 0.5832981939848182;
		double a24 = 0.6747270495592943;
		double a25 = 0.04258038995809657;
		double a26 =  - 0.0023671529988773224;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_18_Diagonals() {
	double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 7.382379779647204;
		double gamma02 = 6.118391304701528;
		double gamma10 = 0.06106149690079044;
		double gamma12 = 2.6424801172618326;
		double gamma13 = 1.3710069602113883;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999154;
		double gamma23 = 1.110302482205355;
		double gamma24 = 0.3070536064807964;
		double a00 =  - 3.4387560411379563;
		double a01 =  - 6.120972374567166;
		double a02 = 7.819283685673539;
		double a03 = 2.0327058980606223;
		double a04 =  - 0.3578340204155545;
		double a05 = 0.07704263923889951;
		double a06 =  - 0.011469786857726549;
		double a10 =  - 0.25047387349211636;
		double a11 =  - 1.7715514547125182;
		double a12 =  - 0.5198462724609013;
		double a13 = 2.2536920432047496;
		double a14 = 0.318588110606521;
		double a15 =  - 0.0328153209921653;
		double a16 = 0.0024067678049133558;
		double a20 =  - 0.058823348930518396;
		double a21 =  - 0.5249167357886487;
		double a22 =  - 0.714498395784144;
		double a23 = 0.5832981939847537;
		double a24 = 0.6747270495592782;
		double a25 = 0.0425803899581001;
		double a26 =  - 0.0023671529988773697;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createA4_19_Diagonals() {
		double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 7.382379779647204;
		double gamma02 = 6.118391304701528;
		double gamma10 = 0.053558769421507044;
		double gamma12 = 2.843009383880429;
		double gamma13 = 1.6480384645594641;
		double gamma20 = 0.014504838045567564;
		double gamma21 = 0.2617273578999154;
		double gamma23 = 1.110302482205355;
		double gamma24 = 0.3070536064807964;
		double a00 =  - 3.4387560411379563;
		double a01 =  - 6.120972374567166;
		double a02 = 7.819283685673539;
		double a03 = 2.0327058980606223;
		double a04 =  - 0.3578340204155545;
		double a05 = 0.07704263923889951;
		double a06 =  - 0.011469786857726549;
		double a10 =  - 0.2311615287844447;
		double a11 =  - 1.8507243442530117;
		double a12 =  - 0.7937803097614184;
		double a13 = 2.470592586730826;
		double a14 = 0.46036823621123646;
		double a15 =  - 0.061408717577990914;
		double a16 = 0.006114077338592431;
		double a20 =  - 0.058823348930518396;
		double a21 =  - 0.5249167357886487;
		double a22 =  - 0.714498395784144;
		double a23 = 0.5832981939847537;
		double a24 = 0.6747270495592782;
		double a25 = 0.0425803899581001;
		double a26 =  - 0.0023671529988773697;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
	MatrixDiagonalEntries* createB4_1_Diagonals() {
	double alpha0 = 0.5942348163052151;
		double alpha = alpha0 ;

		double beta = 0.10275748724572716;
		double a1 = 0.6340472486307104;
		double a2 = 0.26726592813302824;
		double a3 = 0.01004721696654148;
		double a4 =  - 0.000432113061362272;
		double gamma01 = 3.258552321591469;
		double gamma02 = 3.9713593168524697;
		double gamma10 = 0.06191926485496031;
		double gamma12 = 2.0240678742461267;
		double gamma13 = 0.7927070599812164;
		double gamma20 =  - 0.049675740964988974;
		double gamma21 = 0.41403683368328115;
		double gamma23 = 0.557331401383541;
		double gamma24 = 0.0475762903159798;
		double gamma31 = 0.06152352492601939;
		double gamma32 = 0.4513445460734534;
		double gamma34 = 0.7364421712048806;
		double gamma35 = 0.14994836521298124;
		double a00 =  - 1.1516922313137914;
		double a01 =  - 3.9563373394376447;
		double a02 = 3.5592032994641367;
		double a03 = 1.8924696303209219;
		double a04 =  - 0.4222724838806715;
		double a05 = 0.08906811076020506;
		double a06 =  - 0.009529693586625327;
		double a07 =  - 0.0013840014549392028;
		double a08 = 0.0004747091933148014;
		double a10 =  - 0.30084732114693374;
		double a11 =  - 1.5148170310304807;
		double a12 = 0.15314297682362069;
		double a13 = 1.5422949488010151;
		double a14 = 0.12268287511032198;
		double a15 = 0.001965655478423474;
		double a16 =  - 0.006571656384717709;
		double a17 = 0.0025969778309034844;
		double a18 =  - 0.0004474254894815129;
		double a20 =  - 0.1134399758555121;
		double a21 =  - 0.6853799451553532;
		double a22 =  - 0.15180049288285746;
		double a23 = 0.7652655700431873;
		double a24 = 0.1860662489110366;
		double a25 =  - 0.00021508262326884955;
		double a26 =  - 0.0009294994767801919;
		double a27 = 0.0005386721242711686;
		double a28 =  - 0.00010549508480059121;
		double a30 =  - 0.007648819480336699;
		double a31 =  - 0.16596155150300815;
		double a32 =  - 0.6126045501032937;
		double a33 =  - 0.24829934932465766;
		double a34 = 0.6512334472074195;
		double a35 = 0.36673578848776456;
		double a36 = 0.017642516808049794;
		double a37 =  - 0.0011665688689049812;
		double a38 = 0.00006908677697936095;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a4, -a3, -a2, -a1, 0.0, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
	MatrixDiagonalEntries* createB4_2_Diagonals() {
		double alpha0 = 0.5942348163052151;
		double alpha = alpha0 ;

		double beta = 0.10275748724572716;
		double a1 = 0.6340472486307104;
		double a2 = 0.26726592813302824;
		double a3 = 0.01004721696654148;
		double a4 =  - 0.000432113061362272;
		double gamma01 =  - 3.2302485255282924;
		double gamma02 =  - 3.8740710583603564;
		double gamma10 = 0.034553312591525796;
		double gamma12 = 2.2932350583595316;
		double gamma13 = 1.0868840184220743;
		double gamma20 =  - 0.049675740964988974;
		double gamma21 = 0.41403683368328115;
		double gamma23 = 0.557331401383541;
		double gamma24 = 0.0475762903159798;
		double gamma31 = 0.06817786077706668;
		double gamma32 = 0.45657677163676796;
		double gamma34 = 0.7669552204799124;
		double gamma35 = 0.16738821761628633;
		double a00 = 1.247434332017422;
		double a01 = 3.649601749789092;
		double a02 =  - 3.2437101950576603;
		double a03 =  - 2.140663098848015;
		double a04 = 0.6845448723757406;
		double a05 =  - 0.27827838081944706;
		double a06 = 0.10851949533457124;
		double a07 =  - 0.03257912317463041;
		double a08 = 0.0051303502327167405;
		double a10 =  - 0.28054224569703096;
		double a11 =  - 1.6192847312096472;
		double a12 =  - 0.16916653559434483;
		double a13 = 1.844706742431158;
		double a14 = 0.24897027957135032;
		double a15 =  - 0.025994549064399283;
		double a16 = 0.00047165522700310254;
		double a17 = 0.0011017597543377866;
		double a18 =  - 0.00026237541739589055;
		double a20 =  - 0.1134399758555121;
		double a21 =  - 0.6853799451553532;
		double a22 =  - 0.15180049288285746;
		double a23 = 0.7652655700431873;
		double a24 = 0.1860662489110366;
		double a25 =  - 0.00021508262326884955;
		double a26 =  - 0.0009294994767801919;
		double a27 = 0.0005386721242711686;
		double a28 =  - 0.00010549508480059121;
		double a30 =  - 0.009799026774407651;
		double a31 =  - 0.17433707805796222;
		double a32 =  - 0.5984475804837898;
		double a33 =  - 0.2671267567857216;
		double a34 = 0.6333386471959103;
		double a35 = 0.3964863153342717;
		double a36 = 0.021300842819348294;
		double a37 =  - 0.0015087751353093529;
		double a38 = 0.00009341188766025977;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a4, -a3, -a2, -a1, 0.0, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createC4_1_Diagonals() {
		double alpha0 = 0.5362930888943283;
		double alpha = alpha0;

		double beta = 0.05831696520179083;
		double a1 = 0.6899191224266673;
		double a2 = 0.20234546583472587;
		double gamma01 = 5.887245740617931;
		double gamma02 = 3.4740401606206683;
		double gamma10 = 0.0481407122009377;
		double gamma12 = 1.7713756159896565;
		double gamma13 = 0.4570840961794273;
		double a00 =  - 3.26564142176947;
		double a01 =  - 3.222064890928744;
		double a02 = 5.830868610926915;
		double a03 = 0.705737236775412;
		double a04 =  - 0.04889953500022839;
		double a10 =  - 0.24076885710110105;
		double a11 =  - 1.593145513766301;
		double a12 = 0.6699517191280673;
		double a13 = 1.1260081070772001;
		double a14 = 0.03795454466215297;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2, -a1, 0.0, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}

MatrixDiagonalEntries* createC4_2_Diagonals() {
		double alpha0 = 0.5362930888943283;
		double alpha = alpha0;

		double beta = 0.05831696520179083;
		double a1 = 0.6899191224266673;
		double a2 = 0.20234546583472587;
		double gamma01 = 6.055735035499609;
		double gamma02 = 3.270740467439099;
		double gamma10 = 0.0481407122009377;
		double gamma12 = 1.7713756159896565;
		double gamma13 = 0.4570840961794273;
		double a00 =  - 3.324705386587386;
		double a01 =  - 3.226939507879095;
		double a02 = 6.083602553253239;
		double a03 = 0.4859594605444292;
		double a04 =  - 0.017917119329313384;
		double a10 =  - 0.24076885710110105;
		double a11 =  - 1.593145513766301;
		double a12 = 0.6699517191280673;
		double a13 = 1.1260081070772001;
		double a14 = 0.03795454466215297;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2, -a1, 0.0, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createC4_3_Diagonals() {
double alpha0 = 0.5362930888943283;
		double alpha = alpha0;

		double beta = 0.05831696520179083;
		double a1 = 0.6899191224266673;
		double a2 = 0.20234546583472587;
		double gamma01 = 7.235122100310267;
		double gamma02 = 3.337128838451273;
		double gamma10 = 0.0481407122009377;
		double gamma12 = 1.7713756159896565;
		double gamma13 = 0.4570840961794273;
		double a00 =  - 3.6140197885357606;
		double a01 =  - 4.254020976098135;
		double a02 = 7.852683150645544;
		double a03 =  - 0.05947515769874682;
		double a04 = 0.07483277178245445;
		double a10 =  - 0.24076885710110105;
		double a11 =  - 1.593145513766301;
		double a12 = 0.6699517191280673;
		double a13 = 1.1260081070772001;
		double a14 = 0.03795454466215297;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2, -a1, 0.0, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createC4_4_Diagonals() {
double alpha0 = 0.5362930888943283;
		double alpha = alpha0;

		double beta = 0.05831696520179083;
		double a1 = 0.6899191224266673;
		double a2 = 0.20234546583472587;
		double gamma01 = 6.055735035499609;
		double gamma02 = 3.270740467439099;
		double gamma10 = 0.055125777588289925;
		double gamma12 = 1.6545638752817395;
		double gamma13 = 0.32970265502925983;
		double a00 =  - 3.324705386587386;
		double a01 =  - 3.226939507879095;
		double a02 = 6.083602553253239;
		double a03 = 0.4859594605444292;
		double a04 =  - 0.017917119329313384;
		double a10 =  - 0.2544402682879356;
		double a11 =  - 1.5510214789869237;
		double a12 = 0.8400686846907492;
		double a13 = 0.9512958328302001;
		double a14 = 0.01409722975341476;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2, -a1, 0.0, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* createC4_5_Diagonals() {
		double alpha0 = 0.5362930888943283;
		double alpha = alpha0;

		double beta = 0.05831696520179083;
		double a1 = 0.6899191224266673;
		double a2 = 0.20234546583472587;
		double gamma01 = 7.235122100310267;
		double gamma02 = 3.337128838451273;
		double gamma10 = 0.055125777588289925;
		double gamma12 = 1.6545638752817395;
		double gamma13 = 0.32970265502925983;
		double a00 =  - 3.6140197885357606;
		double a01 =  - 4.254020976098135;
		double a02 = 7.852683150645544;
		double a03 =  - 0.05947515769874682;
		double a04 = 0.07483277178245445;
		double a10 =  - 0.2544402682879356;
		double a11 =  - 1.5510214789869237;
		double a12 = 0.8400686846907492;
		double a13 = 0.9512958328302001;
		double a14 = 0.01409722975341476;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a2, -a1, 0.0, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
	MatrixDiagonalEntries* createA6_1_Diagonals() {
		double alpha0 = 0.5573223518365282;
		double alpha = alpha0;

		double beta = 0.076203478481807;
		double a1 = 0.6694377318500324;
		double a2 = 0.22598218207734894;
		double a3 = 0.0040412447712016904;
		double gamma01 = 9.288564149070755;
		double gamma02 = 10.424083510062767;
		double gamma10 =  - 0.5351188365509763;
		double gamma12 = 17.591679228204764;
		double gamma13 = 13.339737108948365;
		double gamma20 = 0.012387685030313872;
		double gamma21 = 0.30280422035275817;
		double gamma23 = 0.6842610550203767;
		double gamma24 = 0.08589862906533306;
		double a00 =  - 3.6506245745105415;
		double a01 =  - 10.089957395330378;
		double a02 = 9.640694991800002;
		double a03 = 5.084504431624974;
		double a04 =  - 1.2215716307922506;
		double a05 = 0.26773676407201624;
		double a06 =  - 0.03078258686510679;
		double a10 = 1.5084348386745443;
		double a11 =  - 9.529757477583644;
		double a12 =  - 13.753224440694218;
		double a13 = 18.2214467272648;
		double a14 = 4.04899218801074;
		double a15 =  - 0.5475459397764549;
		double a16 = 0.05165410410427152;
		double a20 =  - 0.057456572148986036;
		double a21 =  - 0.6230866315598078;
		double a22 =  - 0.389476896912155;
		double a23 = 0.7967126941942263;
		double a24 = 0.2691863563169919;
		double a25 = 0.004217793654643562;
		double a26 =  - 0.0000967435451386881;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
		MatrixDiagonalEntries* createA6_2_Diagonals() {
		double alpha0 = 0.5573223518365282;
		double alpha = alpha0;

		double beta = 0.076203478481807;
		double a1 = 0.6694377318500324;
		double a2 = 0.22598218207734894;
		double a3 = 0.0040412447712016904;
		double gamma01 = 7.197716094824484;
		double gamma02 = 6.126551926460928;
		double gamma10 = 0.08918059126032331;
		double gamma12 = 1.5358275855335228;
		double gamma13 = 0.39888676733488315;
		double gamma20 = 0.014249184858449039;
		double gamma21 = 0.26879840403244193;
		double gamma23 = 1.0681671800782038;
		double gamma24 = 0.27905591015869174;
		double a00 =  - 3.445400951639534;
		double a01 =  - 5.687689758411866;
		double a02 = 6.920468274013684;
		double a03 = 2.8392090828531993;
		double a04 =  - 0.8151792182119252;
		double a05 = 0.21744456676079005;
		double a06 =  - 0.028851995651904917;
		double a10 =  - 0.3406129751922464;
		double a11 =  - 1.3027478048845684;
		double a12 = 0.6360810651517776;
		double a13 = 0.9756407224469644;
		double a14 = 0.030157398841492614;
		double a15 = 0.001960705816633491;
		double a16 =  - 0.0004791121800322922;
		double a20 =  - 0.0595287580739372;
		double a21 =  - 0.5364452203671104;
		double a22 =  - 0.6798036396699008;
		double a23 = 0.6082560121241458;
		double a24 = 0.6344722261258422;
		double a25 = 0.03463004120708508;
		double a26 =  - 0.001580661345977278;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}
		MatrixDiagonalEntries* createA6_3_Diagonals() {
double alpha0 = 0.5573223518365282;
		double alpha = alpha0;

		double beta = 0.076203478481807;
		double a1 = 0.6694377318500324;
		double a2 = 0.22598218207734894;
		double a3 = 0.0040412447712016904;
		double gamma01 = 10.344913263079071;
		double gamma02 = 12.31738667865771;
		double gamma10 = 0.05633022607373964;
		double gamma12 = 2.699693552673342;
		double gamma13 = 1.3822895775536894;
		double gamma20 = 0.014249184858449039;
		double gamma21 = 0.26879840403244193;
		double gamma23 = 1.0681671800782038;
		double gamma24 = 0.27905591015869174;
		double a00 =  - 3.7635726583218965;
		double a01 =  - 12.202926634464577;
		double a02 = 11.17714063816103;
		double a03 = 5.848326840829939;
		double a04 =  - 1.2879318527051586;
		double a05 = 0.25608981052230173;
		double a06 =  - 0.02712599235878713;
		double a10 =  - 0.2377240950694543;
		double a11 =  - 1.8178859613447598;
		double a12 =  - 0.5340151179060226;
		double a13 = 2.3084595775966275;
		double a14 = 0.3089653921493875;
		double a15 =  - 0.029788024933338528;
		double a16 = 0.00198822939611522;
		double a20 =  - 0.0595287580739372;
		double a21 =  - 0.5364452203671104;
		double a22 =  - 0.6798036396699008;
		double a23 = 0.6082560121241458;
		double a24 = 0.6344722261258422;
		double a25 = 0.03463004120708508;
		double a26 =  - 0.001580661345977278;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}

MatrixDiagonalEntries* createA6_4_Diagonals() {
double alpha0 = 0.5573223518365282;
		double alpha = alpha0;

		double beta = 0.076203478481807;
		double a1 = 0.6694377318500324;
		double a2 = 0.22598218207734894;
		double a3 = 0.0040412447712016904;
		double gamma01 = 9.288564149070755;
		double gamma02 = 10.424083510062767;
		double gamma10 = 0.09343773584279384;
		double gamma12 = 1.4376629777153611;
		double gamma13 = 0.2998018368521858;
		double gamma20 = 0.014249184858449039;
		double gamma21 = 0.26879840403244193;
		double gamma23 = 1.0681671800782038;
		double gamma24 = 0.27905591015869174;
		double a00 =  - 3.6506245745105415;
		double a01 =  - 10.089957395330378;
		double a02 = 9.640694991800002;
		double a03 = 5.084504431624974;
		double a04 =  - 1.2215716307922506;
		double a05 = 0.26773676407201624;
		double a06 =  - 0.03078258686510679;
		double a10 =  - 0.35266371750520614;
		double a11 =  - 1.2528018338347733;
		double a12 = 0.7357288665391961;
		double a13 = 0.8731355425724548;
		double a14 =  - 0.011038287295735444;
		double a15 = 0.008843404512244612;
		double a16 =  - 0.0012039749881859441;
		double a20 =  - 0.0595287580739372;
		double a21 =  - 0.5364452203671104;
		double a22 =  - 0.6798036396699008;
		double a23 = 0.6082560121241458;
		double a24 = 0.6344722261258422;
		double a25 = 0.03463004120708508;
		double a26 =  - 0.001580661345977278;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* createA6_5_Diagonals() {
		double alpha0 = 0.5573223518365282;
		double alpha = alpha0 ;

		double beta = 0.076203478481807;
		double a1 = 0.6694377318500324;
		double a2 = 0.22598218207734894;
		double a3 = 0.0040412447712016904;
		double gamma01 = 9.288564149070755;
		double gamma02 = 10.424083510062767;
		double gamma10 = 0.08451369984437408;
		double gamma12 = 1.5122958103182786;
		double gamma13 = 0.34083648807424693;
		double gamma20 = 0.014249184858449039;
		double gamma21 = 0.26879840403244193;
		double gamma23 = 1.0681671800782038;
		double gamma24 = 0.27905591015869174;
		double a00 =  - 3.6506245745105415;
		double a01 =  - 10.089957395330378;
		double a02 = 9.640694991800002;
		double a03 = 5.084504431624974;
		double a04 =  - 1.2215716307922506;
		double a05 = 0.26773676407201624;
		double a06 =  - 0.03078258686510679;
		double a10 =  - 0.3289959790758886;
		double a11 =  - 1.330043985184392;
		double a12 = 0.728347329090123;
		double a13 = 0.913152412730422;
		double a14 = 0.015886419808613456;
		double a15 = 0.0019304079788664582;
		double a16 =  - 0.0002766053449263889;
		double a20 =  - 0.0595287580739372;
		double a21 =  - 0.5364452203671104;
		double a22 =  - 0.6798036396699008;
		double a23 = 0.6082560121241458;
		double a24 = 0.6344722261258422;
		double a25 = 0.03463004120708508;
		double a26 =  - 0.001580661345977278;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* createA6_6_Diagonals() {
	double alpha0 = 0.5573223518365282;
		double alpha = alpha0;

		double beta = 0.076203478481807;
		double a1 = 0.6694377318500324;
		double a2 = 0.22598218207734894;
		double a3 = 0.0040412447712016904;
		double gamma01 = 10.344913263079071;
		double gamma02 = 12.31738667865771;
		double gamma10 =  - 0.01088917605866635;
		double gamma12 = 4.605108734192861;
		double gamma13 = 2.6688808066274703;
		double gamma20 = 0.014249184858449039;
		double gamma21 = 0.26879840403244193;
		double gamma23 = 1.0681671800782038;
		double gamma24 = 0.27905591015869174;
		double a00 =  - 3.7635726583218965;
		double a01 =  - 12.202926634464577;
		double a02 = 11.17714063816103;
		double a03 = 5.848326840829939;
		double a04 =  - 1.2879318527051586;
		double a05 = 0.25608981052230173;
		double a06 =  - 0.02712599235878713;
		double a10 =  - 0.03096590762567285;
		double a11 =  - 2.790379762388547;
		double a12 =  - 2.1063052128530435;
		double a13 = 4.400883805309275;
		double a14 = 0.5732739813825203;
		double a15 =  - 0.04938463436134828;
		double a16 = 0.0028777305489861094;
		double a20 =  - 0.0595287580739372;
		double a21 =  - 0.5364452203671104;
		double a22 =  - 0.6798036396699008;
		double a23 = 0.6082560121241458;
		double a24 = 0.6344722261258422;
		double a25 = 0.03463004120708508;
		double a26 =  - 0.001580661345977278;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* createA6_7_Diagonals() {
double alpha0 = 0.5573223518365282;
		double alpha = alpha0;

		double beta = 0.076203478481807;
		double a1 = 0.6694377318500324;
		double a2 = 0.22598218207734894;
		double a3 = 0.0040412447712016904;
		double gamma01 = 10.344913263079071;
		double gamma02 = 12.31738667865771;
		double gamma10 = 0.05512040745115604;
		double gamma12 = 2.6702875425745463;
		double gamma13 = 1.383328787658446;
		double gamma20 = 0.020711717451278075;
		double gamma21 = 0.34572227444689885;
		double gamma23 = 0.7791310604917956;
		double gamma24 = 0.1487319658370784;
		double a00 =  - 3.7635726583218965;
		double a01 =  - 12.202926634464577;
		double a02 = 11.17714063816103;
		double a03 = 5.848326840829939;
		double a04 =  - 1.2879318527051586;
		double a05 = 0.25608981052230173;
		double a06 =  - 0.02712599235878713;
		double a10 =  - 0.23575755996227593;
		double a11 =  - 1.8132265875001963;
		double a12 =  - 0.5085673798124102;
		double a13 = 2.2611861064540553;
		double a14 = 0.32898462480697077;
		double a15 =  - 0.035316490190758505;
		double a16 = 0.0026972861754763437;
		double a20 =  - 0.08553740507435961;
		double a21 =  - 0.6223678838703601;
		double a22 =  - 0.3843478405509792;
		double a23 = 0.6968983711475849;
		double a24 = 0.3815415637039321;
		double a25 = 0.014379952924204267;
		double a26 =  - 0.0005667582800224041;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
}

//Second Derivatives



    MatrixDiagonalEntries* create2B4_1_Diagonals() {
double alpha0 = 0.5443196687355368;
		double alpha = alpha0;

		double beta = 0.07728851013567789;
		double a1 = 0.10624114531946292;
		double a2 = 0.44820542026997306;
		double a3 = 0.040353966028400544;
		double a4 =  - 0.0011895101820331873;
		double gamma01 =  - 9.08459826478248;
		double gamma02 =  - 25.71363251024003;
		double gamma10 =  - 0.133984019279111;
		double gamma12 = 2.4951036451754596;
		double gamma13 = 0.4123449516198014;
		double gamma20 =  - 0.02151917117828344;
		double gamma21 = 0.32098475801942616;
		double gamma23 = 0.5978810174357946;
		double gamma24 = 0.13363937320515867;
		double gamma31 = 0.06544469559131794;
		double gamma32 = 0.48963264996225114;
		double gamma34 = 0.5116877293377498;
		double gamma35 = 0.06704135028368141;
		double a00 =  - 6.6705826312289105;
		double a01 =  - 19.0528028978955;
		double a02 = 62.97350601697924;
		double a03 =  - 43.32893064487769;
		double a04 = 7.63935311361443;
		double a05 =  - 1.985506734444297;
		double a06 = 0.5265961795964231;
		double a07 =  - 0.11618615404780643;
		double a08 = 0.014553751968020763;
		double a10 = 0.8028940456973566;
		double a11 = 1.4551309417195069;
		double a12 =  - 5.261235243878468;
		double a13 = 2.9965856154362647;
		double a14 =  - 0.0663866171518484;
		double a15 = 0.10257699309312651;
		double a16 =  - 0.03879704736258897;
		double a17 = 0.010824012532857464;
		double a18 =  - 0.0015927000849847654;
		double a20 = 0.2428464429656088;
		double a21 = 0.6679974422958201;
		double a22 =  - 1.4901708753503622;
		double a23 = 0.10947313538836716;
		double a24 = 0.35489933574508953;
		double a25 = 0.12749921423072738;
		double a26 =  - 0.015007572943870444;
		double a27 = 0.0028260955807972513;
		double a28 =  - 0.0003632179120144587;
		double a30 = 0.034006449362369115;
		double a31 = 0.3850247295575769;
		double a32 = 0.2642667881567995;
		double a33 =  - 1.3387347356082686;
		double a34 = 0.20667213978114238;
		double a35 = 0.41545919420331817;
		double a36 = 0.034525284360193025;
		double a37 =  - 0.0012684631565034407;
		double a38 = 0.00004861334334597006;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2A6_1_Diagonals() {
double alpha0 = 0.4631165969478096;
		double alpha = alpha0 ;

		double beta = 0.0428502505546325;
		double a1 = 0.32943690520352964;
		double a2 = 0.39288484117098216;
		double a3 = 0.012328602790825066;
		double gamma01 = 14.227445502373572;
		double gamma02 = 15.250950263311985;
		double gamma10 = 0.041130293875330966;
		double gamma12 = 2.9087914858519386;
		double gamma13 = 0.9111720758280886;
		double gamma20 = 0.008405796421889166;
		double gamma21 = 0.23127448297326125;
		double gamma23 = 1.1387162951487462;
		double gamma24 = 0.2575555182261651;
		double a00 = 14.238320446778651;
		double a01 =  - 9.701210159760524;
		double a02 =  - 26.491098408639015;
		double a03 = 25.87271910358145;
		double a04 =  - 4.755876189185934;
		double a05 = 0.9341168252169539;
		double a06 =  - 0.09697161668346488;
		double a10 = 0.7466991859032148;
		double a11 = 2.015459623946592;
		double a12 =  - 5.6340275907583965;
		double a13 = 2.201900484034453;
		double a14 = 0.7044739198435027;
		double a15 =  - 0.03603216402565966;
		double a16 = 0.0015265410238673712;
		double a20 = 0.15723696938146278;
		double a21 = 0.7435538356768784;
		double a22 =  - 0.6855652362962593;
		double a23 =  - 1.335901776685687;
		double a24 = 0.9629559431156159;
		double a25 = 0.1628633943388043;
		double a26 =  - 0.005143129530533028;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;	
}
MatrixDiagonalEntries* create2A6_2_Diagonals() {
double alpha0 = 0.4631165969478096;
		double alpha = alpha0;

		double beta = 0.0428502505546325;
		double a1 = 0.32943690520352964;
		double a2 = 0.39288484117098216;
		double a3 = 0.012328602790825066;
		double gamma01 = 16.356735903163955;
		double gamma02 = 26.96204746741359;
		double gamma10 = 0.041130293875330966;
		double gamma12 = 2.9087914858519386;
		double gamma13 = 0.9111720758280886;
		double gamma20 = 0.008405796421889166;
		double gamma21 = 0.23127448297326125;
		double gamma23 = 1.1387162951487462;
		double gamma24 = 0.2575555182261651;
		double a00 = 15.013145564757906;
		double a01 = 3.393925804484586;
		double a02 =  - 56.833486620075405;
		double a03 = 44.44486315530521;
		double a04 =  - 7.151327891001162;
		double a05 = 1.253510385466834;
		double a06 =  - 0.1206303989235141;
		double a10 = 0.7466991859032148;
		double a11 = 2.015459623946592;
		double a12 =  - 5.6340275907583965;
		double a13 = 2.201900484034453;
		double a14 = 0.7044739198435027;
		double a15 =  - 0.03603216402565966;
		double a16 = 0.0015265410238673712;
		double a20 = 0.15723696938146278;
		double a21 = 0.7435538356768784;
		double a22 =  - 0.6855652362962593;
		double a23 =  - 1.335901776685687;
		double a24 = 0.9629559431156159;
		double a25 = 0.1628633943388043;
		double a26 =  - 0.005143129530533028;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2A6_3_Diagonals() {
double alpha0 = 0.4631165969478096;
		double alpha = alpha0;

		double beta = 0.0428502505546325;
		double a1 = 0.32943690520352964;
		double a2 = 0.39288484117098216;
		double a3 = 0.012328602790825066;
		double gamma01 = 14.227445502373572;
		double gamma02 = 15.250950263311985;
		double gamma10 = 0.038346862830425196;
		double gamma12 = 3.084147641680087;
		double gamma13 = 0.9988110965651857;
		double gamma20 = 0.008405796421889166;
		double gamma21 = 0.23127448297326125;
		double gamma23 = 1.1387162951487462;
		double gamma24 = 0.2575555182261651;
		double a00 = 14.238320446778651;
		double a01 =  - 9.701210159760524;
		double a02 =  - 26.491098408639015;
		double a03 = 25.87271910358145;
		double a04 =  - 4.755876189185934;
		double a05 = 0.9341168252169539;
		double a06 =  - 0.09697161668346488;
		double a10 = 0.7224519748301444;
		double a11 = 2.2728632683898398;
		double a12 =  - 5.9931487812474815;
		double a13 = 2.236722376901147;
		double a14 = 0.8046188516876706;
		double a15 =  - 0.04583789988015513;
		double a16 = 0.0023302093569763536;
		double a20 = 0.15723696938146278;
		double a21 = 0.7435538356768784;
		double a22 =  - 0.6855652362962593;
		double a23 =  - 1.335901776685687;
		double a24 = 0.9629559431156159;
		double a25 = 0.1628633943388043;
		double a26 =  - 0.005143129530533028;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2A6_4_Diagonals() {
double alpha0 = 0.4631165969478096;
		double alpha = alpha0;

		double beta = 0.0428502505546325;
		double a1 = 0.32943690520352964;
		double a2 = 0.39288484117098216;
		double a3 = 0.012328602790825066;
		double gamma01 = 16.256649303307295;
		double gamma02 = 26.411571167985016;
		double gamma10 = 0.041130293875330966;
		double gamma12 = 2.9087914858519386;
		double gamma13 = 0.9111720758280886;
		double gamma20 = 0.008405796421889166;
		double gamma21 = 0.23127448297326125;
		double gamma23 = 1.1387162951487462;
		double gamma24 = 0.2575555182261651;
		double a00 = 14.976725163243854;
		double a01 = 2.778393215919138;
		double a02 =  - 55.407252572816844;
		double a03 = 43.57188558881627;
		double a04 =  - 7.0387304650283005;
		double a05 = 1.2384973952870615;
		double a06 =  - 0.11951832557873347;
		double a10 = 0.7466991859032148;
		double a11 = 2.015459623946592;
		double a12 =  - 5.6340275907583965;
		double a13 = 2.201900484034453;
		double a14 = 0.7044739198435027;
		double a15 =  - 0.03603216402565966;
		double a16 = 0.0015265410238673712;
		double a20 = 0.15723696938146278;
		double a21 = 0.7435538356768784;
		double a22 =  - 0.6855652362962593;
		double a23 =  - 1.335901776685687;
		double a24 = 0.9629559431156159;
		double a25 = 0.1628633943388043;
		double a26 =  - 0.005143129530533028;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2A6_5_Diagonals() {
double alpha0 = 0.4631165969478096;
		double alpha = alpha0;

		double beta = 0.0428502505546325;
		double a1 = 0.32943690520352964;
		double a2 = 0.39288484117098216;
		double a3 = 0.012328602790825066;
		double gamma01 = 14.287799533931434;
		double gamma02 = 15.582897659969301;
		double gamma10 = 0.038346862830425196;
		double gamma12 = 3.084147641680087;
		double gamma13 = 0.9988110965651857;
		double gamma20 = 0.008405796421889166;
		double gamma21 = 0.23127448297326125;
		double gamma23 = 1.1387162951487462;
		double gamma24 = 0.2575555182261651;
		double a00 = 14.260282613166723;
		double a01 =  - 9.330032725985884;
		double a02 =  - 27.35114372471562;
		double a03 = 26.39914082606353;
		double a04 =  - 4.823774627817952;
		double a05 = 0.9431699573462978;
		double a06 =  - 0.09764221858928672;
		double a10 = 0.7224519748301444;
		double a11 = 2.2728632683898398;
		double a12 =  - 5.9931487812474815;
		double a13 = 2.236722376901147;
		double a14 = 0.8046188516876706;
		double a15 =  - 0.04583789988015513;
		double a16 = 0.0023302093569763536;
		double a20 = 0.15723696938146278;
		double a21 = 0.7435538356768784;
		double a22 =  - 0.6855652362962593;
		double a23 =  - 1.335901776685687;
		double a24 = 0.9629559431156159;
		double a25 = 0.1628633943388043;
		double a26 =  - 0.005143129530533028;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2A6_6_Diagonals() {
double alpha0 = 0.4631165969478096;
		double alpha = alpha0;

		double beta = 0.0428502505546325;
		double a1 = 0.32943690520352964;
		double a2 = 0.39288484117098216;
		double a3 = 0.012328602790825066;
		double gamma01 = 16.364231401157664;
		double gamma02 = 27.003272528217813;
		double gamma10 = 0.041130293875330966;
		double gamma12 = 2.9087914858519386;
		double gamma13 = 0.9111720758280886;
		double gamma20 = 0.008405796421889166;
		double gamma21 = 0.23127448297326125;
		double gamma23 = 1.1387162951487462;
		double gamma24 = 0.2575555182261651;
		double a00 = 15.015873088900591;
		double a01 = 3.440023001825902;
		double a02 =  - 56.9402971723096;
		double a03 = 44.5102402017398;
		double a04 =  - 7.159760210093785;
		double a05 = 1.2546346894840152;
		double a06 =  - 0.12071368106522763;
		double a10 = 0.7466991859032148;
		double a11 = 2.015459623946592;
		double a12 =  - 5.6340275907583965;
		double a13 = 2.201900484034453;
		double a14 = 0.7044739198435027;
		double a15 =  - 0.03603216402565966;
		double a16 = 0.0015265410238673712;
		double a20 = 0.15723696938146278;
		double a21 = 0.7435538356768784;
		double a22 =  - 0.6855652362962593;
		double a23 =  - 1.335901776685687;
		double a24 = 0.9629559431156159;
		double a25 = 0.1628633943388043;
		double a26 =  - 0.005143129530533028;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2A6_7_Diagonals() {
double alpha0 = 0.4631165969478096;
		double alpha = alpha0;

		double beta = 0.0428502505546325;
		double a1 = 0.32943690520352964;
		double a2 = 0.39288484117098216;
		double a3 = 0.012328602790825066;
		double gamma01 = 16.364231401157664;
		double gamma02 = 27.003272528217813;
		double gamma10 = 0.038346862830425196;
		double gamma12 = 3.084147641680087;
		double gamma13 = 0.9988110965651857;
		double gamma20 = 0.008405796421889166;
		double gamma21 = 0.23127448297326125;
		double gamma23 = 1.1387162951487462;
		double gamma24 = 0.2575555182261651;
		double a00 = 15.015873088900591;
		double a01 = 3.440023001825902;
		double a02 =  - 56.9402971723096;
		double a03 = 44.5102402017398;
		double a04 =  - 7.159760210093785;
		double a05 = 1.2546346894840152;
		double a06 =  - 0.12071368106522763;
		double a10 = 0.7224519748301444;
		double a11 = 2.2728632683898398;
		double a12 =  - 5.9931487812474815;
		double a13 = 2.236722376901147;
		double a14 = 0.8046188516876706;
		double a15 =  - 0.04583789988015513;
		double a16 = 0.0023302093569763536;
		double a20 = 0.15723696938146278;
		double a21 = 0.7435538356768784;
		double a22 =  - 0.6855652362962593;
		double a23 =  - 1.335901776685687;
		double a24 = 0.9629559431156159;
		double a25 = 0.1628633943388043;
		double a26 =  - 0.005143129530533028;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

    MatrixDiagonalEntries* create2B6_1_Diagonals() {
    double alpha0 = 0.5303539471149911;
		double alpha = alpha0;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 =  - 10.966637367978482;
		double gamma02 =  - 30.304418061370452;
		double gamma10 =  - 0.11492362311841817;
		double gamma12 = 2.6182545028668254;
		double gamma13 = 0.1554165882805328;
		double gamma20 =  - 0.019197909225382476;
		double gamma21 = 0.300733077029695;
		double gamma23 = 0.6117125584008577;
		double gamma24 = 0.16027058561285706;
		double gamma31 = 0.06501691728096612;
		double gamma32 = 0.490673782347286;
		double gamma34 = 0.5075650430038768;
		double gamma35 = 0.06491104602937041;
		double a00 =  - 8.123817706123804;
		double a01 =  - 21.971305632575913;
		double a02 = 74.10137118963104;
		double a03 =  - 51.28631096526587;
		double a04 = 9.191367184308135;
		double a05 =  - 2.4272201286673427;
		double a06 = 0.6256846563520914;
		double a07 =  - 0.12216889784772732;
		double a08 = 0.012400300281663666;
		double a10 = 0.76717381091414;
		double a11 = 1.7404167443125544;
		double a12 =  - 6.087873305074748;
		double a13 = 3.9934420448156502;
		double a14 =  - 0.5554175934084806;
		double a15 = 0.18583506488460344;
		double a16 =  - 0.0537678037088644;
		double a17 = 0.011422695284901046;
		double a18 =  - 0.0012316580159781076;
		double a20 = 0.2179990997373567;
		double a21 = 0.7185740245929354;
		double a22 =  - 1.5028916509117003;
		double a23 = 0.11391053987056853;
		double a24 = 0.30236931317313814;
		double a25 = 0.166674067211463;
		double a26 =  - 0.01936699035708396;
		double a27 = 0.0030207624554990193;
		double a28 =  - 0.0002891657721055215;
		double a30 = 0.03335693518984774;
		double a31 = 0.38750825957584134;
		double a32 = 0.2618795518216098;
		double a33 =  - 1.3439223076658227;
		double a34 = 0.2156185492609101;
		double a35 = 0.4143156021927101;
		double a36 = 0.032291932058388144;
		double a37 =  - 0.0010839388887478403;
		double a38 = 0.000035416455269440454;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2B6_2_Diagonals() {
	double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 =  - 10.966637367978482;
		double gamma02 =  - 30.304418061370452;
		double gamma10 = 0.007231008320535141;
		double gamma12 = 2.69425263036507;
		double gamma13 = 0.28941109404967313;
		double gamma20 =  - 0.01213077541170243;
		double gamma21 = 0.2725716635030012;
		double gamma23 = 0.7334839314641245;
		double gamma24 = 0.23782962429071505;
		double gamma31 = 0.07160180980823996;
		double gamma32 = 0.5105175853574312;
		double gamma34 = 0.4870081066438877;
		double gamma35 = 0.05887220941193474;
		double a00 =  - 8.123817706123804;
		double a01 =  - 21.971305632575913;
		double a02 = 74.10137118963104;
		double a03 =  - 51.28631096526587;
		double a04 = 9.191367184308135;
		double a05 =  - 2.4272201286673427;
		double a06 = 0.6256846563520914;
		double a07 =  - 0.12216889784772732;
		double a08 = 0.012400300281663666;
		double a10 = 0.7602547688812334;
		double a11 = 1.8261158850963042;
		double a12 =  - 6.074746092625187;
		double a13 = 3.7134996435879426;
		double a14 =  - 0.3371205862382884;
		double a15 = 0.1481979635854135;
		double a16 =  - 0.04483513637453837;
		double a17 = 0.009687719253295729;
		double a18 =  - 0.0010541649061336703;
		double a20 = 0.18977546054073308;
		double a21 = 0.7543810716464031;
		double a22 =  - 1.3392824839428328;
		double a23 =  - 0.13460170960584616;
		double a24 = 0.30028333291609516;
		double a25 = 0.2527966156458421;
		double a26 =  - 0.026641113879392317;
		double a27 = 0.0035913150024260857;
		double a28 =  - 0.00030248832141958654;
		double a30 = 0.03817919596469591;
		double a31 = 0.40200717892753707;
		double a32 = 0.21853881346376297;
		double a33 =  - 1.3451767097053786;
		double a34 = 0.2619989200086989;
		double a35 = 0.3974653278525915;
		double a36 = 0.027810985676208752;
		double a37 =  - 0.0008467874112833383;
		double a38 = 0.000023075223112788605;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2B6_3_Diagonals() {
	double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 =  - 10.966637367978482;
		double gamma02 =  - 30.304418061370452;
		double gamma10 =  - 0.038569791275243764;
		double gamma12 = 2.7338695476690758;
		double gamma13 = 0.30818307414877355;
		double gamma20 =  - 0.019197909225382476;
		double gamma21 = 0.300733077029695;
		double gamma23 = 0.6117125584008577;
		double gamma24 = 0.16027058561285706;
		double gamma31 = 0.06801243933070299;
		double gamma32 = 0.4799578622948373;
		double gamma34 = 0.6679373011705586;
		double gamma35 = 0.12057646664909265;
		double a00 =  - 8.123817706123804;
		double a01 =  - 21.971305632575913;
		double a02 = 74.10137118963104;
		double a03 =  - 51.28631096526587;
		double a04 = 9.191367184308135;
		double a05 =  - 2.4272201286673427;
		double a06 = 0.6256846563520914;
		double a07 =  - 0.12216889784772732;
		double a08 = 0.012400300281663666;
		double a10 = 0.7550916092741523;
		double a11 = 1.8827628299899253;
		double a12 =  - 6.153279351664581;
		double a13 = 3.7176032968353927;
		double a14 =  - 0.3090554759859175;
		double a15 = 0.14117344509279187;
		double a16 =  - 0.04239825525723455;
		double a17 = 0.009085011132319977;
		double a18 =  - 0.0009831114313225144;
		double a20 = 0.2179990997373567;
		double a21 = 0.7185740245929354;
		double a22 =  - 1.5028916509117003;
		double a23 = 0.11391053987056853;
		double a24 = 0.30236931317313814;
		double a25 = 0.166674067211463;
		double a26 =  - 0.01936699035708396;
		double a27 = 0.0030207624554990193;
		double a28 =  - 0.0002891657721055215;
		double a30 = 0.0366460018335268;
		double a31 = 0.37495055545497974;
		double a32 = 0.24758315585792942;
		double a33 =  - 1.0876644418241608;
		double a34 =  - 0.1805752738192111;
		double a35 = 0.5360082630861024;
		double a36 = 0.07637866273240493;
		double a37 =  - 0.0034796483264718544;
		double a38 = 0.00015272500481513978;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	
}
MatrixDiagonalEntries* create2B6_4_Diagonals() {
		double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 =  - 10.966637367978482;
		double gamma02 =  - 30.304418061370452;
		double gamma10 =  - 0.08357015554229973;
		double gamma12 = 2.7268901310261704;
		double gamma13 = 0.2601415282632231;
		double gamma20 = 0.000023410102770690376;
		double gamma21 = 0.1449635317301256;
		double gamma23 = 1.7971071968105605;
		double gamma24 = 0.5693342620125411;
		double gamma31 = 0.08075867224970161;
		double gamma32 = 0.556308421480436;
		double gamma34 = 0.32018448581760395;
		double gamma35 = 0.0011602385158419003;
		double a00 =  - 8.123817706123804;
		double a01 =  - 21.971305632575913;
		double a02 = 74.10137118963104;
		double a03 =  - 51.28631096526587;
		double a04 = 9.191367184308135;
		double a05 =  - 2.4272201286673427;
		double a06 = 0.6256846563520914;
		double a07 =  - 0.12216889784772732;
		double a08 = 0.012400300281663666;
		double a10 = 0.754546880304877;
		double a11 = 1.8836238663929987;
		double a12 =  - 6.214797932053918;
		double a13 = 3.8487615400902846;
		double a14 =  - 0.38920054533414544;
		double a15 = 0.15328967795591789;
		double a16 =  - 0.044610333319912786;
		double a17 = 0.009391608708114872;
		double a18 =  - 0.0010047607040897487;
		double a20 = 0.07843448919007862;
		double a21 = 0.7925421816736662;
		double a22 = 0.26766049193215696;
		double a23 =  - 2.80138218604522;
		double a24 = 1.2085547124440164;
		double a25 = 0.48588237304344334;
		double a26 =  - 0.03469540701413635;
		double a27 = 0.0032016385198230914;
		double a28 =  - 0.00019829374359747512;
		double a30 = 0.04373725032368315;
		double a31 = 0.4408046760211261;
		double a32 = 0.14843564462720454;
		double a33 =  - 1.5532900267938194;
		double a34 = 0.6641132677516352;
		double a35 = 0.27297113142637586;
		double a36 =  - 0.018397013912867757;
		double a37 = 0.0017330940089714736;
		double a38 =  - 0.00010802345222657461;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2B6_5_Diagonals() {
		double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 = 46.27220177429061;
		double gamma02 = 123.64634674052958;
		double gamma10 =  - 0.10947926470598214;
		double gamma12 = 2.725756773745823;
		double gamma13 = 0.4071177468005932;
		double gamma20 =  - 0.049531656854435654;
		double gamma21 =  - 0.557268653525614;
		double gamma23 = 7.781498998314866;
		double gamma24 = 2.5315734460296517;
		double gamma31 = 0.08075867224970161;
		double gamma32 = 0.556308421480436;
		double gamma34 = 0.32018448581760395;
		double gamma35 = 0.0011602385158419003;
		double a00 = 34.90837318296423;
		double a01 = 86.10560029621546;
		double a02 =  - 300.72515906087415;
		double a03 = 209.74185430260437;
		double a04 =  - 38.06695893701957;
		double a05 = 10.259989871727536;
		double a06 =  - 2.708953509473865;
		double a07 = 0.5412688572989648;
		double a08 =  - 0.05601492978284683;
		double a10 = 0.7595063048226752;
		double a11 = 1.8461936891429334;
		double a12 =  - 5.963967311560987;
		double a13 = 3.413677613911776;
		double a14 =  - 0.14200596175206215;
		double a15 = 0.117644456500634;
		double a16 =  - 0.038829570286761705;
		double a17 = 0.008759156570699917;
		double a18 =  - 0.0009783770917945432;
		double a20 =  - 0.5300773048821206;
		double a21 = 0.9738248911885589;
		double a22 = 9.303852745029829;
		double a23 =  - 17.640374582556646;
		double a24 = 6.030101025553685;
		double a25 = 1.9616713220254751;
		double a26 =  - 0.10375954139686289;
		double a27 = 0.004701657843191203;
		double a28 = 0.000059787194931058;
		double a30 = 0.04373725032368315;
		double a31 = 0.4408046760211261;
		double a32 = 0.14843564462720454;
		double a33 =  - 1.5532900267938194;
		double a34 = 0.6641132677516352;
		double a35 = 0.27297113142637586;
		double a36 =  - 0.018397013912867757;
		double a37 = 0.0017330940089714736;
		double a38 =  - 0.00010802345222657461;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2B6_6_Diagonals() {
	double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 = 36.19021847472254;
		double gamma02 = 113.25941023826337;
		double gamma10 = 0.007231008320535141;
		double gamma12 = 2.69425263036507;
		double gamma13 = 0.28941109404967313;
		double gamma20 =  - 0.07163193273547488;
		double gamma21 = 0.6578363253161784;
		double gamma23 =  - 2.46813125661314;
		double gamma24 =  - 0.8534025355100913;
		double gamma31 = 0.06801243933070299;
		double gamma32 = 0.4799578622948373;
		double gamma34 = 0.6679373011705586;
		double gamma35 = 0.12057646664909265;
		double a00 = 24.88999966588646;
		double a01 = 92.91622199913782;
		double a02 =  - 281.2361358139152;
		double a03 = 188.89783829334607;
		double a04 =  - 31.25428285256421;
		double a05 = 7.015818196227269;
		double a06 =  - 1.4228783104279645;
		double a07 = 0.2095054501853543;
		double a08 =  - 0.016086643554247482;
		double a10 = 0.7602547688812334;
		double a11 = 1.8261158850963042;
		double a12 =  - 6.074746092625187;
		double a13 = 3.7134996435879426;
		double a14 =  - 0.3371205862382884;
		double a15 = 0.1481979635854135;
		double a16 =  - 0.04483513637453837;
		double a17 = 0.009687719253295729;
		double a18 =  - 0.0010541649061336703;
		double a20 = 0.5262639067522898;
		double a21 = 0.634893210962032;
		double a22 =  - 6.156494312364238;
		double a23 = 7.742265488948831;
		double a24 =  - 2.1655039728516434;
		double a25 =  - 0.6006994698028445;
		double a26 = 0.017724774066942988;
		double a27 = 0.001939451342880214;
		double a28 =  - 0.00038907705929594824;
		double a30 = 0.0366460018335268;
		double a31 = 0.37495055545497974;
		double a32 = 0.24758315585792942;
		double a33 =  - 1.0876644418241608;
		double a34 =  - 0.1805752738192111;
		double a35 = 0.5360082630861024;
		double a36 = 0.07637866273240493;
		double a37 =  - 0.0034796483264718544;
		double a38 = 0.00015272500481513978;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2B6_7_Diagonals() {
		double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 = 36.19021847472254;
		double gamma02 = 113.25941023826337;
		double gamma10 = 0.007231008320535141;
		double gamma12 = 2.69425263036507;
		double gamma13 = 0.28941109404967313;
		double gamma20 = 0.000023410102770690376;
		double gamma21 = 0.1449635317301256;
		double gamma23 = 1.7971071968105605;
		double gamma24 = 0.5693342620125411;
		double gamma31 = 0.07922749385978499;
		double gamma32 = 0.5184974267516888;
		double gamma34 = 0.6318059589000218;
		double gamma35 = 0.10611676404998778;
		double a00 = 24.88999966588646;
		double a01 = 92.91622199913782;
		double a02 =  - 281.2361358139152;
		double a03 = 188.89783829334607;
		double a04 =  - 31.25428285256421;
		double a05 = 7.015818196227269;
		double a06 =  - 1.4228783104279645;
		double a07 = 0.2095054501853543;
		double a08 =  - 0.016086643554247482;
		double a10 = 0.7602547688812334;
		double a11 = 1.8261158850963042;
		double a12 =  - 6.074746092625187;
		double a13 = 3.7134996435879426;
		double a14 =  - 0.3371205862382884;
		double a15 = 0.1481979635854135;
		double a16 =  - 0.04483513637453837;
		double a17 = 0.009687719253295729;
		double a18 =  - 0.0010541649061336703;
		double a20 = 0.07843448919007862;
		double a21 = 0.7925421816736662;
		double a22 = 0.26766049193215696;
		double a23 =  - 2.80138218604522;
		double a24 = 1.2085547124440164;
		double a25 = 0.48588237304344334;
		double a26 =  - 0.03469540701413635;
		double a27 = 0.0032016385198230914;
		double a28 =  - 0.00019829374359747512;
		double a30 = 0.04429475720503604;
		double a31 = 0.40638732792684873;
		double a32 = 0.16191719116992373;
		double a33 =  - 1.0853614962252085;
		double a34 =  - 0.10375586497590743;
		double a35 = 0.5158244594699389;
		double a36 = 0.06319221006496846;
		double a37 =  - 0.0025993576578962094;
		double a38 = 0.00010077302230089306;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2B6_8_Diagonals() {
	double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 = 22.653918364613663;
		double gamma02 = 57.595270456736245;
		double gamma10 = 0.007231008320535141;
		double gamma12 = 2.69425263036507;
		double gamma13 = 0.28941109404967313;
		double gamma20 =  - 0.004680706235330545;
		double gamma21 = 0.1411363882989607;
		double gamma23 = 1.9144076893977993;
		double gamma24 = 0.5929862727139488;
		double gamma31 = 0.07160180980823996;
		double gamma32 = 0.5105175853574312;
		double gamma34 = 0.4870081066438877;
		double gamma35 = 0.05887220941193474;
		double a00 = 17.517035818126466;
		double a01 = 37.6213773326356;
		double a02 =  - 139.06464793868213;
		double a03 = 98.27847458196185;
		double a04 =  - 18.372653857827878;
		double a05 = 5.196474429261478;
		double a06 =  - 1.4452698354750804;
		double a07 = 0.30131622180376216;
		double a08 =  - 0.03210719404247203;
		double a10 = 0.7602547688812334;
		double a11 = 1.8261158850963042;
		double a12 =  - 6.074746092625187;
		double a13 = 3.7134996435879426;
		double a14 =  - 0.3371205862382884;
		double a15 = 0.1481979635854135;
		double a16 =  - 0.04483513637453837;
		double a17 = 0.009687719253295729;
		double a18 =  - 0.0010541649061336703;
		double a20 = 0.07785151424640029;
		double a21 = 0.7733506882986915;
		double a22 = 0.4568475926462478;
		double a23 =  - 3.110870876023315;
		double a24 = 1.3398567408770043;
		double a25 = 0.49444405837214994;
		double a26 =  - 0.034487810125716285;
		double a27 = 0.003218373805875371;
		double a28 =  - 0.0002102820969620759;
		double a30 = 0.03817919596469591;
		double a31 = 0.40200717892753707;
		double a32 = 0.21853881346376297;
		double a33 =  - 1.3451767097053786;
		double a34 = 0.2619989200086989;
		double a35 = 0.3974653278525915;
		double a36 = 0.027810985676208752;
		double a37 =  - 0.0008467874112833383;
		double a38 = 0.000023075223112788605;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}
MatrixDiagonalEntries* create2B6_9_Diagonals() {
	double alpha0 = 0.5303539471149911;
		double alpha = alpha0 ;

		double beta = 0.07046082051827662;
		double a1 = 0.14302426313188774;
		double a2 = 0.4414349375347824;
		double a3 = 0.03405499786284953;
		double a4 =  - 0.0008518411731330032;
		double gamma01 = 55.48077618074177;
		double gamma02 = 162.93299971063203;
		double gamma10 =  - 0.038569791275243764;
		double gamma12 = 2.7338695476690758;
		double gamma13 = 0.30818307414877355;
		double gamma20 =  - 0.019197909225382476;
		double gamma21 = 0.300733077029695;
		double gamma23 = 0.6117125584008577;
		double gamma24 = 0.16027058561285706;
		double gamma31 = 0.08075867224970161;
		double gamma32 = 0.556308421480436;
		double gamma34 = 0.32018448581760395;
		double gamma35 = 0.0011602385158419003;
		double a00 = 39.67640601802065;
		double a01 = 126.06264145757221;
		double a02 =  - 401.59458140450505;
		double a03 = 273.6070591235535;
		double a04 =  - 46.836815450397935;
		double a05 = 11.19788792517495;
		double a06 =  - 2.4869732070599335;
		double a07 = 0.409998736910785;
		double a08 =  - 0.0356232261710893;
		double a10 = 0.7550916092741523;
		double a11 = 1.8827628299899253;
		double a12 =  - 6.153279351664581;
		double a13 = 3.7176032968353927;
		double a14 =  - 0.3090554759859175;
		double a15 = 0.14117344509279187;
		double a16 =  - 0.04239825525723455;
		double a17 = 0.009085011132319977;
		double a18 =  - 0.0009831114313225144;
		double a20 = 0.2179990997373567;
		double a21 = 0.7185740245929354;
		double a22 =  - 1.5028916509117003;
		double a23 = 0.11391053987056853;
		double a24 = 0.30236931317313814;
		double a25 = 0.166674067211463;
		double a26 =  - 0.01936699035708396;
		double a27 = 0.0030207624554990193;
		double a28 =  - 0.0002891657721055215;
		double a30 = 0.04373725032368315;
		double a31 = 0.4408046760211261;
		double a32 = 0.14843564462720454;
		double a33 =  - 1.5532900267938194;
		double a34 = 0.6641132677516352;
		double a35 = 0.27297113142637586;
		double a36 =  - 0.018397013912867757;
		double a37 = 0.0017330940089714736;
		double a38 =  - 0.00010802345222657461;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24},
			{0.0, gamma31, gamma32, 1.0, gamma34, gamma35}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06, a07, a08},
			{a10, a11, a12, a13, a14, a15, a16, a17, a18},
			{a20, a21, a22, a23, a24, a25, a26, a27, a28},
			{a30, a31, a32, a33, a34, a35, a36, a37, a38}
		};

		double t1 = -2.0 * (a1 + a2 + a3 + a4);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a4, a3, a2, a1, t1, a1, a2, a3, a4
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_1_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 8.684466815062526;
		double gamma02 = 5.2334494385813395;
		double gamma10 =  - 0.2908610497433836;
		double gamma12 = 19.151728922470173;
		double gamma13 = 3.7205127814515646;
		double a00 = 10.44130712725852;
		double a01 =  - 16.162845440315607;
		double a02 = 0.758609811076265;
		double a03 = 5.206088189802421;
		double a04 =  - 0.24315968780464542;
		double a10 =  - 1.8376982037450174;
		double a11 = 27.629938588220078;
		double a12 =  - 48.28224588801183;
		double a13 = 21.025468826343424;
		double a14 = 1.4645366771933166;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_2_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 8.684466815062526;
		double gamma02 = 5.2334494385813395;
		double gamma10 =  - 1.8243854693914217;
		double gamma12 = 80.20500865797653;
		double gamma13 = 23.19607329683229;
		double a00 = 10.44130712725852;
		double a01 =  - 16.162845440315607;
		double a02 = 0.758609811076265;
		double a03 = 5.206088189802421;
		double a04 =  - 0.24315968780464542;
		double a10 =  - 13.021214448625619;
		double a11 = 128.81671004430598;
		double a12 =  - 205.74614695574016;
		double a13 = 77.12702157307483;
		double a14 = 12.823629786989436;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_3_Diagonals() {
	double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 8.684466815062526;
		double gamma02 = 5.2334494385813395;
		double gamma10 = 0.07107618833329107;
		double gamma12 = 2.0880384167821795;
		double gamma13 = 0.36211301767325815;
		double a00 = 10.44130712725852;
		double a01 =  - 16.162845440315607;
		double a02 = 0.758609811076265;
		double a03 = 5.206088189802421;
		double a04 =  - 0.24315968780464542;
		double a10 = 0.9197929297674693;
		double a11 = 0.6220952627121306;
		double a12 =  - 3.8638157439525695;
		double a13 = 2.1821739806988014;
		double a14 = 0.13975357077415287;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_4_Diagonals() {
	double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 8.684466815062526;
		double gamma02 = 5.2334494385813395;
		double gamma10 = 0.03723398173179819;
		double gamma12 = 2.9583667070542674;
		double gamma13 = 0.6798229192749783;
		double a00 = 10.44130712725852;
		double a01 =  - 16.162845440315607;
		double a02 = 0.758609811076265;
		double a03 = 5.206088189802421;
		double a04 =  - 0.24315968780464542;
		double a10 = 0.7220833111903254;
		double a11 = 2.181735407488611;
		double a12 =  - 6.202282481547309;
		double a13 = 2.971025495866556;
		double a14 = 0.3274382670015599;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_5_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 8.684466815062526;
		double gamma02 = 5.2334494385813395;
		double gamma10 =  - 0.013104751070101929;
		double gamma12 = 5.324398207195127;
		double gamma13 = 1.1231225971528522;
		double a00 = 10.44130712725852;
		double a01 =  - 16.162845440315607;
		double a02 = 0.758609811076265;
		double a03 = 5.206088189802421;
		double a04 =  - 0.24315968780464542;
		double a10 = 0.3411510756831667;
		double a11 = 5.920479651251884;
		double a12 =  - 12.373929354576937;
		double a13 = 5.621815452665472;
		double a14 = 0.4904831749763318;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_6_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 9.339531430253219;
		double gamma02 = 4.150387317337683;
		double gamma10 =  - 0.2908610497433836;
		double gamma12 = 19.151728922470173;
		double gamma13 = 3.7205127814515646;
		double a00 = 11.132038201282032;
		double a01 =  - 18.698702627287826;
		double a02 = 3.7937974228554268;
		double a03 = 3.980360233036804;
		double a04 =  - 0.2074932289548284;
		double a10 =  - 1.8376982037450174;
		double a11 = 27.629938588220078;
		double a12 =  - 48.28224588801183;
		double a13 = 21.025468826343424;
		double a14 = 1.4645366771933166;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_7_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 9.339531430253219;
		double gamma02 = 4.150387317337683;
		double gamma10 =  - 1.8243854693914217;
		double gamma12 = 80.20500865797653;
		double gamma13 = 23.19607329683229;
		double a00 = 11.132038201282032;
		double a01 =  - 18.698702627287826;
		double a02 = 3.7937974228554268;
		double a03 = 3.980360233036804;
		double a04 =  - 0.2074932289548284;
		double a10 =  - 13.021214448625619;
		double a11 = 128.81671004430598;
		double a12 =  - 205.74614695574016;
		double a13 = 77.12702157307483;
		double a14 = 12.823629786989436;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_8_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 9.339531430253219;
		double gamma02 = 4.150387317337683;
		double gamma10 = 0.07107618833329107;
		double gamma12 = 2.0880384167821795;
		double gamma13 = 0.36211301767325815;
		double a00 = 11.132038201282032;
		double a01 =  - 18.698702627287826;
		double a02 = 3.7937974228554268;
		double a03 = 3.980360233036804;
		double a04 =  - 0.2074932289548284;
		double a10 = 0.9197929297674693;
		double a11 = 0.6220952627121306;
		double a12 =  - 3.8638157439525695;
		double a13 = 2.1821739806988014;
		double a14 = 0.13975357077415287;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_9_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 9.339531430253219;
		double gamma02 = 4.150387317337683;
		double gamma10 = 0.03723398173179819;
		double gamma12 = 2.9583667070542674;
		double gamma13 = 0.6798229192749783;
		double a00 = 11.132038201282032;
		double a01 =  - 18.698702627287826;
		double a02 = 3.7937974228554268;
		double a03 = 3.980360233036804;
		double a04 =  - 0.2074932289548284;
		double a10 = 0.7220833111903254;
		double a11 = 2.181735407488611;
		double a12 =  - 6.202282481547309;
		double a13 = 2.971025495866556;
		double a14 = 0.3274382670015599;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C4_10_Diagonals() {
double alpha0 = 0.39755582066409934;
		double alpha = alpha0;

		double beta = 0.021185375111255712;
		double a1 = 0.5207872376311244;
		double a2 = 0.32917378847989637;
		double gamma01 = 9.339531430253219;
		double gamma02 = 4.150387317337683;
		double gamma10 =  - 0.013104751070101929;
		double gamma12 = 5.324398207195127;
		double gamma13 = 1.1231225971528522;
		double a00 = 11.132038201282032;
		double a01 =  - 18.698702627287826;
		double a02 = 3.7937974228554268;
		double a03 = 3.980360233036804;
		double a04 =  - 0.2074932289548284;
		double a10 = 0.3411510756831667;
		double a11 = 5.920479651251884;
		double a12 =  - 12.373929354576937;
		double a13 = 5.621815452665472;
		double a14 = 0.4904831749763318;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C6_1_Diagonals() {
	double alpha0 = 0.3552519833863128;
		double alpha = alpha0 ;

		double beta = 0.015385256590721305;
		double a1 = 0.6545272676086319;
		double a2 = 0.27168680308635906;
		double gamma01 = 8.389045509463868;
		double gamma02 = 4.997784937575292;
		double gamma10 = 0.07252339307985996;
		double gamma12 = 2.274638288052355;
		double gamma13 = 0.3624606194793401;
		double a00 = 10.190142972214264;
		double a01 =  - 15.984695932338362;
		double a02 = 1.2000604108003015;
		double a03 = 4.793395086593271;
		double a04 =  - 0.19890253725298698;
		double a10 = 0.9084349875219685;
		double a11 = 0.8584685172041276;
		double a12 =  - 4.316393176132551;
		double a13 = 2.423640850564837;
		double a14 = 0.12584882084157425;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
MatrixDiagonalEntries* create2C6_2_Diagonals() {
double alpha0 = 0.3552519833863128;
		double alpha = alpha0;

		double beta = 0.015385256590721305;
		double a1 = 0.6545272676086319;
		double a2 = 0.27168680308635906;
		double gamma01 = 8.389045509463868;
		double gamma02 = 4.997784937575292;
		double gamma10 = 0.2261511976657074;
		double gamma12 =  - 6.485820928805345;
		double gamma13 =  - 1.1945389411449001;
		double a00 = 10.190142972214264;
		double a01 =  - 15.984695932338362;
		double a02 = 1.2000604108003015;
		double a03 = 4.793395086593271;
		double a04 =  - 0.19890253725298698;
		double a10 = 2.2163043156875064;
		double a11 =  - 12.672584598558226;
		double a12 = 18.265719229265148;
		double a13 =  - 7.378901925605597;
		double a14 =  - 0.43053702078881434;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C6_3_Diagonals() {
	double alpha0 = 0.3552519833863128;
		double alpha = alpha0;

		double beta = 0.015385256590721305;
		double a1 = 0.6545272676086319;
		double a2 = 0.27168680308635906;
		double gamma01 = 8.389045509463868;
		double gamma02 = 4.997784937575292;
		double gamma10 = 0.03379866035548878;
		double gamma12 = 3.1696580371351843;
		double gamma13 = 0.7427266152046031;
		double a00 = 10.190142972214264;
		double a01 =  - 15.984695932338362;
		double a02 = 1.2000604108003015;
		double a03 = 4.793395086593271;
		double a04 =  - 0.19890253725298698;
		double a10 = 0.6892140383422434;
		double a11 = 2.514197864832381;
		double a12 =  - 6.731694511856689;
		double a13 = 3.1639392758425444;
		double a14 = 0.3643433328357203;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C6_4_Diagonals() {
		double alpha0 = 0.3552519833863128;
		double alpha = alpha0;

		double beta = 0.015385256590721305;
		double a1 = 0.6545272676086319;
		double a2 = 0.27168680308635906;
		double gamma01 = 16.3694267511415;
		double gamma02 = 35.99044586155462;
		double gamma10 = 0.2261511976657074;
		double gamma12 =  - 6.485820928805345;
		double gamma13 =  - 1.1945389411449001;
		double a00 = 14.922770700062632;
		double a01 = 12.038216560971286;
		double a02 =  - 72.29140127540336;
		double a03 = 48.77707006193137;
		double a04 =  - 3.446656050957871;
		double a10 = 2.2163043156875064;
		double a11 =  - 12.672584598558226;
		double a12 = 18.265719229265148;
		double a13 =  - 7.378901925605597;
		double a14 =  - 0.43053702078881434;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C6_5_Diagonals() {
double alpha0 = 0.3552519833863128;
		double alpha = alpha0;

		double beta = 0.015385256590721305;
		double a1 = 0.6545272676086319;
		double a2 = 0.27168680308635906;
		double gamma01 = 16.3694267511415;
		double gamma02 = 35.99044586155462;
		double gamma10 = 0.07022296702089101;
		double gamma12 = 1.614453897769722;
		double gamma13 = 0.3059055205951845;
		double a00 = 14.922770700062632;
		double a01 = 12.038216560971286;
		double a02 =  - 72.29140127540336;
		double a03 = 48.77707006193137;
		double a04 =  - 3.446656050957871;
		double a10 = 0.9614537022805272;
		double a11 =  - 0.020692010289698207;
		double a12 =  - 2.71606379742824;
		double a13 = 1.6483888166035034;
		double a14 = 0.1269132888339242;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C6_6_Diagonals() {
double alpha0 = 0.3552519833863128;
		double alpha = alpha0 ;

		double beta = 0.015385256590721305;
		double a1 = 0.6545272676086319;
		double a2 = 0.27168680308635906;
		double gamma01 = 16.3694267511415;
		double gamma02 = 35.99044586155462;
		double gamma10 = 0.03379866035548878;
		double gamma12 = 3.1696580371351843;
		double gamma13 = 0.7427266152046031;
		double a00 = 14.922770700062632;
		double a01 = 12.038216560971286;
		double a02 =  - 72.29140127540336;
		double a03 = 48.77707006193137;
		double a04 =  - 3.446656050957871;
		double a10 = 0.6892140383422434;
		double a11 = 2.514197864832381;
		double a12 =  - 6.731694511856689;
		double a13 = 3.1639392758425444;
		double a14 = 0.3643433328357203;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}

MatrixDiagonalEntries* create2C6_7_Diagonals() {
double alpha0 = 0.3552519833863128;
		double alpha = alpha0;

		double beta = 0.015385256590721305;
		double a1 = 0.6545272676086319;
		double a2 = 0.27168680308635906;
		double gamma01 = 16.3694267511415;
		double gamma02 = 35.99044586155462;
		double gamma10 = 0.07230092751175442;
		double gamma12 = 0.9510516934910406;
		double gamma13 = 0.3056958077656543;
		double a00 = 14.922770700062632;
		double a01 = 12.038216560971286;
		double a02 =  - 72.29140127540336;
		double a03 = 48.77707006193137;
		double a04 =  - 3.446656050957871;
		double a10 = 1.0228154134715508;
		double a11 =  - 0.9233071778605519;
		double a12 =  - 1.0379225184812009;
		double a13 = 0.7545049166530351;
		double a14 = 0.1839093662135712;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
}
}  // namespace dendroderivs
