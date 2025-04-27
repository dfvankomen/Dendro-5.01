#include "derivatives/impl_byuderivs.h"

#include "derivatives/derivs_compact.h"

namespace dendroderivs {

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
    double alpha0  = 1.0 / 3.0;
    double alpha   = alpha0 + D_coeffs[0];
    double a07     = D_coeffs[1];
    double a17     = D_coeffs[2];
    double a27     = D_coeffs[3];
    double a37     = D_coeffs[4];

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
    double a30     = (1.0 - 900.0 * a37) / 480.0;
    double a31     = (-1.0 + 700.0 * a37) / 20.0;
    double a32     = (-25.0 + 6468.0 * a37) / 32.0;
    double a33     = -525.0 * a37;
    double a34     = (25.0 * (1.0 + 252.0 * a37)) / 32.0;
    double a35     = (1.0 + 2100.0 * a37) / 20.0;
    double a36     = (-1.0 - 6300.0 * a37) / 480.0;
    double gamma01 = -6.0 * (-1.0 + 7.0 * a07);
    double gamma10 = (1.0 - 210.0 * a17) / 12.0;
    double gamma12 = (5.0 * (1.0 + 294.0 * a17)) / 4.0;
    double gamma21 = (1.0 + 420.0 * a27) / 5.0;
    double gamma23 = (-2.0 * (-1.0 + 630.0 * a27)) / 3.0;
    double gamma32 = (-3.0 * (-1.0 + 420.0 * a37)) / 8.0;
    double gamma34 = (3.0 * (1.0 + 700.0 * a37)) / 8.0;

    // clang-format off
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
        {a00, a01, a02, a03, a04, a05, a06, a07},
        {a10, a11, a12, a13, a14, a15, a16, a17},
        {a20, a21, a22, a23, a24, a25, a26, a27},
        {a30, a31, a32, a33, a34, a35, a36, a37}
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

}  // namespace dendroderivs
