#include "derivatives/impl_kimderivs.h"

#include "derivatives/derivs_compact.h"
#include "refel.h"

namespace dendroderivs {

MatrixDiagonalEntries* createKimDiagonals() {
    // NOTE: this is standard stencil, not updated, this is **INCORRECT** but
    // left here for previous versions for now

    // const double alpha   = 0.5;
    // const double beta    = 1.0 / 20.0;
    //
    // const double a1      = 17.0 / 24.0;
    // const double a2      = 101.0 / 600.0;
    // const double a3      = 1.0 / 600.0;

    // these are the correct alpha, beta, a1, a2, a3
    const double alpha   = 0.5862704032801503;
    const double beta    = 9.549533555017055e-2;

    const double a1      = 0.6431406736919156;
    const double a2      = 0.2586011023495066;
    const double a3      = 7.140953479797375e-3;

    // i = 0
    const double alpha01 = 43.65980335321481;
    const double beta02  = 92.40143116322876;

    const double b01     = -86.92242000231872;
    const double b02     = 47.58661913475775;
    const double b03     = 57.30693626084370;
    const double b04     = -13.71254216556246;
    const double b05     = 2.659826729790792;
    const double b06     = -0.2598929200600359;

    // i = 1
    const double alpha10 = 0.08351537442980239;
    const double alpha12 = 1.961483362670730;
    const double beta13  = 0.8789761422182460;

    const double b10     = -0.3199960780333493;
    const double b12     = 0.07735499170041915;
    const double b13     = 1.496612372811008;
    const double b14     = 0.2046919801608821;
    const double b15     = -0.02229717539815850;
    const double b16     = 0.001702365014746567;

    // i = 2
    const double beta20  = 0.008073091519768687;
    const double alpha21 = 0.2162434143850924;
    const double alpha23 = 1.052242062502679;
    const double beta24  = 0.2116022463346598;

    const double b20     = -0.03644974757120792;
    const double b21     = -0.4997030280694729;
    const double b23     = 0.7439822445654316;
    const double b24     = 0.5629384925762924;
    const double b25     = 0.01563884275691290;
    const double b26     = -0.0003043666146108995;

    const double b00     = -(b01 + b02 + b03 + b04 + b05 + b06);
    const double b11     = -(b10 + b12 + b13 + b14 + b15 + b16);
    const double b22     = -(b20 + b21 + b23 + b24 + b25 + b26);

    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, alpha01, beta02},
        {alpha10, 1.0, alpha12, beta13},
        {beta20, alpha21, 1.0, alpha23, beta24}};
    // diagonal elements for P matrix for 1st derivative
    std::vector<double> P1DiagInterior{beta, alpha, 1.0, alpha, beta};
    // boundary elements for Q matrix for 1st derivative
    std::vector<std::vector<double>> Q1DiagBoundary{
        {b00, b01, b02, b03, b04, b05, b06},
        {b10, b11, b12, b13, b14, b15, b16},
        {b20, b21, b22, b23, b24, b25, b26},
    };
    // diagonal elements for Q matrix for 1st derivative
    std::vector<double> Q1DiagInterior{-a3, -a2, -a1, 0.0, a1, a2, a3};
    // boundary elements for P matrix for 2nd derivative
    std::vector<std::vector<double>> P2DiagBoundary{
        {1.0, 0.0, 0.0},
    };
    // diagonal elements for P matrix for 2nd derivative
    std::vector<double> P2DiagInterior{0.0, 1.0, 0.0};
    // boundary elements for Q matrix for 2nd derivative
    std::vector<std::vector<double>> Q2DiagBoundary{{1.0, 0.0}};
    std::vector<double> Q2DiagInterior{0.0, 1.0, 0.0};

    // store the entries for matrix creation
    MatrixDiagonalEntries* diagEntries = new MatrixDiagonalEntries{
        P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary};

    return diagEntries;
}

}  // namespace dendroderivs
