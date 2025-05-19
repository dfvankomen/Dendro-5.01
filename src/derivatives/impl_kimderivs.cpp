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
MatrixDiagonalEntries* createKim07Diagonals() {
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

    // Gamma Coefficients
    const double gamma01 = 5.912678614078549;
    const double gamma02 = 3.775623951744012;

    const double gamma10 = 0.08360703307833438;
    const double gamma12 = 2.058102869495757;

    const double gamma20 = 0.03250008295108466;
    const double gamma21 = 0.3998040493524358;

    const double gamma13 = 0.9704052014790193;
    const double gamma23 = 0.7719261277615860;

    const double gamma24 = 0.1626635931256900;

    // b Coefficients
    const double b01 = -3.456878182643609;
    const double b02 = 5.839043358834730;

    const double b10 = -0.3177447290722621;
    const double b12 = -0.02807631929593225;

    const double b20 = -0.1219006056449124;
    const double b21 = -0.6301651351188667;

    const double b03 = 1.015886726041007;
    const double b13 = 1.593461635747659;
    const double b23 = 0.6521195063966084;

    const double b04 = -0.2246526470654333;
    const double b14 = 0.2533207046976367;
    const double b24 = 0.3938843551210350;

    const double b05 = 0.08564940889936562;
    const double b15 = -0.03619652460174756;
    const double b25 = 0.01904944407973912;

    const double b06 = -0.01836710059356763;
    const double b16 = 0.004080281419108407;
    const double b26 = -0.001027260523947668;

    const double b00     = -(b01 + b02 + b03 + b04 + b05 + b06);
    const double b11     = -(b10 + b12 + b13 + b14 + b15 + b16);
    const double b22     = -(b20 + b21 + b23 + b24 + b25 + b26);

    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, gamma01, gamma02},
        {gamma10, 1.0, gamma12, gamma13},
        {gamma20, gamma21, 1.0, gamma23, gamma24}};
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
MatrixDiagonalEntries* createKim16Diagonals() {
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

    // Gamma Coefficients
const double gamma01 = 9.2793108237360826;
const double gamma02 = 9.8711877434133051;

const double gamma10 = 0.11737546726594537;
const double gamma12 = 0.92985849448052303;

const double gamma20 = -0.067747720334188354;
const double gamma21 = -0.1945053944676567;

const double gamma13 = -0.067839996199150834;
const double gamma23 = 1.279653347145571;

const double gamma24 = 0.208243248769505742;

// b Coefficients

const double b01 = -9.9196421679170452;
const double b02 = 10.088151775649886;

const double b10 = -0.4197688256685424;

const double b12 = 1.1593253854830003;

const double b20 = -0.36722447739446801;
const double b21 = -0.20875339530974462;

const double b03 = 4.1769640418803286;
const double b13 = 0.31685779023808876;
const double b23 = 0.9891760218458036;

const double b04 = -0.82222300519220712;
const double b14 = -0.096453054902842381;
const double b24 = 0.63518699715000262;

const double b05 = 0.14757709267988142;
const double b15 = 0.015579947274307879;
const double b25 = 0.002145463566246068;

const double b06 = -0.014332365879513103;
const double b16 = -0.001455364158464077;
const double b26 = 0.0010119119030585999;

    const double b00     = -(b01 + b02 + b03 + b04 + b05 + b06);
    const double b11     = -(b10 + b12 + b13 + b14 + b15 + b16);
    const double b22     = -(b20 + b21 + b23 + b24 + b25 + b26);

    // boundary elements for P matrix for 1st derivative
    std::vector<std::vector<double>> P1DiagBoundary{
        {1.0, gamma01, gamma02},
        {gamma10, 1.0, gamma12, gamma13},
        {gamma20, gamma21, 1.0, gamma23, gamma24}};
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
