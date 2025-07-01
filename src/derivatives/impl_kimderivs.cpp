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
MatrixDiagonalEntries* createBYU_KIM_17_Diagonals() {
		double alpha0 = 0.5770460201292381;
		double alpha = alpha0;

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
MatrixDiagonalEntries* createBYU_KIM_3_Diagonals() {
		double alpha0 = 0.5770460201292381;
		double alpha = alpha0;

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
    MatrixDiagonalEntries* createBYU_KIM_1_Diagonals() {
		double alpha0 = 0.5770460201292381;
		double alpha = alpha0 ;

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
    MatrixDiagonalEntries* createBYU_KIM_12_Diagonals() {
		double alpha0 = 0.5770460201292381;
		double alpha = alpha0;

		double beta = 0.08900836845092142;
		double a1 = 0.6517078440898839;
		double a2 = 0.24815882627035918;
		double a3 = 0.006009630649852392;
		double gamma01 = 11.58496157030741;
		double gamma02 = 15.852448345394338;
		double gamma10 = 0.10511747198457709;
		double gamma12 = 1.3999606300191687;
		double gamma13 = 0.36626783281004827;
		double gamma20 = 0.015218619469010378;
		double gamma21 = 0.313606951149977;
		double gamma23 = 0.7939030813731411;
		double gamma24 = 0.14913730019362592;
		double a00 =  - 3.8476643371286463;
		double a01 =  - 15.182388400177983;
		double a02 = 11.824878603208575;
		double a03 = 8.313625081405567;
		double a04 =  - 1.3597705976146777;
		double a05 = 0.28263904140786467;
		double a06 =  - 0.031319390435685976;
		double a10 =  - 0.380753971849515;
		double a11 =  - 1.1730254249425538;
		double a12 = 0.6536944559326225;
		double a13 = 0.8627866179739128;
		double a14 = 0.03735935464342531;
		double a15 = 0.0004486340570757499;
		double a16 =  - 0.0005096658149371253;
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
			-a3, -a2, -a1, 0.0, a1/2.0, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}

MatrixDiagonalEntries* createBYU_KIM_4_Diagonals() {
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

	MatrixDiagonalEntries* createBYU_KIM_8_Diagonals() {
		double alpha0 = 0.5770460201292381;
		double alpha = alpha0;

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

}  // namespace dendroderivs
