// yHat0 = 0.48
// yHat1 = -0.76
// yHat2 = 3.45


MatrixDiagonalEntries* createBYU_A6_1ST_R060_OP22_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5288235111192885;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06232711462542066;
		double a1 = 0.690190101550906;
		double a2 = 0.19644389104322207;
		double a3 = 0.00269091403578636;
		double gamma01 = 10.654929781152678;
		double gamma02 = 12.522455029641955;
		double gamma10 = 0.09046373793616828;
		double gamma12 = 1.3099752707708212;
		double gamma13 = 0.13204722920024792;
		double gamma20 = 0.009211441307324996;
		double gamma21 = 0.22981644974398696;
		double gamma23 = 1.0977239002663328;
		double gamma24 = 0.26174135180497493;
		double a00 =  - 3.808406462522949;
		double a01 =  - 12.682808564359744;
		double a02 = 11.83255901885571;
		double a03 = 5.605057071042739;
		double a04 =  - 1.1321193639079083;
		double a05 = 0.2059282253403907;
		double a06 =  - 0.020209924456862743;
		double a10 =  - 0.3468377694046317;
		double a11 =  - 1.2447339296445776;
		double a12 = 0.958334302295508;
		double a13 = 0.6830586139355143;
		double a14 =  - 0.061857897412414854;
		double a15 = 0.013412770579448889;
		double a16 =  - 0.0013760903488681598;
		double a20 =  - 0.04147048196795475;
		double a21 =  - 0.5099027245284201;
		double a22 =  - 0.7703002680757134;
		double a23 = 0.6627270567360155;
		double a24 = 0.6329461836366255;
		double a25 = 0.02697090614818951;
		double a26 =  - 0.0009706719488186131;

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