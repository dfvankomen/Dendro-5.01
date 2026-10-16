// RE-EMITTED BYU_A6_1st_r_070_Op27.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,3.7e-05,9.1e-13,3.3e-08); (1,6,2.5e-11,1.1e-13,6.6e-14); (2,6,2.7e-11,1.8e-15,4.4e-14)
// yHat0 = 3.96
// yHat1 = 0.95
// yHat2 = 4.47


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP27_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5406172854667417;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06788016563593934;
		double a1 = 0.6819178310623292;
		double a2 = 0.20847205673939975;
		double a3 = 0.003211835520517426;
		double gamma01 = 10.70862212106964;
		double gamma02 = 12.788810882271866;
		double gamma10 = 0.08637035120115655;
		double gamma12 = 1.5167106272387805;
		double gamma13 = 0.34731587522923224;
		double gamma20 = 0.01344998474446978;
		double gamma21 = 0.2694329755370874;
		double gamma23 = 1.021901738355757;
		double gamma24 = 0.24832151038037753;
		double a00 =  -3.808476657435878;
		double a01 =  -12.858256074948118;
		double a02 = 11.811415621348845;
		double a03 = 5.870710974579755;
		double a04 =  -1.2205536735779;
		double a05 = 0.22801925403550555;
		double a06 =  -0.022859444002209777;
		double a10 =  -0.33350560412202807;
		double a11 =  -1.3196980957375213;
		double a12 = 0.7069875936801131;
		double a13 = 0.931416510992751;
		double a14 = 0.0115761091315302;
		double a15 = 0.0037751237888404266;
		double a16 =  -0.0005516377336852299;
		double a20 =  -0.05741762901305519;
		double a21 =  -0.5448966841031303;
		double a22 =  -0.6528913286507672;
		double a23 = 0.6428495918941495;
		double a24 = 0.5853705549778491;
		double a25 = 0.0281584145412127;
		double a26 =  -0.0011729196462586867;

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