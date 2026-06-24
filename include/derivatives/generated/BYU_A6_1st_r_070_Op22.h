// yHat0 = 3.96
// yHat1 = 0.95
// yHat2 = 3.36


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP22_Diagonals(
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
		double gamma20 = 0.008963126848149181;
		double gamma21 = 0.22095269862252;
		double gamma23 = 1.1609462822934218;
		double gamma24 = 0.29334309320121;
		double a00 =  - 3.808476657556039;
		double a01 =  - 12.85825606879239;
		double a02 = 11.811415653474334;
		double a03 = 5.870711007337067;
		double a04 =  - 1.2205536961143357;
		double a05 = 0.22801926373736134;
		double a06 =  - 0.022859445027364946;
		double a10 =  - 0.3335056041220351;
		double a11 =  - 1.3196980957375661;
		double a12 = 0.7069875936801275;
		double a13 = 0.9314165109926849;
		double a14 = 0.01157610913155957;
		double a15 = 0.0037751237888358925;
		double a16 =  - 0.0005516377336847323;
		double a20 =  - 0.03991183036659779;
		double a21 =  - 0.49474767222612337;
		double a22 =  - 0.8222132032575865;
		double a23 = 0.6337088903486695;
		double a24 = 0.6923420392588971;
		double a25 = 0.032046205831962236;
		double a26 =  - 0.0012244295890919026;

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