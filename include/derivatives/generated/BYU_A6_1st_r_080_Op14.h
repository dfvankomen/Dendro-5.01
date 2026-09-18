// RE-EMITTED BYU_A6_1st_r_080_Op14.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,5.1e-04,4.5e-13,2.0e-07); (1,6,1.6e-07,5.7e-14,8.6e-11); (2,6,1.3e-10,1.1e-13,1.8e-13)
// yHat0 = 2.37
// yHat1 = 2.45
// yHat2 = 4.41


MatrixDiagonalEntries* createBYU_A6_1ST_R080_OP14_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5551807442234595;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07520673263376323;
		double a1 = 0.670920507629016;
		double a2 = 0.22381002064073385;
		double a3 = 0.003948975982246324;
		double gamma01 = 13.490805146770562;
		double gamma02 = 18.134636077578367;
		double gamma10 = 0.0861929342247668;
		double gamma12 = 1.4034902721703322;
		double gamma13 = 0.19760981966353192;
		double gamma20 = 0.013553130952545497;
		double gamma21 = 0.2630022074719273;
		double gamma23 = 1.079077063763526;
		double gamma24 = 0.2808148214643671;
		double a00 =  -4.093979655209148;
		double a01 =  -18.567054369386902;
		double a02 = 15.648475155005691;
		double a03 = 8.361506192153554;
		double a04 =  -1.574980416480382;
		double a05 = 0.24525019031780848;
		double a06 =  -0.01921709640062072;
		double a10 =  -0.3343498434393931;
		double a11 =  -1.2979303639033357;
		double a12 = 0.886642969800573;
		double a13 = 0.7792732577255549;
		double a14 =  -0.0434279413470593;
		double a15 = 0.010922084409567991;
		double a16 =  -0.0011301632459078037;
		double a20 =  -0.05700990945071034;
		double a21 =  -0.5317811305044204;
		double a22 =  -0.6963766838880673;
		double a23 = 0.610930765277935;
		double a24 = 0.6414607088314191;
		double a25 = 0.034310907629624035;
		double a26 =  -0.0015346578957801417;

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