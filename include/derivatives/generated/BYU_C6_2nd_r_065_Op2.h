// yHat0 = 1.26
// yHat1 = 0.04


MatrixDiagonalEntries* createBYU_C6_2ND_R065_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3236183938704073;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.012579051069149041;
		double a1 = 0.7341214605841361;
		double a2 = 0.23456835732374415;
		double gamma01 = 10.000000000035156;
		double gamma02 = 2.581081081048821;
		double gamma10 =  - 2.545568658882944;
		double gamma12 = 148.57134870537942;
		double gamma13 = 26.455686588829366;
		double a00 = 11.868243243275927;
		double a01 =  - 21.891891891990483;
		double a02 = 8.047297297340974;
		double a03 = 2.108108108151334;
		double a04 =  - 0.13175675675469503;
		double a10 =  - 21.093494862926008;
		double a11 = 227.30862218043436;
		double a12 =  - 381.8834307284226;
		double a13 = 166.2149743672443;
		double a14 = 9.453329043669232;

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