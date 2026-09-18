// RE-EMITTED BYU_C6_2nd_r_065_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,5,1.8e-08,4.0e-12,9.0e-11); (1,5,7.3e-12,7.3e-12,8.0e-13)
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
		double a00 = 11.868243243289172;
		double a01 =  -21.891891892022795;
		double a02 = 8.047297297426033;
		double a03 = 2.1081081080615465;
		double a04 =  -0.13175675675395662;
		double a10 =  -21.09349486292596;
		double a11 = 227.30862218043447;
		double a12 =  -381.8834307284218;
		double a13 = 166.214974367244;
		double a14 = 9.453329043669275;

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