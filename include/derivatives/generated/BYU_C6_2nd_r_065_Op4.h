// yHat0 = 1.26
// yHat1 = 2.56


MatrixDiagonalEntries* createBYU_C6_2ND_R065_OP4_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3236183938704073;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.012579051069149041;
		double a1 = 0.7341214605841361;
		double a2 = 0.23456835732374415;
		double gamma01 = 10.000000000035156;
		double gamma02 = 2.581081081048821;
		double gamma10 = 0.03669271357757209;
		double gamma12 = 3.058624093036091;
		double gamma13 = 0.633072864224361;
		double a00 = 11.868243243275927;
		double a01 =  - 21.891891891990483;
		double a02 = 8.047297297340974;
		double a03 = 2.108108108151334;
		double a04 =  - 0.13175675675469503;
		double a10 = 0.7160456681628649;
		double a11 = 2.3045195611166958;
		double a12 =  - 6.481443021490878;
		double a13 = 3.185144686978906;
		double a14 = 0.27573310523214944;

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