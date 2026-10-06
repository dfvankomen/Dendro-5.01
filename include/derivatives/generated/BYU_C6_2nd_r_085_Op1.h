// yHat0 = 1.5
// yHat1 = -0.68


MatrixDiagonalEntries* createBYU_C6_2ND_R085_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.35527536685216693;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.015387330930434163;
		double a1 = 0.6544684317913216;
		double a2 = 0.27171424094347013;
		double gamma01 = 9.999999999978664;
		double gamma02 = 5.499999999880623;
		double gamma10 = 0.06897035006771575;
		double gamma12 = 1.5810429446598506;
		double gamma13 = 0.3102964993228524;
		double a00 = 11.625000000000858;
		double a01 =  - 18.000000000182425;
		double a02 = 0.7500000002727678;
		double a03 = 6.0000000000568585;
		double a04 =  - 0.3750000000073313;
		double a10 = 0.9602185673656155;
		double a11 =  - 0.05292027459945512;
		double a12 =  - 2.6422407863448867;
		double a13 = 1.6023681270257069;
		double a14 = 0.13257436655303143;

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