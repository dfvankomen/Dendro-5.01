// yHat0 = 1.38
// yHat1 = 0.13


MatrixDiagonalEntries* createBYU_C6_2ND_R075_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3372311995327969;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.013786638668231979;
		double a1 = 0.6998698850465109;
		double a2 = 0.2505414478388867;
		double gamma01 = 9.999999999991703;
		double gamma02 = 3.7580645161701454;
		double gamma10 =  - 4.124830517813965;
		double gamma12 = 236.8765225487121;
		double gamma13 = 42.24830517813927;
		double a00 = 11.770161290279571;
		double a01 =  - 20.322580645165438;
		double a02 = 5.104838709619704;
		double a03 = 3.677419354833005;
		double a04 =  - 0.22983870968084544;
		double a10 =  - 34.374491320862006;
		double a11 = 363.99999627871694;
		double a12 =  - 609.753043701944;
		double a13 = 265.0040638511826;
		double a14 = 15.123474892905486;

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