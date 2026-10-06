// yHat0 = 1.2
// yHat1 = 1.33


MatrixDiagonalEntries* createBYU_C6_2ND_R060_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3181041957525139;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.012089888332884303;
		double a1 = 0.7479958945581906;
		double a2 = 0.22809806840315142;
		double gamma01 = 10.000000000022066;
		double gamma02 = 2.1250000000657745;
		double gamma10 = 0.056619238266565564;
		double gamma12 = 2.5680530273756346;
		double gamma13 = 0.43380761733443307;
		double a00 = 11.906250000011257;
		double a01 =  - 22.49999999994552;
		double a02 = 9.187499999992419;
		double a03 = 1.5000000000082934;
		double a04 =  - 0.09374999999803411;
		double a10 = 0.8316510578850088;
		double a11 = 1.411306510635555;
		double a12 =  - 5.165345996239873;
		double a13 = 2.7701682290329948;
		double a14 = 0.15222019868624972;

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