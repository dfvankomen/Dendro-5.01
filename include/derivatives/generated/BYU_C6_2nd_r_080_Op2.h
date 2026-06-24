// yHat0 = 1.44
// yHat1 = 0.13


MatrixDiagonalEntries* createBYU_C6_2ND_R080_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3456089830706943;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.014529829143368032;
		double a1 = 0.6787903006608338;
		double a2 = 0.26037183094182276;
		double gamma01 = 10.000000000006331;
		double gamma02 = 4.535714285872265;
		double gamma10 =  - 2.975912432587205;
		double gamma12 = 172.42959070387073;
		double gamma13 = 30.759124325872357;
		double a00 = 11.70535714282384;
		double a01 =  - 19.285714285638996;
		double a02 = 3.1607142854903083;
		double a03 = 4.7142857143554195;
		double a04 =  - 0.29464285713151844;
		double a10 =  - 24.695470847524543;
		double a11 = 264.28373679620717;
		double a12 =  - 443.46558270631937;
		double a13 = 192.86183841411426;
		double a14 = 11.01547834352214;

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