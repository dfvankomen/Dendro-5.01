// yHat0 = 1.44
// yHat1 = 1.15


MatrixDiagonalEntries* createBYU_C6_2ND_R080_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3456089830706943;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.014529829143368032;
		double a1 = 0.6787903006608338;
		double a2 = 0.26037183094182276;
		double gamma01 = 10.000000000006331;
		double gamma02 = 4.535714285872265;
		double gamma10 = 0.06824798320979118;
		double gamma12 = 1.8688058345142786;
		double gamma13 = 0.3175201679020739;
		double a00 = 11.70535714282384;
		double a01 =  - 19.285714285638996;
		double a02 = 3.1607142854903083;
		double a03 = 4.7142857143554195;
		double a04 =  - 0.29464285713151844;
		double a10 = 0.9335294508272222;
		double a11 = 0.33943198083488896;
		double a12 =  - 3.364898661841703;
		double a13 = 1.977383577869971;
		double a14 = 0.11455365230968596;

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