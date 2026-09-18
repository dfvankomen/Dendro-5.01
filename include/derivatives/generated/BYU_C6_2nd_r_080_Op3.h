// RE-EMITTED BYU_C6_2nd_r_080_Op3.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,5,3.9e-09,7.2e-13,1.6e-10); (1,5,1.7e-11,5.7e-14,8.8e-14)
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
		double a00 = 11.705357142851765;
		double a01 =  -19.285714285519475;
		double a02 = 3.160714285327997;
		double a03 = 4.714285714495714;
		double a04 =  -0.29464285715600197;
		double a10 = 0.9335294508271994;
		double a11 = 0.33943198083486065;
		double a12 =  -3.364898661841631;
		double a13 = 1.9773835778698834;
		double a14 = 0.11455365230968766;

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