// RE-EMITTED BYU_C6_2nd_r_080_Op4.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,5,3.9e-09,7.2e-13,1.6e-10); (1,5,2.2e-10,1.8e-15,1.1e-12)
// yHat0 = 1.44
// yHat1 = 2.35


MatrixDiagonalEntries* createBYU_C6_2ND_R080_OP4_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3456089830706943;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.014529829143368032;
		double a1 = 0.6787903006608338;
		double a2 = 0.26037183094182276;
		double gamma01 = 10.000000000006331;
		double gamma02 = 4.535714285872265;
		double gamma10 = 0.03803309490681825;
		double gamma12 = 2.7860670134350127;
		double gamma13 = 0.6196690509318286;
		double a00 = 11.705357142851765;
		double a01 =  -19.285714285519475;
		double a02 = 3.160714285327997;
		double a03 = 4.714285714495714;
		double a04 =  -0.29464285715600197;
		double a10 = 0.743785188114313;
		double a11 = 1.9250255456982108;
		double a12 =  -5.794018606506854;
		double a13 = 2.8378198234618224;
		double a14 = 0.28738804923250755;

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