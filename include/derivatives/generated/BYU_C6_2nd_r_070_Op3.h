// yHat0 = 1.35
// yHat1 = 1.24


MatrixDiagonalEntries* createBYU_C6_2ND_R070_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.32995227499332586;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.013140927620375692;
		double a1 = 0.7181845984038895;
		double a2 = 0.2420004517058784;
		double gamma01 = 10.000000000003332;
		double gamma02 = 3.4230769230261777;
		double gamma10 = 0.061428182196262386;
		double gamma12 = 2.2701708413648234;
		double gamma13 = 0.38571817803736047;
		double a00 = 11.79807692308712;
		double a01 =  - 20.769230769295014;
		double a02 = 5.942307692320533;
		double a03 = 3.2307692308215756;
		double a04 =  - 0.2019230769240336;
		double a10 = 0.8745081131222286;
		double a11 = 0.9564229354645998;
		double a12 =  - 4.399000283528836;
		double a13 = 2.4306993081749035;
		double a14 = 0.13736992676708767;

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