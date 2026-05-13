MatrixDiagonalEntries* createBYU_A6_2ND_R080_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.45380344635841907;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.04071702966194011;
		double a1 = 0.35679115670773653;
		double a2 = 0.38234902673152094;
		double a3 = 0.011428187600766462;
		double gamma01 = 16.299502091289924;
		double gamma02 = 26.64726150052526;
		double gamma10 = 0.046380388347500405;
		double gamma12 = 2.578035534097252;
		double gamma13 = 0.6472783969387073;
		double gamma20 = 0.008331962973647163;
		double gamma21 = 0.23066717168863576;
		double gamma23 = 1.1423689577625122;
		double gamma24 = 0.256244223052779;
		double a00 = 14.99231881665744;
		double a01 = 3.0419378592637814;
		double a02 =  - 56.017904788565176;
		double a03 = 43.94565712760628;
		double a04 =  - 7.086939851580807;
		double a05 = 1.2449253135539702;
		double a06 =  - 0.1199944676719127;
		double a10 = 0.7913387232911623;
		double a11 = 1.5447344931033504;
		double a12 =  - 5.104538958511662;
		double a13 = 2.4046017752471864;
		double a14 = 0.36769696429880044;
		double a15 =  - 0.002748225555957909;
		double a16 =  - 0.0010847718574315103;
		double a20 = 0.15646768231983002;
		double a21 = 0.7448740288451795;
		double a22 =  - 0.6814947876818805;
		double a23 =  - 1.3468041439195213;
		double a24 = 0.9712379500789251;
		double a25 = 0.16073944418386377;
		double a26 =  - 0.005020173826236259;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}