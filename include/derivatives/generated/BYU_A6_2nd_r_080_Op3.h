// RE-EMITTED BYU_A6_2nd_r_080_Op3.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,9.4e-06,2.3e-11,8.0e-09); (1,7,1.2e-07,1.8e-12,1.9e-10); (2,7,7.2e-10,3.6e-12,5.3e-13)
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
		double a00 = 14.992318816763525;
		double a01 = 3.041937859079051;
		double a02 =  -56.017904796590464;
		double a03 = 43.94565712721056;
		double a04 =  -7.086939852454896;
		double a05 = 1.2449253136728058;
		double a06 =  -0.11999446768058344;
		double a10 = 0.7913387232716541;
		double a11 = 1.544734493066821;
		double a12 =  -5.1045389583170415;
		double a13 = 2.4046017750746835;
		double a14 = 0.3676969643190446;
		double a15 =  -0.0027482255579412977;
		double a16 =  -0.0010847718572199647;
		double a20 = 0.15646768231989341;
		double a21 = 0.744874028844807;
		double a22 =  -0.6814947876813523;
		double a23 =  -1.3468041439198912;
		double a24 = 0.9712379500789101;
		double a25 = 0.16073944418387007;
		double a26 =  -0.005020173826237073;

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