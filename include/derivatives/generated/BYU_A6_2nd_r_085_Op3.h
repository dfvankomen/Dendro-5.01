// RE-EMITTED BYU_A6_2nd_r_085_Op3.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,1.5e-06,7.3e-12,5.4e-09); (1,7,1.6e-08,1.8e-12,3.2e-11); (2,7,3.1e-10,3.6e-12,2.3e-13)
MatrixDiagonalEntries* createBYU_A6_2ND_R085_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.46751933569329174;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.043965240266860744;
		double a1 = 0.316185773889511;
		double a2 = 0.39783363075189193;
		double a3 = 0.01282765055813622;
		double gamma01 = 16.08195466751146;
		double gamma02 = 25.45075067166764;
		double gamma10 = 0.041876067411967685;
		double gamma12 = 2.8618077530643324;
		double gamma13 = 0.7696048229263016;
		double gamma20 = 0.008314800428082319;
		double gamma21 = 0.22820136665676552;
		double gamma23 = 1.172380075955744;
		double gamma24 = 0.2687249103571382;
		double a00 = 14.913155726185703;
		double a01 = 1.7040212057273947;
		double a02 =  -52.91785401300791;
		double a03 = 42.048160156053015;
		double a04 =  -6.8421990010060405;
		double a05 = 1.2122932001313926;
		double a06 =  -0.1175772740835546;
		double a10 = 0.7518837533018305;
		double a11 = 1.9642055241452407;
		double a12 =  -5.714935884301271;
		double a13 = 2.514026471819473;
		double a14 = 0.5005129927712176;
		double a15 =  -0.015692004333065317;
		double a16 =  -8.534034256935709e-07;
		double a20 = 0.15498563973448504;
		double a21 = 0.741852684364467;
		double a22 =  -0.6324468337789007;
		double a23 =  -1.42058777218265;
		double a24 = 0.9907540329571454;
		double a25 = 0.17086532552001812;
		double a26 =  -0.00542307661456472;

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
