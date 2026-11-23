// RE-EMITTED BYU_A6_2nd_r_085_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,1.5e-06,7.3e-12,5.4e-09); (1,7,6.9e-08,1.8e-12,1.4e-11); (2,7,4.7e-11,1.1e-13,1.3e-14)
MatrixDiagonalEntries* createBYU_A6_2ND_R085_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.46751933569329174;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.043965240266860744;
		double a1 = 0.316185773889511;
		double a2 = 0.39783363075189193;
		double a3 = 0.01282765055813622;
		double gamma01 = 16.08195466751146;
		double gamma02 = 25.45075067166764;
		double gamma10 = 0.0385408314149095;
		double gamma12 = 3.0719276208800346;
		double gamma13 = 0.9501313668819447;
		double gamma20 = 0.010225812964810238;
		double gamma21 = 0.2768837819049139;
		double gamma23 = 0.7402479122719887;
		double gamma24 = 0.12136541630606675;
		double a00 = 14.913155726185703;
		double a01 = 1.7040212057273947;
		double a02 =  -52.91785401300791;
		double a03 = 42.048160156053015;
		double a04 =  -6.8421990010060405;
		double a05 = 1.2122932001313926;
		double a06 =  -0.1175772740835546;
		double a10 = 0.7236686598389657;
		double a11 = 2.2613114814675046;
		double a12 =  -6.031981412852145;
		double a13 = 2.3501873934306166;
		double a14 = 0.7337814037376149;
		double a15 =  -0.03876870272992829;
		double a16 = 0.0018011771073720647;
		double a20 = 0.19422036058642098;
		double a21 = 0.7434875512620748;
		double a22 =  -1.3059946753777538;
		double a23 =  -0.334778589544395;
		double a24 = 0.6408458229304276;
		double a25 = 0.06386290445444594;
		double a26 =  -0.0016433743112203856;

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