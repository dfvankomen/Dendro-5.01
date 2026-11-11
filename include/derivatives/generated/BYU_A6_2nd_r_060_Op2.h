// RE-EMITTED BYU_A6_2nd_r_060_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,2.7e-04,1.1e-11,8.1e-07); (1,7,1.2e-07,1.8e-12,5.0e-11); (2,7,9.8e-11,4.5e-13,5.0e-14)
MatrixDiagonalEntries* createBYU_A6_2ND_R060_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.41306375445879423;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.031838395471255235;
		double a1 = 0.47509136605394736;
		double a2 = 0.3361249867091765;
		double a3 = 0.007801442996605072;
		double gamma01 = 13.59852618075485;
		double gamma02 = 11.791893994157206;
		double gamma10 = 0.03614767034846456;
		double gamma12 = 3.222696768051901;
		double gamma13 = 1.0640756170744514;
		double gamma20 = 0.010047093356878511;
		double gamma21 = 0.28249436229985436;
		double gamma23 = 0.5869003138351572;
		double gamma24 = 0.0792478888341707;
		double a00 = 14.009463693551936;
		double a01 =  -13.569063988350077;
		double a02 =  -17.52899807577076;
		double a03 = 20.38714502103599;
		double a04 =  -4.048341953349684;
		double a05 = 0.8397789271131918;
		double a06 =  -0.08998362423059772;
		double a10 = 0.7032500087353868;
		double a11 = 2.4768350995758297;
		double a12 =  -6.282859675485576;
		double a13 = 2.2750674215226256;
		double a14 = 0.8777747170310923;
		double a15 =  -0.052988546979257055;
		double a16 = 0.002920975599898701;
		double a20 = 0.1955125769233573;
		double a21 = 0.7678252733812091;
		double a22 =  -1.5527017377471712;
		double a23 = 0.05556411119774525;
		double a24 = 0.4972663635534025;
		double a25 = 0.03738006184245489;
		double a26 =  -0.0008466491509978186;

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