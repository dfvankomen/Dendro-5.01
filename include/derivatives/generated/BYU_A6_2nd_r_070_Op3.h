MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.4309329692460775;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.035593529920944555;
		double a1 = 0.42362022943349215;
		double a2 = 0.35644150411900966;
		double a3 = 0.009296305824945953;
		double gamma01 = 13.503763169558841;
		double gamma02 = 11.270697424892315;
		double gamma10 = 0.04039232520947982;
		double gamma12 = 2.955283511737379;
		double gamma13 = 0.7402157798056678;
		double gamma20 = 0.008556874749086792;
		double gamma21 = 0.24037616932520273;
		double gamma23 = 1.0234691356920145;
		double gamma24 = 0.21701417790412234;
		double a00 = 13.974980509619714;
		double a01 =  - 14.151856444150768;
		double a02 =  - 16.17862660455774;
		double a03 = 19.560601303532042;
		double a04 =  - 3.941733567547596;
		double a05 = 0.8255644506151476;
		double a06 =  - 0.08893070195792584;
		double a10 = 0.7381128554218048;
		double a11 = 2.112833622602912;
		double a12 =  - 6.020529011914938;
		double a13 = 2.7397664342706296;
		double a14 = 0.43973732831981377;
		double a15 =  - 0.009503157223304831;
		double a16 =  - 0.0004180714613059667;
		double a20 = 0.16311491178332635;
		double a21 = 0.753481858869672;
		double a22 =  - 0.8702894351038358;
		double a23 =  - 1.0477175836154684;
		double a24 = 0.8727634536901678;
		double a25 = 0.13268481883100747;
		double a26 =  - 0.004038024455398282;

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