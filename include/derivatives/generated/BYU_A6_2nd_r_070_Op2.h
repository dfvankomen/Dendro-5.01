MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.4309329692460775;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.035593529920944555;
		double a1 = 0.42362022943349215;
		double a2 = 0.35644150411900966;
		double a3 = 0.009296305824945953;
		double gamma01 = 13.503763169558841;
		double gamma02 = 11.270697424892315;
		double gamma10 = 0.03720859758781716;
		double gamma12 = 3.1558583519997545;
		double gamma13 = 1.153395432691522;
		double gamma20 = 0.010065992847351608;
		double gamma21 = 0.27872724988737285;
		double gamma23 = 0.6702657731502768;
		double gamma24 = 0.10115767500261996;
		double a00 = 13.974980509619714;
		double a01 =  - 14.151856444150768;
		double a02 =  - 16.17862660455774;
		double a03 = 19.560601303532042;
		double a04 =  - 3.941733567547596;
		double a05 = 0.8255644506151476;
		double a06 =  - 0.08893070195792584;
		double a10 = 0.7138556307169072;
		double a11 = 2.3603149995920023;
		double a12 =  - 5.9618915262354655;
		double a13 = 1.9277124035671103;
		double a14 = 1.0236898718976732;
		double a15 =  - 0.06765963200831972;
		double a16 = 0.00397825243792773;
		double a20 = 0.19390036846069136;
		double a21 = 0.756607426075347;
		double a22 =  - 1.4199381805798086;
		double a23 =  - 0.15740072255884324;
		double a24 = 0.5774681543789373;
		double a25 = 0.05057924009688566;
		double a26 =  - 0.001216285873234134;

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