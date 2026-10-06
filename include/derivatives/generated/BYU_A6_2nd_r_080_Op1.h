MatrixDiagonalEntries* createBYU_A6_2ND_R080_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.45380344635841907;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.04071702966194011;
		double a1 = 0.35679115670773653;
		double a2 = 0.38234902673152094;
		double a3 = 0.011428187600766462;
		double gamma01 = 14.20182115333874;
		double gamma02 = 15.110016341750255;
		double gamma10 = 0.04125627493568223;
		double gamma12 = 2.9008546790563345;
		double gamma13 = 0.9211656908051901;
		double gamma20 = 0.012204856906641892;
		double gamma21 = 0.29723221557261104;
		double gamma23 = 0.7793236352970117;
		double gamma24 = 0.13412879946906758;
		double a00 = 14.228996030855507;
		double a01 =  - 9.858799909447146;
		double a02 =  - 26.125951422017497;
		double a03 = 25.64921783376672;
		double a04 =  - 4.727048795944358;
		double a05 = 0.9302731728037936;
		double a06 =  - 0.09668690169043859;
		double a10 = 0.7479517544569176;
		double a11 = 2.0017152226303634;
		double a12 =  - 5.596833006372181;
		double a13 = 2.1623215035881134;
		double a14 = 0.7208816292310689;
		double a15 =  - 0.03768238354277288;
		double a16 = 0.001645280032754811;
		double a20 = 0.21921145672873935;
		double a21 = 0.6857220485143045;
		double a22 =  - 1.2172573879739867;
		double a23 =  - 0.4296908509351978;
		double a24 = 0.6701140518206448;
		double a25 = 0.07399501811625134;
		double a26 =  - 0.002094336270766047;

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