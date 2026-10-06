MatrixDiagonalEntries* createBYU_A6_2ND_R075_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.44166857544275956;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.037953543243916414;
		double a1 = 0.3923850755220408;
		double a2 = 0.36861622755813694;
		double a3 = 0.01026602795764041;
		double gamma01 = 17.371763034764584;
		double gamma02 = 32.54469668969328;
		double gamma10 = 0.04272799765757154;
		double gamma12 = 2.808136147723417;
		double gamma13 = 0.7053760151352628;
		double gamma20 = 0.008134465047227818;
		double gamma21 = 0.22760866124366744;
		double gamma23 = 1.173662532583605;
		double gamma24 = 0.26062366113506025;
		double a00 = 15.38250265992654;
		double a01 = 9.636342665210794;
		double a02 =  - 71.29762325090532;
		double a03 = 53.29815535728832;
		double a04 =  - 8.293233411466934;
		double a05 = 1.405764454952095;
		double a06 =  - 0.13190847815352105;
		double a10 = 0.7588895344503678;
		double a11 = 1.8910322256434933;
		double a12 =  - 5.661126390610678;
		double a13 = 2.6051930781069355;
		double a14 = 0.41375399622986375;
		double a15 =  - 0.007079999446540129;
		double a16 =  - 0.0006624444203008905;
		double a20 = 0.15364524775647617;
		double a21 = 0.7458222775327222;
		double a22 =  - 0.6356333968268097;
		double a23 =  - 1.429538446635371;
		double a24 = 1.0095435492036995;
		double a25 = 0.16107895460441177;
		double a26 =  - 0.00491818563492362;

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