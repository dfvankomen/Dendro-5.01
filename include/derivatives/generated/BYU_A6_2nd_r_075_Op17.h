// yHat0 = 3.5
// yHat1 = 1.35
// yHat2 = 1.77


MatrixDiagonalEntries* create2A6_r075_Op17_Diagonals() {
		double alpha0 = 0.44166857544275956;
		double alpha = alpha0;

		double beta = 0.037953543243916414;
		double a1 = 0.3923850755220408;
		double a2 = 0.36861622755813694;
		double a3 = 0.01026602795764041;
		double gamma01 = 13.897291665413453;
		double gamma02 = 13.435104363432599;
		double gamma10 = 0.04272799765757154;
		double gamma12 = 2.808136147723417;
		double gamma13 = 0.7053760151352628;
		double gamma20 = 0.010123124132127428;
		double gamma21 = 0.27906922247130983;
		double gamma23 = 0.6743643006413672;
		double gamma24 = 0.1028760967318001;
		double a00 = 14.118181153424748;
		double a01 =  - 11.731656132985075;
		double a02 =  - 21.786406476275655;
		double a03 = 22.99304420857628;
		double a04 =  - 4.384453277658937;
		double a05 = 0.8845937621981218;
		double a06 =  - 0.09330324123179526;
		double a10 = 0.7588895344503678;
		double a11 = 1.8910322256434933;
		double a12 =  - 5.661126390610678;
		double a13 = 2.6051930781069355;
		double a14 = 0.41375399622986375;
		double a15 =  - 0.007079999446540129;
		double a16 =  - 0.0006624444203008905;
		double a20 = 0.19448300593668907;
		double a21 = 0.7546047235377337;
		double a22 =  - 1.4124605586131673;
		double a23 =  - 0.16736791109779983;
		double a24 = 0.5800075045217016;
		double a25 = 0.05200930539410005;
		double a26 =  - 0.0012760696792016243;

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