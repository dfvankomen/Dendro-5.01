MatrixDiagonalEntries* createBYU_A6_2ND_R075_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.44166857544275956;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.037953543243916414;
		double a1 = 0.3923850755220408;
		double a2 = 0.36861622755813694;
		double a3 = 0.01026602795764041;
		double gamma01 = 17.371763034764584;
		double gamma02 = 32.54469668969328;
		double gamma10 = 0.037487061931642716;
		double gamma12 = 3.138315098327134;
		double gamma13 = 0.9651520092434557;
		double gamma20 = 0.010123124132127428;
		double gamma21 = 0.27906922247130983;
		double gamma23 = 0.6743643006413672;
		double gamma24 = 0.1028760967318001;
		double a00 = 15.38250265992654;
		double a01 = 9.636342665210794;
		double a02 =  - 71.29762325090532;
		double a03 = 53.29815535728832;
		double a04 =  - 8.293233411466934;
		double a05 = 1.405764454952095;
		double a06 =  - 0.13190847815352105;
		double a10 = 0.7142872335026575;
		double a11 = 2.3614847788777706;
		double a12 =  - 6.195177320711232;
		double a13 = 2.412801447348967;
		double a14 = 0.744457460592591;
		double a15 =  - 0.039757275703756395;
		double a16 = 0.0019036761096103431;
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