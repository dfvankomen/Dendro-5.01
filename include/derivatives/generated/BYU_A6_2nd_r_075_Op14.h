// yHat0 = 3.7
// yHat1 = 4.11
// yHat2 = 0.72


MatrixDiagonalEntries* create2A6_r075_Op14_Diagonals() {
		double alpha0 = 0.44166857544275956;
		double alpha = alpha0;

		double beta = 0.037953543243916414;
		double a1 = 0.3923850755220408;
		double a2 = 0.36861622755813694;
		double a3 = 0.01026602795764041;
		double gamma01 = 16.844177734233075;
		double gamma02 = 29.642977415862923;
		double gamma10 = 0.037487061931642716;
		double gamma12 = 3.138315098327134;
		double gamma13 = 0.9651520092434557;
		double gamma20 = 0.011862934165547963;
		double gamma21 = 0.2926245321701123;
		double gamma23 = 0.7950647826715415;
		double gamma24 = 0.13792992549391464;
		double a00 = 15.19052021867285;
		double a01 = 6.391692980568217;
		double a02 =  - 63.77953254455311;
		double a03 = 48.69643898749566;
		double a04 =  - 7.6996998627037785;
		double a05 = 1.3266266531431825;
		double a06 =  - 0.12604641899113458;
		double a10 = 0.7142872335026575;
		double a11 = 2.3614847788777706;
		double a12 =  - 6.195177320711232;
		double a13 = 2.412801447348967;
		double a14 = 0.744457460592591;
		double a15 =  - 0.039757275703756395;
		double a16 = 0.0019036761096103431;
		double a20 = 0.21437918258922814;
		double a21 = 0.6928198651468838;
		double a22 =  - 1.1968025951002588;
		double a23 =  - 0.47069963250723973;
		double a24 = 0.6865102523170461;
		double a25 = 0.07591435201882522;
		double a26 =  - 0.0021214244644953894;

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