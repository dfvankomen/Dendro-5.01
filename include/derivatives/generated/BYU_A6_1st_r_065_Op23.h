// yHat0 = 2.46
// yHat1 = 3.92
// yHat2 = 4.53


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP23_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 11.42421093751791;
		double gamma02 = 13.881711866493903;
		double gamma10 = 0.09812313629205824;
		double gamma12 = 0.7575972392818038;
		double gamma13 =  - 0.5134965091566963;
		double gamma20 = 0.0133156110798937;
		double gamma21 = 0.2717451732928028;
		double gamma23 = 0.9942636649673103;
		double gamma24 = 0.23259939213577063;
		double a00 =  - 3.891311427885356;
		double a01 =  - 14.213755455226558;
		double a02 = 12.962862085169553;
		double a03 = 6.1352642681546765;
		double a04 =  - 1.1706801544209744;
		double a05 = 0.19484218130042263;
		double a06 =  - 0.017221499858034322;
		double a10 =  - 0.3732568341329534;
		double a11 =  - 1.0746578876894624;
		double a12 = 1.7072671368447747;
		double a13 =  - 0.002382772517953456;
		double a14 =  - 0.2985494291978242;
		double a15 = 0.04578520514630434;
		double a16 =  - 0.00420541852097271;
		double a20 =  - 0.0572751805750658;
		double a21 =  - 0.5507196751194307;
		double a22 =  - 0.6332355358580858;
		double a23 = 0.6590629288635496;
		double a24 = 0.5579014969990745;
		double a25 = 0.025275980415216504;
		double a26 =  - 0.0010100147252926654;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}