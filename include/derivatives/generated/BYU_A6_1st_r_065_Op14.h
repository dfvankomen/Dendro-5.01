// yHat0 = 1.8
// yHat1 = 0.89
// yHat2 = 1.5


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP14_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 10.121749695600322;
		double gamma02 = 11.294215822998105;
		double gamma10 = 0.08602524354762144;
		double gamma12 = 1.497475083239552;
		double gamma13 = 0.3219675134755059;
		double gamma20 = 0.015676071490883525;
		double gamma21 = 0.31443161599328323;
		double gamma23 = 0.7753042511904301;
		double gamma24 = 0.1335131522459523;
		double a00 =  - 3.7604844225108827;
		double a01 =  - 11.507265113037956;
		double a02 = 11.216081672863723;
		double a03 = 4.856038277703331;
		double a04 =  - 0.962316498318217;
		double a05 = 0.17545801873727876;
		double a06 =  - 0.017511940465420973;
		double a10 =  - 0.3328788024749603;
		double a11 =  - 1.3178767783220673;
		double a12 = 0.7398079063964412;
		double a13 = 0.9034684013034955;
		double a14 = 0.003476763516665529;
		double a15 = 0.004598509667748392;
		double a16 =  - 0.0005960000873368163;
		double a20 =  - 0.06817482946728391;
		double a21 =  - 0.610970260866978;
		double a22 =  - 0.42954644180160495;
		double a23 = 0.7357702469558547;
		double a24 = 0.3626019391065397;
		double a25 = 0.010646338343893325;
		double a26 =  - 0.00032699227039623466;

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