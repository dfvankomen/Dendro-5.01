// yHat0 = 4.05
// yHat1 = 0.89
// yHat2 = 0.93


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP10_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 10.717082473768622;
		double gamma02 = 12.693424732548488;
		double gamma10 = 0.08602524354762144;
		double gamma12 = 1.497475083239552;
		double gamma13 = 0.3219675134755059;
		double gamma20 = 0.04517799379565325;
		double gamma21 = 0.5579904097529904;
		double gamma23 = 0.42386907760088066;
		double gamma24 = 0.0595153498039436;
		double a00 =  - 3.8130661558462733;
		double a01 =  - 12.83095784412892;
		double a02 = 11.888208290635918;
		double a03 = 5.729426893197368;
		double a04 =  - 1.1658091474644807;
		double a05 = 0.21318592759261884;
		double a06 =  - 0.020987659645343965;
		double a10 =  - 0.3328788024749603;
		double a11 =  - 1.3178767783220673;
		double a12 = 0.7398079063964412;
		double a13 = 0.9034684013034955;
		double a14 = 0.003476763516665529;
		double a15 = 0.004598509667748392;
		double a16 =  - 0.0005960000873368163;
		double a20 =  - 0.17642371522146824;
		double a21 =  - 0.7893747480761649;
		double a22 = 0.18466360428304454;
		double a23 = 0.6251821426440751;
		double a24 = 0.14819362698011015;
		double a25 = 0.008275101731307842;
		double a26 =  - 0.000516012340958446;

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