// yHat0 = 1.83
// yHat1 = 3.95
// yHat2 = 2.1


MatrixDiagonalEntries* createBYU_A6_1ST_R060_OP13_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5288235111192885;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06232711462542066;
		double a1 = 0.690190101550906;
		double a2 = 0.19644389104322207;
		double a3 = 0.00269091403578636;
		double gamma01 = 10.493343500167311;
		double gamma02 = 12.072599632102033;
		double gamma10 = 0.09294284700659586;
		double gamma12 = 0.9328968537216235;
		double gamma13 =  - 0.3503260463984965;
		double gamma20 =  - 0.05906359559032393;
		double gamma21 =  - 0.4893924716466426;
		double gamma23 = 2.88730261195265;
		double gamma24 = 0.8307071176208214;
		double a00 =  - 3.7964705971207646;
		double a01 =  - 12.295497335570657;
		double a02 = 11.691008982939273;
		double a03 = 5.274560321329439;
		double a04 =  - 1.0418468908354217;
		double a05 = 0.1863440748235999;
		double a06 =  - 0.018098543698789944;
		double a10 =  - 0.35744131260369494;
		double a11 =  - 1.1513838995157082;
		double a12 = 1.5214833508232977;
		double a13 = 0.19681478506955283;
		double a14 =  - 0.24439530450969618;
		double a15 = 0.03846657051012963;
		double a16 =  - 0.003544189500706765;
		double a20 = 0.22532796289854465;
		double a21 = 0.19600654118220665;
		double a22 =  - 3.113960945675503;
		double a23 = 0.6476206586458645;
		double a24 = 1.9637175346685765;
		double a25 = 0.0839925917919136;
		double a26 =  - 0.0027043435112985344;

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