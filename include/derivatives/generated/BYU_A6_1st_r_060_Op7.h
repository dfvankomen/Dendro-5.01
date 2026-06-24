// yHat0 = 2.49
// yHat1 = -0.16
// yHat2 = 1.44


MatrixDiagonalEntries* createBYU_A6_1ST_R060_OP7_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5288235111192885;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06232711462542066;
		double a1 = 0.690190101550906;
		double a2 = 0.19644389104322207;
		double a3 = 0.00269091403578636;
		double gamma01 = 11.16507525053737;
		double gamma02 = 13.381651379236542;
		double gamma10 = 0.025493114185064256;
		double gamma12 = 3.5316917784950306;
		double gamma13 = 2.019628280262894;
		double gamma20 = 0.0151011103390593;
		double gamma21 = 0.30919880265818833;
		double gamma23 = 0.7844200188413499;
		double gamma24 = 0.1354315949671251;
		double a00 =  - 3.864790826762315;
		double a01 =  - 13.681173809113629;
		double a02 = 12.606724780237169;
		double a03 = 5.90040979932169;
		double a04 =  - 1.1365963227300764;
		double a05 = 0.1929513718322449;
		double a06 =  - 0.017525014777683222;
		double a10 =  - 0.14506220847459764;
		double a11 =  - 2.24010711758169;
		double a12 =  - 1.2660731040408477;
		double a13 = 3.212209799227339;
		double a14 = 0.4866094760887315;
		double a15 =  - 0.0514602678846433;
		double a16 = 0.0038834226652815227;
		double a20 =  - 0.06601399450496159;
		double a21 =  - 0.6065930112130826;
		double a22 =  - 0.4441938708782771;
		double a23 = 0.738100604540617;
		double a24 = 0.36835328297217473;
		double a25 = 0.010664600236306922;
		double a26 =  - 0.00031761115278566637;

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