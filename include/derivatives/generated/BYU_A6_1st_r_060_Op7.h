// RE-EMITTED BYU_A6_1st_r_060_Op7.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,3.0e-06,4.5e-13,4.5e-08); (1,6,5.4e-10,2.3e-13,2.7e-13); (2,6,1.2e-11,4.5e-13,3.6e-14)
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
		double a00 =  -3.86479082911501;
		double a01 =  -13.681173789884243;
		double a02 = 12.606724821788776;
		double a03 = 5.900409754753106;
		double a04 =  -1.1365963141704627;
		double a05 = 0.19295137126386308;
		double a06 =  -0.01752501463603003;
		double a10 =  -0.1450622084746213;
		double a11 =  -2.240107117581526;
		double a12 =  -1.266073104040587;
		double a13 = 3.2122097992271357;
		double a14 = 0.48660947608899774;
		double a15 =  -0.05146026788468628;
		double a16 = 0.0038834226652870174;
		double a20 =  -0.06601399450496376;
		double a21 =  -0.6065930112130667;
		double a22 =  -0.4441938708782571;
		double a23 = 0.7381006045405814;
		double a24 = 0.36835328297218667;
		double a25 = 0.010664600236304956;
		double a26 =  -0.0003176111527852762;

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