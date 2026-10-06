// yHat0 = 4.17
// yHat1 = 2.45
// yHat2 = 1.44


MatrixDiagonalEntries* createBYU_A6_1ST_R060_OP12_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5288235111192885;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06232711462542066;
		double a1 = 0.690190101550906;
		double a2 = 0.19644389104322207;
		double a3 = 0.00269091403578636;
		double gamma01 = 10.773099653348766;
		double gamma02 = 12.721019019105636;
		double gamma10 = 0.0862025651064293;
		double gamma12 = 1.264277213952073;
		double gamma13 = 0.041866355889274995;
		double gamma20 = 0.0151011103390593;
		double gamma21 = 0.30919880265818833;
		double gamma23 = 0.7844200188413499;
		double gamma24 = 0.1354315949671251;
		double a00 =  - 3.8214826389480483;
		double a01 =  - 12.91388553874572;
		double a02 = 12.012154611357936;
		double a03 = 5.67285937905288;
		double a04 =  - 1.1329264691514902;
		double a05 = 0.20286095344902905;
		double a06 =  - 0.019580328417545975;
		double a10 =  - 0.3364181499714709;
		double a11 =  - 1.2655488750744235;
		double a12 = 1.0845859533606337;
		double a13 = 0.5937200525935493;
		double a14 =  - 0.0906651259728886;
		double a15 = 0.01573341995871301;
		double a16 =  - 0.0014072751544298514;
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