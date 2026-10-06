// yHat0 = 3.9
// yHat1 = 2.45
// yHat2 = 1.59


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP14_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 10.701581468084028;
		double gamma02 = 12.890848958458404;
		double gamma10 = 0.08587765270802927;
		double gamma12 = 1.3801196265129891;
		double gamma13 = 0.1680708408821211;
		double gamma20 = 0.016870975348053176;
		double gamma21 = 0.3253282913167522;
		double gamma23 = 0.7545972513844581;
		double gamma24 = 0.12845231250957087;
		double a00 =  - 3.8039019467881015;
		double a01 =  - 12.890035803787542;
		double a02 = 11.73429178418376;
		double a03 = 6.018496149014959;
		double a04 =  - 1.2774399144904924;
		double a05 = 0.24338449290809694;
		double a06 =  - 0.024794766851979216;
		double a10 =  - 0.33386410893781937;
		double a11 =  - 1.2949046415958065;
		double a12 = 0.9247946919660468;
		double a13 = 0.7460105199263364;
		double a14 =  - 0.05271454691050656;
		double a15 = 0.011858507316839366;
		double a16 =  - 0.0011804218779356112;
		double a20 =  - 0.07265768713677033;
		double a21 =  - 0.6202161757284681;
		double a22 =  - 0.3982667024354261;
		double a23 = 0.7323229334463562;
		double a24 = 0.34871920604433726;
		double a25 = 0.010437768217968178;
		double a26 =  - 0.00033934240802853134;

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