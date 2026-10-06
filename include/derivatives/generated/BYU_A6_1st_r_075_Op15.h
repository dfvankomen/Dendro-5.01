// yHat0 = 2.4
// yHat1 = 4.16
// yHat2 = 1.59


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP15_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 12.388586845434393;
		double gamma02 = 15.851513640248598;
		double gamma10 = 0.05474789128172447;
		double gamma12 = 2.6849462509375877;
		double gamma13 = 1.3719245955864618;
		double gamma20 = 0.016870975348053176;
		double gamma21 = 0.3253282913167522;
		double gamma23 = 0.7545972513844581;
		double gamma24 = 0.12845231250957087;
		double a00 =  - 3.9863806913019832;
		double a01 =  - 16.239291924049727;
		double a02 = 14.224750889230616;
		double a03 = 7.154373354588941;
		double a04 =  - 1.3519344137100775;
		double a05 = 0.2163884369080314;
		double a06 =  - 0.017905665124810323;
		double a10 =  - 0.23416620187115048;
		double a11 =  - 1.8230357966913064;
		double a12 =  - 0.5057712776477317;
		double a13 = 2.2782476097323716;
		double a14 = 0.3144990622392268;
		double a15 =  - 0.032098386341672065;
		double a16 = 0.0023249905306719585;
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