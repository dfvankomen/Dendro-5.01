// RE-EMITTED BYU_A6_1st_r_075_Op15.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,5.7e-05,9.1e-13,9.0e-08); (1,6,7.4e-08,4.4e-17,6.3e-11); (2,6,6.7e-11,7.1e-15,1.7e-14)
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
		double a00 =  -3.986380686230779;
		double a01 =  -16.239291907740242;
		double a02 = 14.224750823440967;
		double a03 = 7.154373444607476;
		double a04 =  -1.3519344489289715;
		double a05 = 0.21638844067454818;
		double a06 =  -0.017905665822996873;
		double a10 =  -0.2341662018687464;
		double a11 =  -1.8230357966800523;
		double a12 =  -0.5057712776830394;
		double a13 = 2.278247609794947;
		double a14 = 0.3144990622479191;
		double a15 =  -0.032098386341554895;
		double a16 = 0.002324990530527158;
		double a20 =  -0.07265768713677044;
		double a21 =  -0.6202161757284537;
		double a22 =  -0.39826670243540974;
		double a23 = 0.7323229334463396;
		double a24 = 0.348719206044354;
		double a25 = 0.010437768217968716;
		double a26 =  -0.0003393424080285149;

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