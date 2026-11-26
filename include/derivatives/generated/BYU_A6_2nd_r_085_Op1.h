// RE-EMITTED BYU_A6_2nd_r_085_Op1.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,2.4e-07,9.9e-12,3.4e-09); (1,7,3.4e-08,4.5e-13,1.6e-11); (2,7,3.1e-11,4.5e-13,1.0e-14)
MatrixDiagonalEntries* createBYU_A6_2ND_R085_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.46751933569329174;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.043965240266860744;
		double a1 = 0.316185773889511;
		double a2 = 0.39783363075189193;
		double a3 = 0.01282765055813622;
		double gamma01 = 13.86655615586013;
		double gamma02 = 13.266058857914487;
		double gamma10 = 0.04126308939422109;
		double gamma12 = 2.900425368148673;
		double gamma13 = 0.9073114071198459;
		double gamma20 = 0.012572751185689146;
		double gamma21 = 0.30138776988943455;
		double gamma23 = 0.7789485264856879;
		double gamma24 = 0.13445059030652365;
		double a00 = 14.106996823290586;
		double a01 =  -11.92067964043454;
		double a02 =  -21.348425222876475;
		double a03 = 22.72496202714764;
		double a04 =  -4.349875675449948;
		double a05 = 0.879983423388031;
		double a06 =  -0.09296173506529368;
		double a10 = 0.7478595645365922;
		double a11 = 2.0031309997906357;
		double a12 =  -5.616413383546661;
		double a13 = 2.1993666112069676;
		double a14 = 0.7001768663671726;
		double a15 =  -0.035612418341245136;
		double a16 = 0.0014917599865381002;
		double a20 = 0.22403331484009176;
		double a21 = 0.6759617989660165;
		double a22 =  -1.2129193629078836;
		double a23 =  -0.42784435606638216;
		double a24 = 0.6682911712522422;
		double a25 = 0.07461929345556344;
		double a26 =  -0.002141859539648183;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}