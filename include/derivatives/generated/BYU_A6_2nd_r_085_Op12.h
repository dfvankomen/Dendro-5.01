// yHat0 = 1.68
// yHat1 = 4.05
// yHat2 = 0.78


MatrixDiagonalEntries* create2A6_r085_Op12_Diagonals() {
		double alpha0 = 0.46751933569329174;
		double alpha = alpha0;

		double beta = 0.043965240266860744;
		double a1 = 0.316185773889511;
		double a2 = 0.39783363075189193;
		double a3 = 0.01282765055813622;
		double gamma01 = 16.08195466751146;
		double gamma02 = 25.45075067166764;
		double gamma10 = 0.0385408314149095;
		double gamma12 = 3.0719276208800346;
		double gamma13 = 0.9501313668819447;
		double gamma20 = 0.012572751185689146;
		double gamma21 = 0.30138776988943455;
		double gamma23 = 0.7789485264856879;
		double gamma24 = 0.13445059030652365;
		double a00 = 14.91315572615458;
		double a01 = 1.7040212064102573;
		double a02 =  - 52.91785401694716;
		double a03 = 42.04816016140869;
		double a04 =  - 6.84219900201827;
		double a05 = 1.212293200252631;
		double a06 =  - 0.11757727409285099;
		double a10 = 0.7236686598441009;
		double a11 = 2.2613114814533946;
		double a12 =  - 6.031981412864531;
		double a13 = 2.3501873934235173;
		double a14 = 0.733781403731932;
		double a15 =  - 0.03876870272707751;
		double a16 = 0.0018011771072184964;
		double a20 = 0.22403331484009387;
		double a21 = 0.6759617989660119;
		double a22 =  - 1.2129193629078734;
		double a23 =  - 0.4278443560663925;
		double a24 = 0.6682911712522429;
		double a25 = 0.07461929345556308;
		double a26 =  - 0.0021418595396481576;

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