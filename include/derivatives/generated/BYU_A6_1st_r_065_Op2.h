// yHat0 = 1.8
// yHat1 = -0.07
// yHat2 = 0.3


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 10.121749695600322;
		double gamma02 = 11.294215822998105;
		double gamma10 = 0.0010971777122586177;
		double gamma12 = 4.238503847119178;
		double gamma13 = 2.5957825017204654;
		double gamma20 = 0.017398619496701056;
		double gamma21 = 0.3209456397245266;
		double gamma23 = 0.8107161220661434;
		double gamma24 = 0.15423845651388887;
		double a00 =  - 3.7604844225108827;
		double a01 =  - 11.507265113037956;
		double a02 = 11.216081672863723;
		double a03 = 4.856038277703331;
		double a04 =  - 0.962316498318217;
		double a05 = 0.17545801873727876;
		double a06 =  - 0.017511940465420973;
		double a10 =  - 0.07133433218646358;
		double a11 =  - 2.5827844306497765;
		double a12 =  - 1.9275262866198486;
		double a13 = 3.9919863142418817;
		double a14 = 0.6568038696430823;
		double a15 =  - 0.07291691572064665;
		double a16 = 0.005771781291289922;
		double a20 =  - 0.07372551881354295;
		double a21 =  - 0.6064462298915332;
		double a22 =  - 0.44237674353993856;
		double a23 = 0.7087634550852783;
		double a24 = 0.4002194011738521;
		double a25 = 0.014063231093877006;
		double a26 =  - 0.0004975951079931993;

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