// RE-EMITTED BYU_A6_1st_r_065_Op6.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,1.6e-06,4.5e-13,8.7e-09); (1,6,1.8e-09,4.5e-13,3.9e-11); (2,6,1.8e-12,1.1e-13,6.7e-16)
// yHat0 = 2.46
// yHat1 = 4.28
// yHat2 = 0.3


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP6_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 11.42421093751791;
		double gamma02 = 13.881711866493903;
		double gamma10 = 0.05502329701552567;
		double gamma12 = 2.646322046300063;
		double gamma13 = 1.3129631508950041;
		double gamma20 = 0.017398619496701056;
		double gamma21 = 0.3209456397245266;
		double gamma23 = 0.8107161220661434;
		double gamma24 = 0.15423845651388887;
		double a00 =  -3.891311427369855;
		double a01 =  -14.213755449745546;
		double a02 = 12.96286208834;
		double a03 = 6.13526425946202;
		double a04 =  -1.170680151982026;
		double a05 = 0.19484218115304275;
		double a06 =  -0.017221499857634697;
		double a10 =  -0.2351457286596192;
		double a11 =  -1.8147778971259538;
		double a12 =  -0.44108495112939905;
		double a13 = 2.2285847085035884;
		double a14 = 0.2885573095463337;
		double a15 =  -0.02807357670894476;
		double a16 = 0.0019401355739947417;
		double a20 =  -0.07372551881354293;
		double a21 =  -0.6064462298915331;
		double a22 =  -0.4423767435399379;
		double a23 = 0.7087634550852776;
		double a24 = 0.4002194011738526;
		double a25 = 0.014063231093877004;
		double a26 =  -0.0004975951079931973;

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