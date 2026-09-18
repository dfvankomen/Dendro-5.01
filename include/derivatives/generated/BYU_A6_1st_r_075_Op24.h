// RE-EMITTED BYU_A6_1st_r_075_Op24.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,5.7e-05,9.1e-13,9.0e-08); (1,6,1.4e-10,3.6e-15,3.2e-13); (2,6,1.2e-10,1.1e-13,7.3e-14)
// yHat0 = 2.4
// yHat1 = 1.01
// yHat2 = 4.44


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP24_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 12.388586845434393;
		double gamma02 = 15.851513640248598;
		double gamma10 = 0.08730912312407724;
		double gamma12 = 1.5181386531660943;
		double gamma13 = 0.3591335737776352;
		double gamma20 = 0.01351531190986825;
		double gamma21 = 0.2665133986466341;
		double gamma23 = 1.0492427411810459;
		double gamma24 = 0.2637738937345838;
		double a00 =  -3.986380686230779;
		double a01 =  -16.239291907740242;
		double a02 = 14.224750823440967;
		double a03 = 7.154373444607476;
		double a04 =  -1.3519344489289715;
		double a05 = 0.21638844067454818;
		double a06 =  -0.017905665822996873;
		double a10 =  -0.33595495611141335;
		double a11 =  -1.3128640197886623;
		double a12 = 0.6902505152226392;
		double a13 = 0.9395790250486407;
		double a14 = 0.01620497536822295;
		double a15 = 0.0033193987710599775;
		double a16 =  -0.0005349385104871911;
		double a20 =  -0.05728922807772393;
		double a21 =  -0.5387170981247585;
		double a22 =  -0.6734597850592525;
		double a23 = 0.6275478900086198;
		double a24 = 0.6122122397744808;
		double a25 = 0.031046504280193354;
		double a26 =  -0.0013405228015589343;

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