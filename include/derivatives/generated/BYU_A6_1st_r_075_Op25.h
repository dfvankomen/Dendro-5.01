// yHat0 = 0.69
// yHat1 = 2.45
// yHat2 = 4.44


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP25_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 9.38983186534384;
		double gamma02 = 10.473862052331517;
		double gamma10 = 0.08587765270802927;
		double gamma12 = 1.3801196265129891;
		double gamma13 = 0.1680708408821211;
		double gamma20 = 0.01351531190986825;
		double gamma21 = 0.2665133986466341;
		double gamma23 = 1.0492427411810459;
		double gamma24 = 0.2637738937345838;
		double a00 =  - 3.6658432424748835;
		double a01 =  - 10.239829048154938;
		double a02 = 9.864826799485416;
		double a03 = 4.9820962941646725;
		double a04 =  - 1.162071138443952;
		double a05 = 0.2490569739806818;
		double a06 =  - 0.02823663869476687;
		double a10 =  - 0.33386410893781937;
		double a11 =  - 1.2949046415958065;
		double a12 = 0.9247946919660468;
		double a13 = 0.7460105199263364;
		double a14 =  - 0.05271454691050656;
		double a15 = 0.011858507316839366;
		double a16 =  - 0.0011804218779356112;
		double a20 =  - 0.057289228077722;
		double a21 =  - 0.5387170981247213;
		double a22 =  - 0.6734597850592533;
		double a23 = 0.6275478900085599;
		double a24 = 0.6122122397745537;
		double a25 = 0.031046504280180062;
		double a26 =  - 0.0013405228015573429;

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