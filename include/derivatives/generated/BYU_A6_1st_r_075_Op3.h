// yHat0 = 0.69
// yHat1 = 1.01
// yHat2 = 0.42


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 9.38983186534384;
		double gamma02 = 10.473862052331517;
		double gamma10 = 0.08730912312407724;
		double gamma12 = 1.5181386531660943;
		double gamma13 = 0.3591335737776352;
		double gamma20 = 0.01926386345301328;
		double gamma21 = 0.33570565454932;
		double gamma23 = 0.7884098090152796;
		double gamma24 = 0.14936803315928954;
		double a00 =  - 3.6658432424748835;
		double a01 =  - 10.239829048154938;
		double a02 = 9.864826799485416;
		double a03 = 4.9820962941646725;
		double a04 =  - 1.162071138443952;
		double a05 = 0.2490569739806818;
		double a06 =  - 0.02823663869476687;
		double a10 =  - 0.3359549561114084;
		double a11 =  - 1.3128640197886536;
		double a12 = 0.6902505152228151;
		double a13 = 0.9395790250483181;
		double a14 = 0.016204975368311607;
		double a15 = 0.00331939877105533;
		double a16 =  - 0.0005349385104855422;
		double a20 =  - 0.08046477081570226;
		double a21 =  - 0.6168933423558267;
		double a22 =  - 0.40517151303944776;
		double a23 = 0.7030922878921699;
		double a24 = 0.38595393361334557;
		double a25 = 0.014009297751043197;
		double a26 =  - 0.000525893045579879;

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