// yHat0 = 1.74
// yHat1 = 0.11
// yHat2 = 0.42


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 9.440989783371029;
		double gamma02 = 9.855215182150074;
		double gamma10 =  - 0.007948158861142196;
		double gamma12 = 4.22437939215253;
		double gamma13 = 2.5136963321191033;
		double gamma20 = 0.01926386345301328;
		double gamma21 = 0.33570565454932;
		double gamma23 = 0.7884098090152796;
		double gamma24 = 0.14936803315928954;
		double a00 =  - 3.6949911246980953;
		double a01 =  - 10.05802296180195;
		double a02 = 10.353598938854793;
		double a03 = 4.071970600580927;
		double a04 =  - 0.8101161037091037;
		double a05 = 0.15378124508049626;
		double a06 =  - 0.016220593576525524;
		double a10 =  - 0.04827596992045357;
		double a11 =  - 2.64371959354227;
		double a12 =  - 1.7898823697200845;
		double a13 = 3.9128514637962737;
		double a14 = 0.6362214820754765;
		double a15 =  - 0.07334165483086778;
		double a16 = 0.0061466421429616815;
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