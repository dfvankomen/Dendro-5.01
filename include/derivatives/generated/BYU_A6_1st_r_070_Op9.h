// yHat0 = 0.63
// yHat1 = 0.95
// yHat2 = 0.96


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP9_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5406172854667417;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06788016563593934;
		double a1 = 0.6819178310623292;
		double a2 = 0.20847205673939975;
		double a3 = 0.003211835520517426;
		double gamma01 = 9.962102444245476;
		double gamma02 = 11.411433814977729;
		double gamma10 = 0.08637035120115655;
		double gamma12 = 1.5167106272387805;
		double gamma13 = 0.34731587522923224;
		double gamma20 = 0.0675299052145681;
		double gamma21 = 0.7491526942371628;
		double gamma23 = 0.07900981026542307;
		double gamma24 =  - 0.0387010993292;
		double a00 =  - 3.7299692802134503;
		double a01 =  - 11.349271662861987;
		double a02 = 10.748586385182973;
		double a03 = 5.27840767961304;
		double a04 =  - 1.1539648705689216;
		double a05 = 0.23099889758755368;
		double a06 =  - 0.02478714877319745;
		double a10 =  - 0.3335056041220351;
		double a11 =  - 1.3196980957375661;
		double a12 = 0.7069875936801275;
		double a13 = 0.9314165109926849;
		double a14 = 0.01157610913155957;
		double a15 = 0.0037751237888358925;
		double a16 =  - 0.0005516377336847323;
		double a20 =  - 0.2589355653084701;
		double a21 =  - 0.9392215748666244;
		double a22 = 0.7044662057866516;
		double a23 = 0.5865463434740799;
		double a24 =  - 0.09226151626659318;
		double a25 =  - 0.00025086523997586255;
		double a26 =  - 0.0003430275791248339;

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