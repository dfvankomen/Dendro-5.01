// RE-EMITTED BYU_A6_1st_r_070_Op21.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,7.8e-08,4.5e-13,8.9e-11); (1,6,2.5e-11,1.1e-13,6.6e-14); (2,6,7.0e-11,1.1e-13,1.1e-13)
// yHat0 = 0.63
// yHat1 = 0.95
// yHat2 = 2.91


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP21_Diagonals(
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
		double gamma20 = 0.015560716034174783;
		double gamma21 = 0.340272618398345;
		double gamma23 = 0.495564207197939;
		double gamma24 =  - 0.013707285732348662;
		double a00 =  -3.7299692802083215;
		double a01 =  -11.349271662772786;
		double a02 = 10.748586385210015;
		double a03 = 5.2784076795611785;
		double a04 =  -1.1539648706176344;
		double a05 = 0.2309988976023282;
		double a06 =  -0.024787148774779617;
		double a10 =  -0.33350560412202807;
		double a11 =  -1.3196980957375213;
		double a12 = 0.7069875936801131;
		double a13 = 0.931416510992751;
		double a14 = 0.0115761091315302;
		double a15 = 0.0037751237888404266;
		double a16 =  -0.0005516377336852299;
		double a20 =  -0.06999038223229051;
		double a21 =  -0.6671566282288234;
		double a22 =  -0.2278839558584103;
		double a23 = 0.8882267905403884;
		double a24 = 0.08888506892504958;
		double a25 =  -0.012879507397873488;
		double a26 = 0.000798614251959641;

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