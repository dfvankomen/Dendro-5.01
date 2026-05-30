// yHat0 = 3.48
// yHat1 = 2.7
// yHat2 = 1.83


MatrixDiagonalEntries* create2A6_r085_Op21_Diagonals() {
		double alpha0 = 0.46751933569329174;
		double alpha = alpha0;

		double beta = 0.043965240266860744;
		double a1 = 0.316185773889511;
		double a2 = 0.39783363075189193;
		double a3 = 0.01282765055813622;
		double gamma01 = 13.896313729988425;
		double gamma02 = 13.429725549315968;
		double gamma10 = 0.04126308939422109;
		double gamma12 = 2.900425368148673;
		double gamma13 = 0.9073114071198459;
		double gamma20 = 0.010225812964810238;
		double gamma21 = 0.2768837819049139;
		double gamma23 = 0.7402479122719887;
		double gamma24 = 0.12136541630606675;
		double a00 = 14.117825282962194;
		double a01 =  - 11.737670574715825;
		double a02 =  - 21.77247053361235;
		double a03 = 22.9845139861464;
		double a04 =  - 4.383352891340728;
		double a05 = 0.8844470532376243;
		double a06 =  - 0.09329237443494652;
		double a10 = 0.7478595645333916;
		double a11 = 2.003130999807058;
		double a12 =  - 5.616413383537435;
		double a13 = 2.1993666112175396;
		double a14 = 0.7001768663713888;
		double a15 =  - 0.035612418343240554;
		double a16 = 0.0014917599866415105;
		double a20 = 0.19422036058641434;
		double a21 = 0.7434875512620778;
		double a22 =  - 1.3059946753777503;
		double a23 =  - 0.3347785895443821;
		double a24 = 0.6408458229304327;
		double a25 = 0.06386290445444479;
		double a26 =  - 0.0016433743112202997;

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