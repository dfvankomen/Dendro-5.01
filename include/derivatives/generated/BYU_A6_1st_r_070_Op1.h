// RE-EMITTED BYU_A6_1st_r_070_Op1.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,7.8e-08,4.5e-13,8.9e-11); (1,6,1.4e-10,1.1e-13,2.5e-14); (2,6,4.5e-13,2.8e-14,2.2e-16)
// yHat0 = 0.63
// yHat1 = -0.4
// yHat2 = 0.36


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5406172854667417;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06788016563593934;
		double a1 = 0.6819178310623292;
		double a2 = 0.20847205673939975;
		double a3 = 0.003211835520517426;
		double gamma01 = 9.962102444245476;
		double gamma02 = 11.411433814977729;
		double gamma10 = 0.09042919160461058;
		double gamma12 = 1.4163705035479883;
		double gamma13 = 0.25273991272421914;
		double gamma20 = 0.018299930368394172;
		double gamma21 = 0.32815026125517044;
		double gamma23 = 0.7998068257830566;
		double gamma24 = 0.15190369182615346;
		double a00 =  -3.7299692802083215;
		double a01 =  -11.349271662772786;
		double a02 = 10.748586385210015;
		double a03 = 5.2784076795611785;
		double a04 =  -1.1539648706176344;
		double a05 = 0.2309988976023282;
		double a06 =  -0.024787148774779617;
		double a10 =  -0.3452181678584333;
		double a11 =  -1.2693953982162323;
		double a12 = 0.8060100013525965;
		double a13 = 0.8246886154280549;
		double a14 =  -0.02440645241478611;
		double a15 = 0.009453443489964922;
		double a16 =  -0.0011320417811645812;
		double a20 =  -0.07699159184437585;
		double a21 =  -0.611609388109799;
		double a22 =  -0.42411043138257926;
		double a23 = 0.7058775112624724;
		double a24 = 0.39329941840037924;
		double a25 = 0.014046137324616615;
		double a26 =  -0.000511655650714185;

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