// RE-EMITTED BYU_A6_1st_r_070_Op14.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,7.8e-08,4.5e-13,8.9e-11); (1,6,9.1e-08,4.5e-13,2.4e-11); (2,6,4.2e-11,4.3e-14,4.8e-14)
// yHat0 = 0.63
// yHat1 = 4.22
// yHat2 = 0.96


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP14_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5406172854667417;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06788016563593934;
		double a1 = 0.6819178310623292;
		double a2 = 0.20847205673939975;
		double a3 = 0.003211835520517426;
		double gamma01 = 9.962102444245476;
		double gamma02 = 11.411433814977729;
		double gamma10 = 0.055108229889549915;
		double gamma12 = 2.6564377716823895;
		double gamma13 = 1.3331110367019765;
		double gamma20 = 0.0675299052145681;
		double gamma21 = 0.7491526942371628;
		double gamma23 = 0.07900981026542307;
		double gamma24 =  - 0.0387010993292;
		double a00 =  -3.7299692802083215;
		double a01 =  -11.349271662772786;
		double a02 = 10.748586385210015;
		double a03 = 5.2784076795611785;
		double a04 =  -1.1539648706176344;
		double a05 = 0.2309988976023282;
		double a06 =  -0.024787148774779617;
		double a10 =  -0.23535242145168392;
		double a11 =  -1.8152924071636931;
		double a12 =  -0.4627337018461673;
		double a13 = 2.242638561506852;
		double a14 = 0.2982918629328088;
		double a15 =  -0.029645076746851306;
		double a16 = 0.002093182768734797;
		double a20 =  -0.2589355653084627;
		double a21 =  -0.9392215748665769;
		double a22 = 0.7044662057866456;
		double a23 = 0.586546343474116;
		double a24 =  -0.0922615162666274;
		double a25 =  -0.00025086523996910835;
		double a26 =  -0.00034302757912553874;

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