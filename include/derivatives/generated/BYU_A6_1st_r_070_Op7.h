// yHat0 = 2.4
// yHat1 = -0.4
// yHat2 = 0.96


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP7_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5406172854667417;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06788016563593934;
		double a1 = 0.6819178310623292;
		double a2 = 0.20847205673939975;
		double a3 = 0.003211835520517426;
		double gamma01 = 12.598094593642406;
		double gamma02 = 16.23386410161242;
		double gamma10 = 0.09042919160461058;
		double gamma12 = 1.4163705035479883;
		double gamma13 = 0.25273991272421914;
		double gamma20 = 0.0675299052145681;
		double gamma21 = 0.7491526942371628;
		double gamma23 = 0.07900981026542307;
		double gamma24 =  - 0.0387010993292;
		double a00 =  - 4.008553625946629;
		double a01 =  - 16.66110032629068;
		double a02 = 14.525482474123672;
		double a03 = 7.314994589771871;
		double a04 =  - 1.3685199243478516;
		double a05 = 0.21499159196683704;
		double a06 =  - 0.0172945847764923;
		double a10 =  - 0.3452181678584214;
		double a11 =  - 1.269395398216239;
		double a12 = 0.8060100013525732;
		double a13 = 0.824688615428068;
		double a14 =  - 0.024406452414810625;
		double a15 = 0.009453443489960625;
		double a16 =  - 0.0011320417811641725;
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