// yHat0 = 0.75
// yHat1 = -0.1
// yHat2 = 0.99


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 7.736066594739316;
		double gamma02 = 7.481743103051507;
		double gamma10 = 0.09305782737081193;
		double gamma12 = 1.4829137873091227;
		double gamma13 = 0.3594006535561008;
		double gamma20 = 0.07499088534084149;
		double gamma21 = 0.8073753219305838;
		double gamma23 =  - 0.07874810440556965;
		double gamma24 =  - 0.11165365154455495;
		double a00 =  - 3.489952995687311;
		double a01 =  - 6.920649371108901;
		double a02 = 7.475816343438309;
		double a03 = 3.7488798127232252;
		double a04 =  - 1.044149389211936;
		double a05 = 0.26354909837239154;
		double a06 =  - 0.03349349855704038;
		double a10 =  - 0.35121789504079215;
		double a11 =  - 1.2642417859987174;
		double a12 = 0.6674827619548951;
		double a13 = 0.9309372322175198;
		double a14 = 0.012460077205330468;
		double a15 = 0.005481133119428622;
		double a16 =  - 0.0009015234576839617;
		double a20 =  - 0.285505315192483;
		double a21 =  - 0.9831114132208124;
		double a22 = 0.8759075839687422;
		double a23 = 0.6365185677808316;
		double a24 =  - 0.23259542679120349;
		double a25 =  - 0.011370679697283984;
		double a26 = 0.00015668315227089114;

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