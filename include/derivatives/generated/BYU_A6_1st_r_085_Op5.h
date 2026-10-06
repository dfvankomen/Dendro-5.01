// yHat0 = 1.62
// yHat1 = 1.55
// yHat2 = 1.62


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP5_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 8.952554944545723;
		double gamma02 = 8.795814814853614;
		double gamma10 =  - 0.05447113958159977;
		double gamma12 = 5.452379009558196;
		double gamma13 = 3.2784359989219225;
		double gamma20 = 0.017470950314353594;
		double gamma21 = 0.32993132599730224;
		double gamma23 = 0.7494112999230135;
		double gamma24 = 0.1285031988686864;
		double a00 =  - 3.6488986635805327;
		double a01 =  - 9.007438105441317;
		double a02 = 9.750495385801761;
		double a03 = 3.4734948472555742;
		double a04 =  - 0.6874449537666463;
		double a05 = 0.13463657268969775;
		double a06 =  - 0.014845082125668214;
		double a10 = 0.09389299230696006;
		double a11 =  - 3.299346374806078;
		double a12 =  - 2.730847874451314;
		double a13 = 5.240031082019511;
		double a14 = 0.7702376012017559;
		double a15 =  - 0.08014689941629337;
		double a16 = 0.006179473089061288;
		double a20 =  - 0.07480751762061762;
		double a21 =  - 0.6233082313377792;
		double a22 =  - 0.38734402120564443;
		double a23 = 0.7285831936085773;
		double a24 = 0.3464453822679065;
		double a25 = 0.010805226770262046;
		double a26 =  - 0.00037403248272221257;

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