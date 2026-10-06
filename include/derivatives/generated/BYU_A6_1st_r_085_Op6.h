// yHat0 = 3.81
// yHat1 = 4.07
// yHat2 = 1.62


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP6_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 10.759193339373669;
		double gamma02 = 13.283470426784763;
		double gamma10 = 0.054039540202692964;
		double gamma12 = 2.7386401072012054;
		double gamma13 = 1.449927785049125;
		double gamma20 = 0.017470950314353594;
		double gamma21 = 0.32993132599730224;
		double gamma23 = 0.7494112999230135;
		double gamma24 = 0.1285031988686864;
		double a00 =  - 3.8004165422509897;
		double a01 =  - 13.121019623413277;
		double a02 = 11.649292264421266;
		double a03 = 6.445971671909907;
		double a04 =  - 1.4257407630570507;
		double a05 = 0.28133105538866665;
		double a06 =  - 0.029418062475013333;
		double a10 =  - 0.2319409996791471;
		double a11 =  - 1.8370629671765395;
		double a12 =  - 0.5902824527907709;
		double a13 = 2.3451170774159613;
		double a14 = 0.34881084282844743;
		double a15 =  - 0.037489705242215836;
		double a16 = 0.002848204599673139;
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