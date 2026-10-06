// yHat0 = 3.81
// yHat1 = 0.2
// yHat2 = 2.13


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP7_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 10.759193339373669;
		double gamma02 = 13.283470426784763;
		double gamma10 =  - 0.358154555727302;
		double gamma12 = 12.499696290881529;
		double gamma13 = 8.628864243686683;
		double gamma20 =  - 0.03630276492668532;
		double gamma21 =  - 0.24707770213745506;
		double gamma23 = 2.31158307698918;
		double gamma24 = 0.6729666405565804;
		double a00 =  - 3.8004165422509897;
		double a01 =  - 13.121019623413277;
		double a02 = 11.649292264421266;
		double a03 = 6.445971671909907;
		double a04 =  - 1.4257407630570507;
		double a05 = 0.28133105538866665;
		double a06 =  - 0.029418062475013333;
		double a10 = 0.9836541338334029;
		double a11 =  - 7.137809547494393;
		double a12 =  - 8.576978517827321;
		double a13 = 12.611898016330098;
		double a14 = 2.3982129546338222;
		double a15 =  - 0.3074889313075378;
		double a16 = 0.028511891834549625;
		double a20 = 0.1361444504860287;
		double a21 =  - 0.043724962342838763;
		double a22 =  - 2.3259608391904605;
		double a23 = 0.605822216642898;
		double a24 = 1.556488464760409;
		double a25 = 0.07398863562990278;
		double a26 =  - 0.002757965985532601;

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