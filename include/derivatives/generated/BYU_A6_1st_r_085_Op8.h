// RE-EMITTED BYU_A6_1st_r_085_Op8.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,4.3e-06,1.4e-12,1.8e-09); (1,6,2.2e-07,2.8e-14,1.8e-10); (2,6,3.7e-11,5.7e-14,4.8e-13)
// yHat0 = 3.81
// yHat1 = 1.55
// yHat2 = 2.13


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP8_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 10.759193339373669;
		double gamma02 = 13.283470426784763;
		double gamma10 =  - 0.05447113958159977;
		double gamma12 = 5.452379009558196;
		double gamma13 = 3.2784359989219225;
		double gamma20 =  - 0.03630276492668532;
		double gamma21 =  - 0.24707770213745506;
		double gamma23 = 2.31158307698918;
		double gamma24 = 0.6729666405565804;
		double a00 =  -3.8004165423361194;
		double a01 =  -13.121019622910113;
		double a02 = 11.649292266143059;
		double a03 = 6.445971670090237;
		double a04 =  -1.4257407639143247;
		double a05 = 0.28133105539455133;
		double a06 =  -0.029418062467290444;
		double a10 = 0.09389299231149391;
		double a11 =  -3.299346374807922;
		double a12 =  -2.7308478745717246;
		double a13 = 5.240031082200263;
		double a14 = 0.7702376011766765;
		double a15 =  -0.08014689939511532;
		double a16 = 0.006179473086328739;
		double a20 = 0.1361444504860782;
		double a21 =  -0.0437249623428783;
		double a22 =  -2.325960839190426;
		double a23 = 0.6058222166424158;
		double a24 = 1.5564884647604145;
		double a25 = 0.07398863562992986;
		double a26 =  -0.0027579659855339602;

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