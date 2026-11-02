// RE-EMITTED BYU_A6_1st_r_085_Op26.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,4.3e-06,1.4e-12,1.8e-09); (1,6,2.2e-07,2.8e-14,1.8e-10); (2,6,9.5e-11,4.5e-13,9.7e-14)
// yHat0 = 3.81
// yHat1 = 1.55
// yHat2 = 4.41


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP26_Diagonals(
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
		double gamma20 = 0.013536681423743759;
		double gamma21 = 0.25970978533202643;
		double gamma23 = 1.1020770961543107;
		double gamma24 = 0.294115552327983;
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
		double a20 =  -0.05658252610728207;
		double a21 =  -0.5259779785208892;
		double a22 =  -0.7150840266330869;
		double a23 = 0.5985741641676037;
		double a24 = 0.6637874934113728;
		double a25 = 0.03698456121686581;
		double a26 =  -0.0017016875345839978;

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