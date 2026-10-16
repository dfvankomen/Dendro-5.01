// RE-EMITTED BYU_A6_1st_r_085_Op6.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,4.3e-06,1.4e-12,1.8e-09); (1,6,7.6e-08,4.5e-13,2.0e-10); (2,6,8.6e-12,7.1e-15,2.8e-14)
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
		double a00 =  -3.8004165423361194;
		double a01 =  -13.121019622910113;
		double a02 = 11.649292266143059;
		double a03 = 6.445971670090237;
		double a04 =  -1.4257407639143247;
		double a05 = 0.28133105539455133;
		double a06 =  -0.029418062467290444;
		double a10 =  -0.231940999674043;
		double a11 =  -1.837062967240289;
		double a12 =  -0.5902824528410774;
		double a13 = 2.3451170776195602;
		double a14 = 0.3488108427594758;
		double a15 =  -0.037489705220643155;
		double a16 = 0.002848204597016502;
		double a20 =  -0.07480751762062213;
		double a21 =  -0.6233082313377891;
		double a22 =  -0.3873440212056466;
		double a23 = 0.728583193608605;
		double a24 = 0.3464453822679197;
		double a25 = 0.010805226770254632;
		double a26 =  -0.00037403248272151386;

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