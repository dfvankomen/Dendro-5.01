// RE-EMITTED BYU_A6_1st_r_085_Op9.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,2.1e-06,1.1e-13,1.7e-09); (1,6,5.1e-08,1.4e-14,9.2e-11); (2,6,3.7e-11,5.7e-14,4.8e-13)
// yHat0 = 1.62
// yHat1 = 2.45
// yHat2 = 2.13


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP9_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 8.952554944545723;
		double gamma02 = 8.795814814853614;
		double gamma10 = 0.08668410720178447;
		double gamma12 = 1.420582690132071;
		double gamma13 = 0.22198254741012227;
		double gamma20 =  - 0.03630276492668532;
		double gamma21 =  - 0.24707770213745506;
		double gamma23 = 2.31158307698918;
		double gamma24 = 0.6729666405565804;
		double a00 =  -3.6488986635958334;
		double a01 =  -9.007438104775122;
		double a02 = 9.750495386033032;
		double a03 = 3.4734948455619477;
		double a04 =  -0.6874449536387048;
		double a05 = 0.1346365725107179;
		double a06 =  -0.014845082096036149;
		double a10 =  -0.335389682096805;
		double a11 =  -1.2981643840639365;
		double a12 = 0.8547090495186499;
		double a13 = 0.8053376348546579;
		double a14 =  -0.035536503181802286;
		double a15 = 0.010134571881565841;
		double a16 =  -0.001090686912329893;
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