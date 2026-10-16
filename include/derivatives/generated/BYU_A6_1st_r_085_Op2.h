// RE-EMITTED BYU_A6_1st_r_085_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,2.1e-06,1.1e-13,1.7e-09); (1,6,7.6e-08,4.5e-13,2.0e-10); (2,6,1.4e-12,7.1e-15,1.8e-15)
// yHat0 = 1.62
// yHat1 = 4.07
// yHat2 = 0.51


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 8.952554944545723;
		double gamma02 = 8.795814814853614;
		double gamma10 = 0.054039540202692964;
		double gamma12 = 2.7386401072012054;
		double gamma13 = 1.449927785049125;
		double gamma20 = 0.02117883505295386;
		double gamma21 = 0.3500264050831283;
		double gamma23 = 0.7680147882716407;
		double gamma24 = 0.14489879940345124;
		double a00 =  -3.6488986635958334;
		double a01 =  -9.007438104775122;
		double a02 = 9.750495386033032;
		double a03 = 3.4734948455619477;
		double a04 =  -0.6874449536387048;
		double a05 = 0.1346365725107179;
		double a06 =  -0.014845082096036149;
		double a10 =  -0.231940999674043;
		double a11 =  -1.837062967240289;
		double a12 =  -0.5902824528410774;
		double a13 = 2.3451170776195602;
		double a14 = 0.3488108427594758;
		double a15 =  -0.037489705220643155;
		double a16 = 0.002848204597016502;
		double a20 =  -0.0872778132080615;
		double a21 =  -0.6262451645520056;
		double a22 =  -0.37067027502467137;
		double a23 = 0.6979498260098769;
		double a24 = 0.37280343030977375;
		double a25 = 0.013998635646730269;
		double a26 =  -0.0005586391816423946;

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