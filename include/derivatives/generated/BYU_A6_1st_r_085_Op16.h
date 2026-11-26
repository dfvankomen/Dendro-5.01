// RE-EMITTED BYU_A6_1st_r_085_Op16.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,2.1e-06,1.1e-13,1.7e-09); (1,6,3.7e-09,4.5e-13,2.4e-12); (2,6,4.5e-11,4.5e-13,2.9e-13)
// yHat0 = 1.62
// yHat1 = 2.87
// yHat2 = 3.33


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP16_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 8.952554944545723;
		double gamma02 = 8.795814814853614;
		double gamma10 = 0.04518472081989665;
		double gamma12 = 3.0141801893644007;
		double gamma13 = 1.6107677817154678;
		double gamma20 = 0.008951758212656707;
		double gamma21 = 0.21441473922607224;
		double gamma23 = 1.2130024721722463;
		double gamma24 = 0.322064175847782;
		double a00 =  -3.6488986635958334;
		double a01 =  -9.007438104775122;
		double a02 = 9.750495386033032;
		double a03 = 3.4734948455619477;
		double a04 =  -0.6874449536387048;
		double a05 = 0.1346365725107179;
		double a06 =  -0.014845082096036149;
		double a10 =  -0.2037426893918579;
		double a11 =  -1.9762819169023935;
		double a12 =  -0.8052330195650594;
		double a13 = 2.653471724618512;
		double a14 = 0.36487637186312133;
		double a15 =  -0.035502810358190744;
		double a16 = 0.00241233973586834;
		double a20 =  -0.039183235764095374;
		double a21 =  -0.48244655201805314;
		double a22 =  -0.8631544380683718;
		double a23 = 0.6062349215772149;
		double a24 = 0.7427324794313218;
		double a25 = 0.03734705789527918;
		double a26 =  -0.0015302330532956717;

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