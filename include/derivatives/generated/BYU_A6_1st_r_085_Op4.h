// RE-EMITTED BYU_A6_1st_r_085_Op4.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,2.1e-06,1.1e-13,1.7e-09); (1,6,6.1e-10,1.1e-13,2.5e-13); (2,6,7.8e-12,1.1e-13,1.2e-13)
// yHat0 = 1.62
// yHat1 = 1.1
// yHat2 = 0.99


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP4_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 8.952554944545723;
		double gamma02 = 8.795814814853614;
		double gamma10 = 0.08927158188544428;
		double gamma12 = 1.5202629960086784;
		double gamma13 = 0.3835243092035516;
		double gamma20 = 0.07499088534084149;
		double gamma21 = 0.8073753219305838;
		double gamma23 =  - 0.07874810440556965;
		double gamma24 =  - 0.11165365154455495;
		double a00 =  -3.6488986635958334;
		double a01 =  -9.007438104775122;
		double a02 = 9.750495386033032;
		double a03 = 3.4734948455619477;
		double a04 =  -0.6874449536387048;
		double a05 = 0.1346365725107179;
		double a06 =  -0.014845082096036149;
		double a10 =  -0.34109868090577505;
		double a11 =  -1.2982803940436063;
		double a12 = 0.6559998229514418;
		double a13 = 0.9554945405811998;
		double a14 = 0.0260766351612418;
		double a15 = 0.0022989846831575128;
		double a16 =  -0.000490908427659493;
		double a20 =  -0.28550531519247535;
		double a21 =  -0.9831114132207618;
		double a22 = 0.8759075839687148;
		double a23 = 0.6365185677807101;
		double a24 =  -0.23259542679117004;
		double a25 =  -0.011370679697289363;
		double a26 = 0.0001566831522715495;

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