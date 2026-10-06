// yHat0 = 0.57
// yHat1 = 0.89
// yHat2 = 2.94


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP15_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 10.366446898794267;
		double gamma02 = 12.072586130398253;
		double gamma10 = 0.08602524354762144;
		double gamma12 = 1.497475083239552;
		double gamma13 = 0.3219675134755059;
		double gamma20 = 0.01645516510846168;
		double gamma21 = 0.3584579212732618;
		double gamma23 = 0.3875867220777551;
		double gamma24 =  - 0.06309251107149001;
		double a00 =  - 3.7753216121109903;
		double a01 =  - 12.132641305499735;
		double a02 = 11.373775337563174;
		double a03 = 5.486036675841435;
		double a04 =  - 1.1475873162452228;
		double a05 = 0.21806642602666762;
		double a06 =  - 0.02232820554754837;
		double a10 =  - 0.3328788024749603;
		double a11 =  - 1.3178767783220673;
		double a12 = 0.7398079063964412;
		double a13 = 0.9034684013034955;
		double a14 = 0.003476763516665529;
		double a15 = 0.004598509667748392;
		double a16 =  - 0.0005960000873368163;
		double a20 =  - 0.074236128613826;
		double a21 =  - 0.6947396651955677;
		double a22 =  - 0.13283856555706447;
		double a23 = 0.929727913362122;
		double a24 =  - 0.009105857995626859;
		double a25 =  - 0.019909961595161806;
		double a26 = 0.0011022655947215759;

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