// yHat0 = 1.83
// yHat1 = 0.8
// yHat2 = 2.94


MatrixDiagonalEntries* createBYU_A6_1ST_R060_OP19_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5288235111192885;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06232711462542066;
		double a1 = 0.690190101550906;
		double a2 = 0.19644389104322207;
		double a3 = 0.00269091403578636;
		double gamma01 = 10.493343500167311;
		double gamma02 = 12.072599632102033;
		double gamma10 = 0.08488950270043998;
		double gamma12 = 1.5052345758504946;
		double gamma13 = 0.31887417201174134;
		double gamma20 = 0.015878054775182004;
		double gamma21 = 0.3514797121414662;
		double gamma23 = 0.4136202360125774;
		double gamma24 =  - 0.05146447868611237;
		double a00 =  - 3.7964705971207646;
		double a01 =  - 12.295497335570657;
		double a02 = 11.691008982939273;
		double a03 = 5.274560321329439;
		double a04 =  - 1.0418468908354217;
		double a05 = 0.1863440748235999;
		double a06 =  - 0.018098543698789944;
		double a10 =  - 0.32978603195463413;
		double a11 =  - 1.3282590216689252;
		double a12 = 0.7461195981584208;
		double a13 = 0.9062427858035514;
		double a14 = 0.0015360392903747354;
		double a15 = 0.0047342208854575804;
		double a16 =  - 0.0005875905140482593;
		double a20 =  - 0.0718992648010697;
		double a21 =  - 0.6868923360366489;
		double a22 =  - 0.1596668801458705;
		double a23 = 0.9220068165127813;
		double a24 = 0.013551285820405864;
		double a25 =  - 0.018111755848191725;
		double a26 = 0.0010121344986069904;

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