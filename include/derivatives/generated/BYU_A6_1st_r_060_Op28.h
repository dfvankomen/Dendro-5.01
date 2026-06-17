// yHat0 = 1.83
// yHat1 = 0.8
// yHat2 = 4.62


MatrixDiagonalEntries* createBYU_A6_1ST_R060_OP28_Diagonals(
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
		double gamma20 = 0.013129241033869962;
		double gamma21 = 0.2732407709002383;
		double gamma23 = 0.9695814338212047;
		double gamma24 = 0.21841727699269886;
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
		double a20 =  - 0.05689283829683436;
		double a21 =  - 0.5555686316445545;
		double a22 =  - 0.616678150706283;
		double a23 = 0.6742372860684218;
		double a24 = 0.5330621421515437;
		double a25 = 0.02270792557280765;
		double a26 =  - 0.0008677331450390108;

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