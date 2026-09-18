// RE-EMITTED BYU_A6_1st_r_065_Op12.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,1.5e-06,4.5e-13,7.8e-09); (1,6,2.1e-09,9.1e-13,1.4e-12); (2,6,2.4e-11,2.8e-14,4.0e-14)
// yHat0 = 1.8
// yHat1 = -0.07
// yHat2 = 1.5


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP12_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 10.121749695600322;
		double gamma02 = 11.294215822998105;
		double gamma10 = 0.0010971777122586177;
		double gamma12 = 4.238503847119178;
		double gamma13 = 2.5957825017204654;
		double gamma20 = 0.015676071490883525;
		double gamma21 = 0.31443161599328323;
		double gamma23 = 0.7753042511904301;
		double gamma24 = 0.1335131522459523;
		double a00 =  -3.7604844218334503;
		double a01 =  -11.507265105219656;
		double a02 = 11.216081675585244;
		double a03 = 4.85603827133027;
		double a04 =  -0.9623164984987843;
		double a05 = 0.17545801916633352;
		double a06 =  -0.01751194052995769;
		double a10 =  -0.0713343321864021;
		double a11 =  -2.582784430649383;
		double a12 =  -1.927526286618476;
		double a13 = 3.991986314240628;
		double a14 = 0.6568038696431237;
		double a15 =  -0.07291691572080242;
		double a16 = 0.005771781291311689;
		double a20 =  -0.06817482946728647;
		double a21 =  -0.6109702608669748;
		double a22 =  -0.4295464418015981;
		double a23 = 0.735770246955815;
		double a24 = 0.3626019391065509;
		double a25 = 0.010646338343889164;
		double a26 =  -0.0003269922703957217;

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