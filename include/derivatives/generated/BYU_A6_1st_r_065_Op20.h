// RE-EMITTED BYU_A6_1st_r_065_Op20.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,1.4e-07,4.5e-13,1.1e-10); (1,6,6.1e-11,1.1e-13,5.9e-14); (2,6,6.6e-11,2.8e-14,4.3e-14)
// yHat0 = 0.57
// yHat1 = 0.89
// yHat2 = 4.53


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP20_Diagonals(
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
		double gamma20 = 0.0133156110798937;
		double gamma21 = 0.2717451732928028;
		double gamma23 = 0.9942636649673103;
		double gamma24 = 0.23259939213577063;
		double a00 =  -3.775321612119103;
		double a01 =  -12.132641305611944;
		double a02 = 11.373775337586688;
		double a03 = 5.486036675873892;
		double a04 =  -1.147587316203904;
		double a05 = 0.2180664260212003;
		double a06 =  -0.02232820554682865;
		double a10 =  -0.3328788024749459;
		double a11 =  -1.3178767783220997;
		double a12 = 0.7398079063964711;
		double a13 = 0.9034684013035457;
		double a14 = 0.0034767635166063566;
		double a15 = 0.004598509667760109;
		double a16 =  -0.0005960000873376748;
		double a20 =  -0.0572751805750657;
		double a21 =  -0.5507196751194076;
		double a22 =  -0.6332355358581266;
		double a23 = 0.6590629288635925;
		double a24 = 0.5579014969990832;
		double a25 = 0.02527598041521679;
		double a26 =  -0.0010100147252927075;

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