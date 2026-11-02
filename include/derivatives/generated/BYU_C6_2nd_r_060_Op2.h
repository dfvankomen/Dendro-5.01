// RE-EMITTED BYU_C6_2nd_r_060_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,5,7.1e-09,2.5e-12,1.3e-10); (1,5,1.5e-11,1.8e-12,1.7e-13)
// yHat0 = 1.2
// yHat1 = -0.02


MatrixDiagonalEntries* createBYU_C6_2ND_R060_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3181041957525139;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.012089888332884303;
		double a1 = 0.7479958945581906;
		double a2 = 0.22809806840315142;
		double gamma01 = 10.000000000022066;
		double gamma02 = 2.1250000000657745;
		double gamma10 =  - 1.6778488611611477;
		double gamma12 = 99.91637626148137;
		double gamma13 = 17.778488611611888;
		double a00 = 11.90625000002166;
		double a01 =  -22.499999999967464;
		double a02 = 9.187499999865745;
		double a03 = 1.500000000085472;
		double a04 =  -0.09375000000541103;
		double a10 =  -13.784964584477915;
		double a11 = 152.0226880159094;
		double a12 =  -256.34126052892873;
		double a13 = 111.75431534804089;
		double a14 = 6.349221749456362;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04},
			{a10, a11, a12, a13, a14}
		};

		double t1 = -2.0 * (a1 + a2);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a2, a1, t1, a1, a2
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}