// RE-EMITTED BYU_C6_2nd_r_075_Op1.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,5,1.3e-08,9.5e-13,8.0e-11); (1,5,1.3e-12,1.1e-13,5.4e-14)
// yHat0 = 1.38
// yHat1 = -0.83


MatrixDiagonalEntries* createBYU_C6_2ND_R075_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.3372311995327969;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.013786638668231979;
		double a1 = 0.6998698850465109;
		double a2 = 0.2505414478388867;
		double gamma01 = 9.999999999991703;
		double gamma02 = 3.7580645161701454;
		double gamma10 = 0.06685704385209638;
		double gamma12 = 1.6951727216964372;
		double gamma13 = 0.3314295614790526;
		double a00 = 11.770161290308948;
		double a01 =  -20.32258064508573;
		double a02 = 5.104838709563287;
		double a03 = 3.6774193548943646;
		double a04 =  -0.22983870968087183;
		double a10 = 0.9427828543039851;
		double a11 = 0.12461243603677884;
		double a12 =  -2.9370751069066654;
		double a13 = 1.7291814884870527;
		double a14 = 0.14049832807884874;

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