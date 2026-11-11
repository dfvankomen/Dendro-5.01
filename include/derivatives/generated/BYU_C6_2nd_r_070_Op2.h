// RE-EMITTED BYU_C6_2nd_r_070_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,5,2.0e-08,3.8e-13,1.2e-10); (1,5,1.5e-11,1.1e-13,1.4e-12)
// yHat0 = 1.35
// yHat1 = 0.07


MatrixDiagonalEntries* createBYU_C6_2ND_R070_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.32995227499332586;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.013140927620375692;
		double a1 = 0.7181845984038895;
		double a2 = 0.2420004517058784;
		double gamma01 = 10.000000000003332;
		double gamma02 = 3.4230769230261777;
		double gamma10 =  - 3.071130791716968;
		double gamma12 = 178.00048366636418;
		double gamma13 = 31.711307917170135;
		double a00 = 11.798076923085251;
		double a01 =  -20.769230769306763;
		double a02 = 5.942307692439114;
		double a03 = 3.230769230701234;
		double a04 =  -0.20192307691883754;
		double a10 =  -25.516780774469158;
		double a11 = 272.8542143890897;
		double a12 =  -457.821297728637;
		double a13 = 199.1470753878814;
		double a14 = 11.336788726135016;

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