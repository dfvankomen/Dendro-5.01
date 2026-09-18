// RE-EMITTED BYU_A6_2nd_r_070_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,1.5e-03,1.6e-11,4.7e-07); (1,7,9.1e-08,1.8e-12,3.7e-11); (2,7,2.7e-11,8.9e-16,4.0e-14)
MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.4309329692460775;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.035593529920944555;
		double a1 = 0.42362022943349215;
		double a2 = 0.35644150411900966;
		double a3 = 0.009296305824945953;
		double gamma01 = 13.503763237286865;
		double gamma02 = 11.270697805076875;
		double gamma10 = 0.03720859758781716;
		double gamma12 = 3.1558583519997545;
		double gamma13 = 1.153395432691522;
		double gamma20 = 0.010065992847351608;
		double gamma21 = 0.27872724988737285;
		double gamma23 = 0.6702657731502768;
		double gamma24 = 0.10115767500261996;
		double a00 = 13.974980511346153;
		double a01 =  -14.151856090687085;
		double a02 =  -16.178626131335385;
		double a03 = 19.560601569667362;
		double a04 =  -3.9417336419475295;
		double a05 = 0.8255644855929981;
		double a06 =  -0.08893070263651753;
		double a10 = 0.7138556307240745;
		double a11 = 2.360314999608797;
		double a12 =  -5.961891526198084;
		double a13 = 1.927712403540687;
		double a14 = 1.0236898718951672;
		double a15 =  -0.06765963200868405;
		double a16 = 0.0039782524380416445;
		double a20 = 0.19390036846069775;
		double a21 = 0.7566074260753402;
		double a22 =  -1.419938180579769;
		double a23 =  -0.15740072255885534;
		double a24 = 0.5774681543789344;
		double a25 = 0.050579240096886184;
		double a26 =  -0.0012162858732341288;

		// boundary elements for P matrix for 2nd derivative
		std::vector<std::vector<double>> P2DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 2nd derivative
		std::vector<double> P2DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 2nd derivative
		std::vector<std::vector<double>> Q2DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		double t1 = -2.0 * (a1 + a2 + a3);
		// diagonal elements for Q matrix for 2nd derivative
		std::vector<double> Q2DiagInterior{
			a3, a2, a1, t1, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P2DiagInterior, P2DiagBoundary, Q2DiagInterior, Q2DiagBoundary
			};
		return diagEntries;
	}