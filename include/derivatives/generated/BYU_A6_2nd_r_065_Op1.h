// RE-EMITTED BYU_A6_2nd_r_065_Op1.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,1.4e-05,5.1e-12,4.5e-09); (1,7,2.6e-08,1.8e-12,3.4e-11); (2,7,1.5e-11,4.5e-13,2.3e-14)
MatrixDiagonalEntries* createBYU_A6_2ND_R065_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.42144193986005557;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.03357238082779708;
		double a1 = 0.4510384928314836;
		double a2 = 0.3456586135837276;
		double a3 = 0.008483966023256807;
		double gamma01 = 18.920308890334717;
		double gamma02 = 41.061698897191455;
		double gamma10 = 0.03892291237508814;
		double gamma12 = 3.047856520339225;
		double gamma13 = 0.909898320360794;
		double gamma20 = 0.011128970535300849;
		double gamma21 = 0.2858513156977236;
		double gamma23 = 0.7629295279259961;
		double gamma24 = 0.12894290738648717;
		double a00 = 15.946001290602494;
		double a01 = 19.15989967608428;
		double a02 =  -93.36440168822811;
		double a03 = 66.8049164328941;
		double a04 =  -10.03534750168156;
		double a05 = 1.638046333554827;
		double a06 =  -0.14911454322603415;
		double a10 = 0.7266837040290061;
		double a11 = 2.23020816904198;
		double a12 =  -6.024989213266888;
		double a13 = 2.4221818456856887;
		double a14 = 0.6777302447656752;
		double a15 =  -0.03319224290462318;
		double a16 = 0.0013774926491611035;
		double a20 = 0.20545610664376412;
		double a21 = 0.7165417151846225;
		double a22 =  -1.2576274415763518;
		double a23 =  -0.39017743971022145;
		double a24 = 0.6578909386315956;
		double a25 = 0.06981499238718253;
		double a26 =  -0.0018988715605916873;

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