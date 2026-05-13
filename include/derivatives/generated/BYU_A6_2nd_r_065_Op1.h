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
		double a00 = 15.946001292219606;
		double a01 = 19.159899679593043;
		double a02 =  - 93.36440169272076;
		double a03 = 66.80491643276137;
		double a04 =  - 10.035347499808632;
		double a05 = 1.638046333230389;
		double a06 =  - 0.1491145431909442;
		double a10 = 0.7266837040238145;
		double a11 = 2.230208169065466;
		double a12 =  - 6.0249892133007386;
		double a13 = 2.422181845704929;
		double a14 = 0.6777302447645523;
		double a15 =  - 0.03319224290546595;
		double a16 = 0.0013774926492344944;
		double a20 = 0.20545610664376326;
		double a21 = 0.7165417151846377;
		double a22 =  - 1.257627441576375;
		double a23 =  - 0.39017743971021346;
		double a24 = 0.6578909386315925;
		double a25 = 0.06981499238718286;
		double a26 =  - 0.0018988715605916984;

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