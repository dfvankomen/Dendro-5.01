MatrixDiagonalEntries* createBYU_A6_2ND_R065_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.42144193986005557;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.03357238082779708;
		double a1 = 0.4510384928314836;
		double a2 = 0.3456586135837276;
		double a3 = 0.008483966023256807;
		double gamma01 = 19.71609026203858;
		double gamma02 = 45.43849632031586;
		double gamma10 = 0.038734266127645245;
		double gamma12 = 3.0597412338419656;
		double gamma13 = 0.7633263411811111;
		double gamma20 = 0.0082710794621832;
		double gamma21 = 0.2348615097972726;
		double gamma23 = 1.0821431602451508;
		double gamma24 = 0.22933970223250955;
		double a00 = 16.235577303619387;
		double a01 = 24.05395511395269;
		double a02 =  - 104.70428618412987;
		double a03 = 73.74589824471893;
		double a04 =  - 10.930601518199973;
		double a05 = 1.7574135443846244;
		double a06 =  - 0.1579565581198954;
		double a10 = 0.7233457818520832;
		double a11 = 2.27053038101505;
		double a12 =  - 6.27809608303431;
		double a13 = 2.83971259799079;
		double a14 = 0.45575000577021374;
		double a15 =  - 0.010980070543648388;
		double a16 =  - 0.00026261327829975133;
		double a20 = 0.15841727272485;
		double a21 = 0.7533355301712154;
		double a22 =  - 0.7817983490569671;
		double a23 =  - 1.2000799007597833;
		double a24 = 0.9360308557721687;
		double a25 = 0.13819009975721508;
		double a26 =  - 0.004095508608765974;

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