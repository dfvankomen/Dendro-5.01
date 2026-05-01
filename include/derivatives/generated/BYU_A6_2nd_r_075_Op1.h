MatrixDiagonalEntries* createBYU_A6_2ND_R075_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.44166857544275956;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.037953543243916414;
		double a1 = 0.3923850755220408;
		double a2 = 0.36861622755813694;
		double a3 = 0.01026602795764041;
		double gamma01 = 13.767455481048394;
		double gamma02 = 12.721005146238568;
		double gamma10 = 0.03917284605874037;
		double gamma12 = 3.0321106982987556;
		double gamma13 = 0.8891017420275389;
		double gamma20 = 0.011862934165547963;
		double gamma21 = 0.2926245321701123;
		double gamma23 = 0.7950647826715415;
		double gamma24 = 0.13792992549391464;
		double a00 = 14.070935188566466;
		double a01 =  - 12.530148789731015;
		double a02 =  - 19.93624060814391;
		double a03 = 21.860583916118966;
		double a04 =  - 4.23838741414119;
		double a05 = 0.8651183219561298;
		double a06 =  - 0.09186061644327752;
		double a10 = 0.7287173078054032;
		double a11 = 2.2090341017921675;
		double a12 =  - 6.012133269122706;
		double a13 = 2.4542457116098197;
		double a14 = 0.6493471311987418;
		double a15 =  - 0.030372676574248143;
		double a16 = 0.0011616932816404324;
		double a20 = 0.21437918258922814;
		double a21 = 0.6928198651468838;
		double a22 =  - 1.1968025951002588;
		double a23 =  - 0.47069963250723973;
		double a24 = 0.6865102523170461;
		double a25 = 0.07591435201882522;
		double a26 =  - 0.0021214244644953894;

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