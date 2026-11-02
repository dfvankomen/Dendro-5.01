// RE-EMITTED BYU_A6_2nd_r_075_Op1.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,1.6e-05,6.8e-12,2.6e-09); (1,7,1.5e-08,1.8e-12,1.9e-11); (2,7,2.4e-11,2.8e-14,2.0e-14)
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
		double a00 = 14.070935188873598;
		double a01 =  -12.530148790843775;
		double a02 =  -19.936240606231287;
		double a03 = 21.860583918747775;
		double a04 =  -4.238387416253575;
		double a05 = 0.8651183221634849;
		double a06 =  -0.0918606164562184;
		double a10 = 0.7287173078104175;
		double a11 = 2.2090341017854036;
		double a12 =  -6.012133269104058;
		double a13 = 2.4542457115989587;
		double a14 = 0.6493471312022319;
		double a15 =  -0.0303726765746272;
		double a16 = 0.001161693281673614;
		double a20 = 0.21437918258922978;
		double a21 = 0.6928198651468791;
		double a22 =  -1.1968025951002386;
		double a23 =  -0.47069963250724456;
		double a24 = 0.6865102523170441;
		double a25 = 0.0759143520188255;
		double a26 =  -0.002121424464495415;

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