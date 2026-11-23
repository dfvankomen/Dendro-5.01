// RE-EMITTED BYU_A6_2nd_r_080_Op2.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,9.4e-06,2.3e-11,8.0e-09); (1,7,5.2e-08,1.8e-12,3.5e-11); (2,7,7.1e-11,1.8e-12,2.3e-14)
MatrixDiagonalEntries* createBYU_A6_2ND_R080_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.45380344635841907;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.04071702966194011;
		double a1 = 0.35679115670773653;
		double a2 = 0.38234902673152094;
		double a3 = 0.011428187600766462;
		double gamma01 = 16.299502091289924;
		double gamma02 = 26.64726150052526;
		double gamma10 = 0.03798448155277675;
		double gamma12 = 3.1069776621627194;
		double gamma13 = 0.9451852957324776;
		double gamma20 = 0.010170030656434854;
		double gamma21 = 0.2779836889816956;
		double gamma23 = 0.7064481194727863;
		double gamma24 = 0.11180164195610012;
		double a00 = 14.992318816763525;
		double a01 = 3.041937859079051;
		double a02 =  -56.017904796590464;
		double a03 = 43.94565712721056;
		double a04 =  -7.086939852454896;
		double a05 = 1.2449253136728058;
		double a06 =  -0.11999446768058344;
		double a10 = 0.7185725555796796;
		double a11 = 2.3161305986917102;
		double a12 =  -6.137457182690626;
		double a13 = 2.4182976179701767;
		double a14 = 0.7201033610693921;
		double a15 =  -0.03735917222336643;
		double a16 = 0.0017122216030338215;
		double a20 = 0.19432405447986395;
		double a21 = 0.7492674598583171;
		double a22 =  -1.3606811799844827;
		double a23 =  -0.24894817813188647;
		double a24 = 0.6097996795884435;
		double a25 = 0.05768826898593432;
		double a26 =  -0.0014501047961896396;

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