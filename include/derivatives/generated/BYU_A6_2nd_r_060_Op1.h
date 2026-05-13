MatrixDiagonalEntries* createBYU_A6_2ND_R060_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.41306375445879423;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.031838395471255235;
		double a1 = 0.47509136605394736;
		double a2 = 0.3361249867091765;
		double a3 = 0.007801442996605072;
		double gamma01 = 20.67326570249215;
		double gamma02 = 50.70296136368158;
		double gamma10 = 0.03758177932227239;
		double gamma12 = 3.132347902685562;
		double gamma13 = 0.8905750711871143;
		double gamma20 = 0.010799389521393124;
		double gamma21 = 0.2809467157114571;
		double gamma23 = 0.7875385004242951;
		double gamma24 = 0.13515460343474503;
		double a00 = 16.583882796932674;
		double a01 = 29.940584078824934;
		double a02 =  - 118.34403625359796;
		double a03 = 82.09459530115294;
		double a04 =  - 12.007423914687068;
		double a05 = 1.9009898550847855;
		double a06 =  - 0.16859184114008968;
		double a10 = 0.7143168456432624;
		double a11 = 2.3634647891664717;
		double a12 =  - 6.290348787628061;
		double a13 = 2.606513092790691;
		double a14 = 0.6336576241561488;
		double a15 =  - 0.028684395865018;
		double a16 = 0.0010808317494965876;
		double a20 = 0.20057883639754837;
		double a21 = 0.7221763892041082;
		double a22 =  - 1.2228884028344746;
		double a23 =  - 0.4537715939180421;
		double a24 = 0.682637969848546;
		double a25 = 0.07323748898909674;
		double a26 =  - 0.0019706876867844293;

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