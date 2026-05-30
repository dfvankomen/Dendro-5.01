// yHat0 = 3.6
// yHat1 = 1.35
// yHat2 = 0.54


MatrixDiagonalEntries* create2A6_r060_Op5_Diagonals() {
		double alpha0 = 0.41306375445879423;
		double alpha = alpha0;

		double beta = 0.031838395471255235;
		double a1 = 0.47509136605394736;
		double a2 = 0.3361249867091765;
		double a3 = 0.007801442996605072;
		double gamma01 = 13.598527057543068;
		double gamma02 = 11.79189436020419;
		double gamma10 = 0.03747777678635672;
		double gamma12 = 3.1389000623117687;
		double gamma13 = 0.7796901349548174;
		double gamma20 = 0.010799389521393124;
		double gamma21 = 0.2809467157114571;
		double gamma23 = 0.7875385004242951;
		double gamma24 = 0.13515460343474503;
		double a00 = 14.009463806497045;
		double a01 =  - 13.56906317837669;
		double a02 =  - 17.528997289451183;
		double a03 = 20.387145416740612;
		double a04 =  - 4.048341812990952;
		double a05 = 0.8397789202597449;
		double a06 =  - 0.08998363138777513;
		double a10 = 0.7121424122329694;
		double a11 = 2.390206575882956;
		double a12 =  - 6.475006638483693;
		double a13 = 2.918581890558989;
		double a14 = 0.46616019131029063;
		double a15 =  - 0.011926852389663765;
		double a16 =  - 0.00015757880815398046;
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