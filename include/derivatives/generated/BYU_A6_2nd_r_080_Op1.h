// RE-EMITTED BYU_A6_2nd_r_080_Op1.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,9.7e-06,2.3e-11,8.6e-09); (1,7,3.7e-08,1.8e-12,1.8e-11); (2,7,2.7e-11,1.1e-13,7.2e-15)
MatrixDiagonalEntries* createBYU_A6_2ND_R080_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.45380344635841907;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.04071702966194011;
		double a1 = 0.35679115670773653;
		double a2 = 0.38234902673152094;
		double a3 = 0.011428187600766462;
		double gamma01 = 14.20182115333874;
		double gamma02 = 15.110016341750255;
		double gamma10 = 0.04125627493568223;
		double gamma12 = 2.9008546790563345;
		double gamma13 = 0.9211656908051901;
		double gamma20 = 0.012204856906641892;
		double gamma21 = 0.29723221557261104;
		double gamma23 = 0.7793236352970117;
		double gamma24 = 0.13412879946906758;
		double a00 = 14.22899603101491;
		double a01 =  -9.858799909385972;
		double a02 =  -26.125951430667165;
		double a03 = 25.649217835014998;
		double a04 =  -4.72704879725299;
		double a05 = 0.9302731729795553;
		double a06 =  -0.09668690170333656;
		double a10 = 0.7479517544533543;
		double a11 = 2.0017152226373813;
		double a12 =  -5.59683300639002;
		double a13 = 2.1623215035764995;
		double a14 = 0.720881629233533;
		double a15 =  -0.037682383543540324;
		double a16 = 0.0016452800327926614;
		double a20 = 0.21921145672873868;
		double a21 = 0.6857220485143117;
		double a22 =  -1.2172573879739883;
		double a23 =  -0.42969085093519455;
		double a24 = 0.6701140518206475;
		double a25 = 0.0739950181162509;
		double a26 =  -0.0020943362707660116;

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