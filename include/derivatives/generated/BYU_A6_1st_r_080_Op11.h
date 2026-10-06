// yHat0 = 3.84
// yHat1 = 1.52
// yHat2 = 4.41


MatrixDiagonalEntries* createBYU_A6_1ST_R080_OP11_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5551807442234595;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07520673263376323;
		double a1 = 0.670920507629016;
		double a2 = 0.22381002064073385;
		double a3 = 0.003948975982246324;
		double gamma01 = 10.80249361398501;
		double gamma02 = 13.26779395231656;
		double gamma10 =  - 0.04803992715894287;
		double gamma12 = 5.383257678095695;
		double gamma13 = 3.2633081488190916;
		double gamma20 = 0.013553130952545497;
		double gamma21 = 0.2630022074719273;
		double gamma23 = 1.079077063763526;
		double gamma24 = 0.2808148214643671;
		double a00 =  - 3.8081558041479995;
		double a01 =  - 13.170317715502168;
		double a02 = 11.76668756013641;
		double a03 = 6.352902574615416;
		double a04 =  - 1.381818958613077;
		double a05 = 0.2684157893250343;
		double a06 =  - 0.027713445308706407;
		double a10 = 0.07608460832391346;
		double a11 =  - 3.235379745215;
		double a12 =  - 2.727415303896414;
		double a13 = 5.19074405648853;
		double a14 = 0.7693353329035763;
		double a15 =  - 0.0793764444810418;
		double a16 = 0.006007495703628292;
		double a20 =  - 0.05700990945071849;
		double a21 =  - 0.5317811305043647;
		double a22 =  - 0.6963766838880415;
		double a23 = 0.6109307652777597;
		double a24 = 0.6414607088315284;
		double a25 = 0.03431090762960477;
		double a26 =  - 0.0015346578957777532;

		// boundary elements for P matrix for 1st derivative
		std::vector<std::vector<double>> P1DiagBoundary{
			{1.0, gamma01, gamma02},
			{gamma10, 1.0, gamma12, gamma13},
			{gamma20, gamma21, 1.0, gamma23, gamma24}
		};

		// diagonal elements for P matrix for 1st derivative
		std::vector<double> P1DiagInterior{
			beta, alpha, 1.0, alpha, beta
		};

		// boundary elements for Q matrix for 1st derivative
		std::vector<std::vector<double>> Q1DiagBoundary{
			{a00, a01, a02, a03, a04, a05, a06},
			{a10, a11, a12, a13, a14, a15, a16},
			{a20, a21, a22, a23, a24, a25, a26}
		};

		// diagonal elements for Q matrix for 1st derivative
		std::vector<double> Q1DiagInterior{
			-a3, -a2, -a1, 0.0, a1, a2, a3
		};

		// store the entries for matrix creation
		MatrixDiagonalEntries* diagEntries = new
			MatrixDiagonalEntries{
				P1DiagInterior, P1DiagBoundary, Q1DiagInterior, Q1DiagBoundary
			};
		return diagEntries;
	}