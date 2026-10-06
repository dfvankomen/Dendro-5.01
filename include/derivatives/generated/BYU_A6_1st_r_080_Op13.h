// yHat0 = 1.68
// yHat1 = 2.45
// yHat2 = 4.41


MatrixDiagonalEntries* createBYU_A6_1ST_R080_OP13_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5551807442234595;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07520673263376323;
		double a1 = 0.670920507629016;
		double a2 = 0.22381002064073385;
		double a3 = 0.003948975982246324;
		double gamma01 = 9.174456640036146;
		double gamma02 = 9.280297990878903;
		double gamma10 = 0.0861929342247668;
		double gamma12 = 1.4034902721703322;
		double gamma13 = 0.19760981966353192;
		double gamma20 = 0.013553130952545497;
		double gamma21 = 0.2630022074719273;
		double gamma23 = 1.079077063763526;
		double gamma24 = 0.2808148214643671;
		double a00 =  - 3.6697328404075233;
		double a01 =  - 9.486005217083617;
		double a02 = 10.02263443827542;
		double a03 = 3.7496362525874405;
		double a04 =  - 0.7447684601611718;
		double a05 = 0.14375890529907068;
		double a06 =  - 0.015523078496983485;
		double a10 =  - 0.334349843437773;
		double a11 =  - 1.2979303639128332;
		double a12 = 0.8866429697150052;
		double a13 = 0.7792732577906494;
		double a14 =  - 0.04342794134535127;
		double a15 = 0.01092208441990981;
		double a16 =  - 0.0011301632470071428;
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