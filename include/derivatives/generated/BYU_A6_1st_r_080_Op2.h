// yHat0 = 1.68
// yHat1 = 3.83
// yHat2 = 2.13


MatrixDiagonalEntries* createBYU_A6_1ST_R080_OP2_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5551807442234595;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07520673263376323;
		double a1 = 0.670920507629016;
		double a2 = 0.22381002064073385;
		double a3 = 0.003948975982246324;
		double gamma01 = 9.174456640036146;
		double gamma02 = 9.280297990878903;
		double gamma10 = 0.09651333407858931;
		double gamma12 = 0.988465089145227;
		double gamma13 =  - 0.2736075074764507;
		double gamma20 =  - 0.03350629264987801;
		double gamma21 =  - 0.21685153967378007;
		double gamma23 = 2.2188719302061104;
		double gamma24 = 0.6336651228741246;
		double a00 =  - 3.6697328404075233;
		double a01 =  - 9.486005217083617;
		double a02 = 10.02263443827542;
		double a03 = 3.7496362525874405;
		double a04 =  - 0.7447684601611718;
		double a05 = 0.14375890529907068;
		double a06 =  - 0.015523078496983485;
		double a10 =  - 0.36561537373103253;
		double a11 =  - 1.1406804906254677;
		double a12 = 1.4047509897637742;
		double a13 = 0.29470901263563326;
		double a14 =  - 0.22802984462970174;
		double a15 = 0.03865247222380275;
		double a16 =  - 0.0037867656221984764;
		double a20 = 0.12514556014889963;
		double a21 =  - 0.07440284017002391;
		double a22 =  - 2.221486373860836;
		double a23 = 0.6264904512907628;
		double a24 = 1.478730917040611;
		double a25 = 0.06797392669053914;
		double a26 =  - 0.0024516411398335736;

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