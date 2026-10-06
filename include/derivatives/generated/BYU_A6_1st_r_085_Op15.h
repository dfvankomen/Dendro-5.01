// yHat0 = 3.81
// yHat1 = 1.55
// yHat2 = 3.33


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP15_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 10.759193339373669;
		double gamma02 = 13.283470426784763;
		double gamma10 =  - 0.05447113958159977;
		double gamma12 = 5.452379009558196;
		double gamma13 = 3.2784359989219225;
		double gamma20 = 0.008951758212656707;
		double gamma21 = 0.21441473922607224;
		double gamma23 = 1.2130024721722463;
		double gamma24 = 0.322064175847782;
		double a00 =  - 3.8004165422509897;
		double a01 =  - 13.121019623413277;
		double a02 = 11.649292264421266;
		double a03 = 6.445971671909907;
		double a04 =  - 1.4257407630570507;
		double a05 = 0.28133105538866665;
		double a06 =  - 0.029418062475013333;
		double a10 = 0.09389299230696006;
		double a11 =  - 3.299346374806078;
		double a12 =  - 2.730847874451314;
		double a13 = 5.240031082019511;
		double a14 = 0.7702376012017559;
		double a15 =  - 0.08014689941629337;
		double a16 = 0.006179473089061288;
		double a20 =  - 0.03918323576408785;
		double a21 =  - 0.48244655201798015;
		double a22 =  - 0.8631544380681885;
		double a23 = 0.6062349215769293;
		double a24 = 0.7427324794313306;
		double a25 = 0.03734705789528659;
		double a26 =  - 0.0015302330532956756;

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