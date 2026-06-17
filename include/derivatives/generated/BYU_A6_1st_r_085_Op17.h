// yHat0 = 1.62
// yHat1 = 3.8
// yHat2 = 3.33


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP17_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 8.952554944545723;
		double gamma02 = 8.795814814853614;
		double gamma10 = 0.0936820088925504;
		double gamma12 = 1.1528707549468167;
		double gamma13 =  - 0.10737986409723717;
		double gamma20 = 0.008951758212656707;
		double gamma21 = 0.21441473922607224;
		double gamma23 = 1.2130024721722463;
		double gamma24 = 0.322064175847782;
		double a00 =  - 3.6488986635805327;
		double a01 =  - 9.007438105441317;
		double a02 = 9.750495385801761;
		double a03 = 3.4734948472555742;
		double a04 =  - 0.6874449537666463;
		double a05 = 0.13463657268969775;
		double a06 =  - 0.014845082125668214;
		double a10 =  - 0.355968898884131;
		double a11 =  - 1.198496561596093;
		double a12 = 1.20541189097167;
		double a13 = 0.49504106596727154;
		double a14 =  - 0.17494447559135912;
		double a15 = 0.03224149095463541;
		double a16 =  - 0.0032845118006592706;
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