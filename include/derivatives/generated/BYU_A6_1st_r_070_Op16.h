// yHat0 = 1.77
// yHat1 = 0.02
// yHat2 = 1.53


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP16_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5406172854667417;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06788016563593934;
		double a1 = 0.6819178310623292;
		double a2 = 0.20847205673939975;
		double a3 = 0.003211835520517426;
		double gamma01 = 9.767497047698585;
		double gamma02 = 10.547246182324539;
		double gamma10 =  - 0.020008695621644475;
		double gamma12 = 4.740462698999408;
		double gamma13 = 2.9752891015868834;
		double gamma20 = 0.016010598975206045;
		double gamma21 = 0.3172470922581119;
		double gamma23 = 0.7716636943573475;
		double gamma24 = 0.13328614056475505;
		double a00 =  - 3.7263413015635836;
		double a01 =  - 10.753853014448275;
		double a02 = 10.766182346927023;
		double a03 = 4.4504998348161084;
		double a04 =  - 0.8840422196768489;
		double a05 = 0.16442523026346628;
		double a06 =  - 0.016870868247930278;
		double a10 =  - 0.009218090786794883;
		double a11 =  - 2.8532772214246473;
		double a12 =  - 2.346671516777103;
		double a13 = 4.520558961187656;
		double a14 = 0.7696014186049054;
		double a15 =  - 0.08824210678407977;
		double a16 = 0.007248555980063632;
		double a20 =  - 0.06940677542881246;
		double a21 =  - 0.6130921058016825;
		double a22 =  - 0.4223997954877698;
		double a23 = 0.7336106519848824;
		double a24 = 0.36083085015551136;
		double a25 = 0.010799181111351721;
		double a26 =  - 0.0003420065334666392;

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