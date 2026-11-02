// RE-EMITTED BYU_A6_1st_r_070_Op19.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,2.3e-04,9.1e-13,1.1e-07); (1,6,9.1e-08,4.5e-13,2.4e-11); (2,6,3.0e-11,4.5e-13,3.8e-14)
// yHat0 = 2.4
// yHat1 = 4.22
// yHat2 = 1.53


MatrixDiagonalEntries* createBYU_A6_1ST_R070_OP19_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5406172854667417;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06788016563593934;
		double a1 = 0.6819178310623292;
		double a2 = 0.20847205673939975;
		double a3 = 0.003211835520517426;
		double gamma01 = 12.598094593642406;
		double gamma02 = 16.23386410161242;
		double gamma10 = 0.055108229889549915;
		double gamma12 = 2.6564377716823895;
		double gamma13 = 1.3331110367019765;
		double gamma20 = 0.016010598975206045;
		double gamma21 = 0.3172470922581119;
		double gamma23 = 0.7716636943573475;
		double gamma24 = 0.13328614056475505;
		double a00 =  -4.008553628886654;
		double a01 =  -16.66110036915272;
		double a02 = 14.525482424832104;
		double a03 = 7.314994479412548;
		double a04 =  -1.3685198894375377;
		double a05 = 0.21499156513772097;
		double a06 =  -0.01729458190546011;
		double a10 =  -0.23535242145168392;
		double a11 =  -1.8152924071636931;
		double a12 =  -0.4627337018461673;
		double a13 = 2.242638561506852;
		double a14 = 0.2982918629328088;
		double a15 =  -0.029645076746851306;
		double a16 = 0.002093182768734797;
		double a20 =  -0.06940677542881667;
		double a21 =  -0.6130921058017058;
		double a22 =  -0.4223997954877321;
		double a23 = 0.7336106519848471;
		double a24 = 0.36083085015552163;
		double a25 = 0.010799181111352514;
		double a26 =  -0.00034200653346665524;

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