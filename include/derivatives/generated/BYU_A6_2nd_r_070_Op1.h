MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP1_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.4309329692460775;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.035593529920944555;
		double a1 = 0.42362022943349215;
		double a2 = 0.35644150411900966;
		double a3 = 0.009296305824945953;
		double gamma01 = 17.486418327128295;
		double gamma02 = 33.17530079966559;
		double gamma10 = 0.03896026476432345;
		double gamma12 = 3.045503319835981;
		double gamma13 = 0.8974938763696754;
		double gamma20 = 0.01147306133299495;
		double gamma21 = 0.289547022390585;
		double gamma23 = 0.7672160031567729;
		double gamma24 = 0.13029424083046517;
		double a00 = 15.42422444670826;
		double a01 = 10.341472712635337;
		double a02 =  - 72.93146115451091;
		double a03 = 54.2982043044822;
		double a04 =  - 8.422220622133137;
		double a05 = 1.4229627495127242;
		double a06 =  - 0.13318242587996487;
		double a10 = 0.7268843310217808;
		double a11 = 2.2284381834435365;
		double a12 =  - 6.037012520694956;
		double a13 = 2.4522806641188897;
		double a14 = 0.659543793152555;
		double a15 =  - 0.03137639917265178;
		double a16 = 0.0012419481397426635;
		double a20 = 0.2098838242104275;
		double a21 = 0.7068033146580226;
		double a22 =  - 1.246256046258851;
		double a23 =  - 0.4004058023820333;
		double a24 = 0.6609935026733978;
		double a25 = 0.07093506828203591;
		double a26 =  - 0.0019538611829988616;

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