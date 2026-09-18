// RE-EMITTED BYU_A6_2nd_r_070_Op1.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,2.4e-05,7.3e-12,8.3e-09); (1,7,3.1e-08,1.8e-12,8.7e-12); (2,7,2.0e-11,1.8e-12,1.0e-14)
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
		double a00 = 15.424224446754343;
		double a01 = 10.34147271252897;
		double a02 =  -72.93146116283589;
		double a03 = 54.29820429842589;
		double a04 =  -8.422220618091513;
		double a05 = 1.4229627490753063;
		double a06 =  -0.13318242585710288;
		double a10 = 0.7268843310201799;
		double a11 = 2.2284381834348466;
		double a12 =  -6.037012520701647;
		double a13 = 2.4522806641247263;
		double a14 = 0.6595437931549788;
		double a15 =  -0.03137639917279288;
		double a16 = 0.0012419481397081332;
		double a20 = 0.20988382421042515;
		double a21 = 0.7068033146580297;
		double a22 =  -1.2462560462588614;
		double a23 =  -0.40040580238202844;
		double a24 = 0.6609935026733978;
		double a25 = 0.07093506828203608;
		double a26 =  -0.0019538611829988707;

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