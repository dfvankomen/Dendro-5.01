// yHat0 = 3.8
// yHat1 = 1.35
// yHat2 = 0.66


MatrixDiagonalEntries* create2A6_r070_Op6_Diagonals() {
		double alpha0 = 0.4309329692460775;
		double alpha = alpha0;

		double beta = 0.035593529920944555;
		double a1 = 0.42362022943349215;
		double a2 = 0.35644150411900966;
		double a3 = 0.009296305824945953;
		double gamma01 = 18.294275138113505;
		double gamma02 = 37.61851326233417;
		double gamma10 = 0.04039232520947982;
		double gamma12 = 2.955283511737379;
		double gamma13 = 0.7402157798056678;
		double gamma20 = 0.01147306133299495;
		double gamma21 = 0.289547022390585;
		double gamma23 = 0.7672160031567729;
		double gamma24 = 0.13029424083046517;
		double a00 = 15.718194556596105;
		double a01 = 15.309792077984973;
		double a02 =  - 84.44342025998756;
		double a03 = 61.344510816082504;
		double a04 =  - 9.331059525305184;
		double a05 = 1.544141278939731;
		double a06 =  - 0.14215861261547869;
		double a10 = 0.7381128554218048;
		double a11 = 2.112833622602912;
		double a12 =  - 6.020529011914938;
		double a13 = 2.7397664342706296;
		double a14 = 0.43973732831981377;
		double a15 =  - 0.009503157223304831;
		double a16 =  - 0.0004180714613059667;
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