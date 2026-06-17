// yHat0 = 1.74
// yHat1 = -0.28
// yHat2 = 0.96


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP6_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 9.440989783371029;
		double gamma02 = 9.855215182150074;
		double gamma10 = 0.09088883019632422;
		double gamma12 = 1.4518980591716348;
		double gamma13 = 0.2988152093035981;
		double gamma20 = 0.057006648773377196;
		double gamma21 = 0.6592882586667739;
		double gamma23 = 0.20982787968748548;
		double gamma24 =  - 0.013628298190755325;
		double a00 =  - 3.6949911246980953;
		double a01 =  - 10.05802296180195;
		double a02 = 10.353598938854793;
		double a03 = 4.071970600580927;
		double a04 =  - 0.8101161037091037;
		double a05 = 0.15378124508049626;
		double a06 =  - 0.016220593576525524;
		double a10 =  - 0.3459279521636606;
		double a11 =  - 1.2739372944285994;
		double a12 = 0.7472818320330895;
		double a13 = 0.8751229468710161;
		double a14 =  - 0.009337402511013192;
		double a15 = 0.007830722729604533;
		double a16 =  - 0.001032852530521786;
		double a20 =  - 0.2199386022372042;
		double a21 =  - 0.8707554176034611;
		double a22 = 0.4731523886722719;
		double a23 = 0.6327349582988459;
		double a24 =  - 0.014946981523476755;
		double a25 =  - 6.254034734862424e-6;
		double a26 =  - 0.00024009157218674905;

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
