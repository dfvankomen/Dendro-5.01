// yHat0 = 0.69
// yHat1 = 1.49
// yHat2 = 0.96


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP8_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 9.38983186534384;
		double gamma02 = 10.473862052331517;
		double gamma10 =  - 0.028762721311611235;
		double gamma12 = 4.9303907339875;
		double gamma13 = 2.9542517589178594;
		double gamma20 = 0.057006648773377196;
		double gamma21 = 0.6592882586667739;
		double gamma23 = 0.20982787968748548;
		double gamma24 =  - 0.013628298190755325;
		double a00 =  - 3.6658432424748835;
		double a01 =  - 10.239829048154938;
		double a02 = 9.864826799485416;
		double a03 = 4.9820962941646725;
		double a04 =  - 1.162071138443952;
		double a05 = 0.2490569739806818;
		double a06 =  - 0.02823663869476687;
		double a10 = 0.01891082904392098;
		double a11 =  - 2.98492819079063;
		double a12 =  - 2.376029671019827;
		double a13 = 4.715436170325902;
		double a14 = 0.6916869903360471;
		double a15 =  - 0.07026759819714562;
		double a16 = 0.0051914706310046445;
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
