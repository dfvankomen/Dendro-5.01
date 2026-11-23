// RE-EMITTED BYU_A6_1st_r_065_Op16.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,1.4e-07,4.5e-13,1.1e-10); (1,6,3.3e-10,1.1e-13,2.2e-13); (2,6,1.6e-10,4.5e-13,1.6e-13)
// yHat0 = 0.57
// yHat1 = -0.55
// yHat2 = 3.39


MatrixDiagonalEntries* createBYU_A6_1ST_R065_OP16_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5343985491442286;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.06491197622400482;
		double a1 = 0.6863465853886775;
		double a2 = 0.2020882683083937;
		double a3 = 0.0029291344542561916;
		double gamma01 = 10.366446898794267;
		double gamma02 = 12.072586130398253;
		double gamma10 = 0.09047046541979682;
		double gamma12 = 1.3632433397211232;
		double gamma13 = 0.1922988396904128;
		double gamma20 = 0.00905748764423051;
		double gamma21 = 0.2247538179310633;
		double gamma23 = 1.133495308127387;
		double gamma24 = 0.2794312038485114;
		double a00 =  -3.775321612119103;
		double a01 =  -12.132641305611944;
		double a02 = 11.373775337586688;
		double a03 = 5.486036675873892;
		double a04 =  -1.147587316203904;
		double a05 = 0.2180664260212003;
		double a06 =  -0.02232820554682865;
		double a10 =  -0.34608284294930497;
		double a11 =  -1.2569630507494398;
		double a12 = 0.8820220980797256;
		double a13 = 0.7541275557601431;
		double a14 =  -0.043328452083656765;
		double a15 = 0.011485511179677367;
		double a16 =  -0.0012608192371446447;
		double a20 =  -0.04055088278818989;
		double a21 =  -0.5013223381068417;
		double a22 =  -0.7997858250086884;
		double a23 = 0.646551949278416;
		double a24 = 0.6664522862838603;
		double a25 = 0.0297620493439406;
		double a26 =  -0.0011072390024969012;

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