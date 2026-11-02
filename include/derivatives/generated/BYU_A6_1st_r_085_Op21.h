// RE-EMITTED BYU_A6_1st_r_085_Op21.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,2.0e-04,9.1e-13,1.0e-07); (1,6,4.4e-09,1.8e-12,3.4e-12); (2,6,9.5e-11,4.5e-13,9.7e-14)
// yHat0 = 2.37
// yHat1 = 0.2
// yHat2 = 4.41


MatrixDiagonalEntries* createBYU_A6_1ST_R085_OP21_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5636298159810625;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07971824466577156;
		double a1 = 0.6641054102221357;
		double a2 = 0.23297808801119738;
		double a3 = 0.004428824800767828;
		double gamma01 = 13.321099554835957;
		double gamma02 = 17.843386501660643;
		double gamma10 =  - 0.358154555727302;
		double gamma12 = 12.499696290881529;
		double gamma13 = 8.628864243686683;
		double gamma20 = 0.013536681423743759;
		double gamma21 = 0.25970978533202643;
		double gamma23 = 1.1020770961543107;
		double gamma24 = 0.294115552327983;
		double a00 =  -4.075403709083972;
		double a01 =  -18.23276569603707;
		double a02 = 15.394106761121185;
		double a03 = 8.256016077487594;
		double a04 =  -1.5707769551336903;
		double a05 = 0.24884331151242967;
		double a06 =  -0.020019789866478798;
		double a10 = 0.9836541338331628;
		double a11 =  -7.1378095474967544;
		double a12 =  -8.576978517824474;
		double a13 = 12.611898016326691;
		double a14 = 2.398212954634964;
		double a15 =  -0.30748893130822774;
		double a16 = 0.028511891834636247;
		double a20 =  -0.05658252610728207;
		double a21 =  -0.5259779785208892;
		double a22 =  -0.7150840266330869;
		double a23 = 0.5985741641676037;
		double a24 = 0.6637874934113728;
		double a25 = 0.03698456121686581;
		double a26 =  -0.0017016875345839978;

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