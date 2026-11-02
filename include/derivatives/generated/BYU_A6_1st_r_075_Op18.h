// RE-EMITTED BYU_A6_1st_r_075_Op18.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,5.7e-05,9.1e-13,9.0e-08); (1,6,8.4e-10,4.5e-13,2.7e-12); (2,6,2.4e-09,2.8e-14,1.0e-12)
// yHat0 = 2.4
// yHat1 = 2.9
// yHat2 = 2.1


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP18_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 12.388586845434393;
		double gamma02 = 15.851513640248598;
		double gamma10 = 0.046559377147835514;
		double gamma12 = 2.9519888021290015;
		double gamma13 = 1.5441477917945718;
		double gamma20 =  - 0.08895892440675712;
		double gamma21 =  - 0.8120533547060657;
		double gamma23 = 3.80890324020952;
		double gamma24 = 1.1718880835653283;
		double a00 =  -3.986380686230779;
		double a01 =  -16.239291907740242;
		double a02 = 14.224750823440967;
		double a03 = 7.154373444607476;
		double a04 =  -1.3519344489289715;
		double a05 = 0.21638844067454818;
		double a06 =  -0.017905665822996873;
		double a10 =  -0.2080733104711398;
		double a11 =  -1.953150422528735;
		double a12 =  -0.7292996403632793;
		double a13 = 2.5797142504909054;
		double a14 = 0.3408521118103784;
		double a15 =  -0.03215240924124965;
		double a16 = 0.002109420303120257;
		double a20 = 0.34267467130349605;
		double a21 = 0.5234653336549591;
		double a22 =  -4.217008175422295;
		double a23 = 0.5311786503779576;
		double a24 = 2.6971636498405327;
		double a25 = 0.12701571011644447;
		double a26 =  -0.00448983987109494;

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