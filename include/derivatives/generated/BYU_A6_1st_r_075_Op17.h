// RE-EMITTED BYU_A6_1st_r_075_Op17.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,1.2e-05,9.1e-13,1.5e-08); (1,6,5.0e-08,2.3e-13,4.2e-10); (2,6,2.4e-09,2.8e-14,1.0e-12)
// yHat0 = 3.9
// yHat1 = 1.49
// yHat2 = 2.1


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP17_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 10.701581468084028;
		double gamma02 = 12.890848958458404;
		double gamma10 =  - 0.028762721311611235;
		double gamma12 = 4.9303907339875;
		double gamma13 = 2.9542517589178594;
		double gamma20 =  - 0.08895892440675712;
		double gamma21 =  - 0.8120533547060657;
		double gamma23 = 3.80890324020952;
		double gamma24 = 1.1718880835653283;
		double a00 =  -3.803901946065391;
		double a01 =  -12.890035800757865;
		double a02 = 11.734291777775999;
		double a03 = 6.01849616447116;
		double a04 =  -1.2774399224925126;
		double a05 = 0.24338449410678034;
		double a06 =  -0.024794767038172485;
		double a10 = 0.018910829031066535;
		double a11 =  -2.984928190960322;
		double a12 =  -2.376029670844019;
		double a13 = 4.715436169905925;
		double a14 = 0.6916869904465199;
		double a15 =  -0.07026759821327902;
		double a16 = 0.0051914706341078575;
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