// RE-EMITTED BYU_A6_1st_r_075_Op5.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,1.2e-05,9.1e-13,1.5e-08); (1,6,7.4e-08,4.4e-17,6.3e-11); (2,6,1.8e-12,1.1e-13,1.0e-15)
// yHat0 = 3.9
// yHat1 = 4.16
// yHat2 = 0.42


MatrixDiagonalEntries* createBYU_A6_1ST_R075_OP5_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5475276528976143;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.0712887129426;
		double a1 = 0.6768127828371342;
		double a2 = 0.21567975158608105;
		double a3 = 0.0035480266103059613;
		double gamma01 = 10.701581468084028;
		double gamma02 = 12.890848958458404;
		double gamma10 = 0.05474789128172447;
		double gamma12 = 2.6849462509375877;
		double gamma13 = 1.3719245955864618;
		double gamma20 = 0.01926386345301328;
		double gamma21 = 0.33570565454932;
		double gamma23 = 0.7884098090152796;
		double gamma24 = 0.14936803315928954;
		double a00 =  -3.803901946065391;
		double a01 =  -12.890035800757865;
		double a02 = 11.734291777775999;
		double a03 = 6.01849616447116;
		double a04 =  -1.2774399224925126;
		double a05 = 0.24338449410678034;
		double a06 =  -0.024794767038172485;
		double a10 =  -0.2341662018687464;
		double a11 =  -1.8230357966800523;
		double a12 =  -0.5057712776830394;
		double a13 = 2.278247609794947;
		double a14 = 0.3144990622479191;
		double a15 =  -0.032098386341554895;
		double a16 = 0.002324990530527158;
		double a20 =  -0.08046477081570237;
		double a21 =  -0.6168933423558277;
		double a22 =  -0.4051715130394478;
		double a23 = 0.7030922878921692;
		double a24 = 0.38595393361334546;
		double a25 = 0.014009297751043146;
		double a26 =  -0.0005258930455798714;

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