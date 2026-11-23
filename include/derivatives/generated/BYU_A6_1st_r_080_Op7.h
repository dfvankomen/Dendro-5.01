// RE-EMITTED BYU_A6_1st_r_080_Op7.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,6,7.9e-09,6.8e-13,1.3e-10); (1,6,2.7e-10,1.1e-13,1.6e-13); (2,6,1.3e-10,1.1e-13,1.8e-13)
// yHat0 = 0.72
// yHat1 = 1.04
// yHat2 = 4.41


MatrixDiagonalEntries* createBYU_A6_1ST_R080_OP7_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.5551807442234595;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.07520673263376323;
		double a1 = 0.670920507629016;
		double a2 = 0.22381002064073385;
		double a3 = 0.003948975982246324;
		double gamma01 = 8.657093219054511;
		double gamma02 = 9.161666760642337;
		double gamma10 = 0.08742916741693876;
		double gamma12 = 1.5437511141513265;
		double gamma13 = 0.3898642493611516;
		double gamma20 = 0.013553130952545497;
		double gamma21 = 0.2630022074719273;
		double gamma23 = 1.079077063763526;
		double gamma24 = 0.2808148214643671;
		double a00 =  -3.5874599778210072;
		double a01 =  -8.774603002043557;
		double a02 = 8.798427437261582;
		double a03 = 4.453733649098931;
		double a04 =  -1.116589031109076;
		double a05 = 0.25728226332201715;
		double a06 =  -0.03079133870888858;
		double a10 =  -0.33590749385580826;
		double a11 =  -1.3177791370880587;
		double a12 = 0.6513615740971551;
		double a13 = 0.9745292683146937;
		double a14 = 0.025996585465013447;
		double a15 = 0.002268845382997301;
		double a16 =  -0.0004696423159927078;
		double a20 =  -0.05700990945071034;
		double a21 =  -0.5317811305044204;
		double a22 =  -0.6963766838880673;
		double a23 = 0.610930765277935;
		double a24 = 0.6414607088314191;
		double a25 = 0.034310907629624035;
		double a26 =  -0.0015346578957801417;

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