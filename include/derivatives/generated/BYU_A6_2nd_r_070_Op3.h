// RE-EMITTED BYU_A6_2nd_r_070_Op3.h: closure Q rows re-projected onto exact Taylor consistency (gamma unchanged unless a row needed a joint gamma+a correction).
// per row: (row, order, max residual before, after, max |delta a|): (0,7,1.5e-03,1.6e-11,4.7e-07); (1,7,9.5e-08,1.8e-12,3.3e-11); (2,7,5.1e-10,1.1e-13,7.3e-13)
MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP3_Diagonals(
	const std::vector<double>& D_coeffs) {
		double alpha0 = 0.4309329692460775;
		double alpha = alpha0 + D_coeffs[0];

		double beta = 0.035593529920944555;
		double a1 = 0.42362022943349215;
		double a2 = 0.35644150411900966;
		double a3 = 0.009296305824945953;
		double gamma01 = 13.503763237286865;
		double gamma02 = 11.270697805076875;
		double gamma10 = 0.04039232520947982;
		double gamma12 = 2.955283511737379;
		double gamma13 = 0.7402157798056678;
		double gamma20 = 0.008556874749086792;
		double gamma21 = 0.24037616932520273;
		double gamma23 = 1.0234691356920145;
		double gamma24 = 0.21701417790412234;
		double a00 = 13.974980511346153;
		double a01 =  -14.151856090687085;
		double a02 =  -16.178626131335385;
		double a03 = 19.560601569667362;
		double a04 =  -3.9417336419475295;
		double a05 = 0.8255644855929981;
		double a06 =  -0.08893070263651753;
		double a10 = 0.7381128554325225;
		double a11 = 2.1128336225696227;
		double a12 =  -6.020529011941879;
		double a13 = 2.7397664342989727;
		double a14 = 0.4397373283254114;
		double a15 =  -0.009503157223087774;
		double a16 =  -0.0004180714615626034;
		double a20 = 0.16311491178336965;
		double a21 = 0.753481858869501;
		double a22 =  -0.8702894351031085;
		double a23 =  -1.047717583615543;
		double a24 = 0.8727634536901621;
		double a25 = 0.13268481883101735;
		double a26 =  -0.004038024455398637;

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