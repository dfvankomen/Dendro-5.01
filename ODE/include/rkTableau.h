/**
 * @file rkTableau.h
 * @brief Butcher tableaux for the ETS Runge-Kutta schemes (single source of
 * truth, shared by ETS and the RK convergence/stability test).
 *
 * Convention (matches the ETS evolve loop): nodes[i] = c_i, weights[i] = b_i,
 * matrix[i*nstages + j] = a_ij (row-major, strictly lower triangular). Returned
 * pointers reference function-static storage; do not free or mutate them.
 */
#pragma once

#include "dendro.h"
#include "logger.h"
#include "ts.h"

namespace ts {

/**
 * @brief Fetch the Butcher tableau for an explicit RK scheme.
 * @param[in]  type     scheme selector
 * @param[out] nstages  number of stages
 * @param[out] nodes    c_i abscissae (length nstages)
 * @param[out] weights  b_i weights   (length nstages)
 * @param[out] matrix   a_ij matrix   (row-major nstages x nstages)
 * @return 0 on success, -1 for an unknown scheme.
 */
inline int get_rk_tableau(ETSType type, unsigned int& nstages,
                          const DendroScalar*& nodes,
                          const DendroScalar*& weights,
                          const DendroScalar*& matrix) {
    if (type == ETSType::RK3) {
        nstages                           = 3;

        static const DendroScalar ETS_C[] = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
        static const DendroScalar ETS_T[] = {0.0, 1.0, 1.0 / 2.0};
        static const DendroScalar ETS_U[] = {
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 4.0, 0.0};

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK3");

    } else if (type == ETSType::RK4) {
        nstages                           = 4;

        static const DendroScalar ETS_C[] = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0,
                                             1.0 / 6.0};
        static const DendroScalar ETS_T[] = {0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
        static const DendroScalar ETS_U[] = {
            0.0, 0.0,       0.0, 0.0, 1.0 / 2.0, 0.0, 0.0, 0.0,
            0.0, 1.0 / 2.0, 0.0, 0.0, 0.0,       0.0, 1.0, 0.0};

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK4");

    } else if (type == ETSType::RK5) {
        // Butcher's 5th-order method (6 stages).
        // Original implementation had nstages=5 and a 5x5 Aij
        // matrix, which was incorrect — the method requires 6 stages.
        // The first row of Aij (all zeros) and the a21=1/4 entry were
        // missing.  Fixed to use the full 6x6 tableau.
        nstages                           = 6;

        static const DendroScalar ETS_C[] = {
            7.0 / 90.0, 0.0, 32.0 / 90.0, 12.0 / 90.0, 32.0 / 90.0, 7.0 / 90.0};

        static const DendroScalar ETS_T[] = {0.0,       1.0 / 4.0, 1.0 / 4.0,
                                             1.0 / 2.0, 3.0 / 4.0, 1.0};

        // clang-format off
        static const DendroScalar ETS_U[] = {
            0.0,        0.0,        0.0,        0.0,         0.0,        0.0,
            1.0 / 4.0,  0.0,        0.0,        0.0,         0.0,        0.0,
            1.0 / 8.0,  1.0 / 8.0,  0.0,        0.0,         0.0,        0.0,
            0.0,       -1.0 / 2.0,  1.0,        0.0,         0.0,        0.0,
            3.0 / 16.0, 0.0,        0.0,        9.0 / 16.0,  0.0,        0.0,
           -3.0 / 7.0,  2.0 / 7.0, 12.0 / 7.0,-12.0 / 7.0,  8.0 / 7.0,  0.0};
        // clang-format on

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(
            dendro::logger::Scope{"ETS"},
            "ETS Coefficients set for RK5 (Butcher, 6 stages)");

    } else if (type == ETSType::RK4_RALSTON) {
        // Ralston's fourth-order method (4 stages), minimal truncation error
        // relative to classic RK4. Ported from the `derivatives` experiment.
        nstages                           = 4;

        static const DendroScalar ETS_C[] = {
            // (263.0 + 24.0 * sqrt(5.0)) / 1812.0,
            0.17476028226269036,
            // (125.0 - 1000.0 * sqrt(5.0)) / 3828.0,
            -0.551480662878733,
            // (3426304.0 + 1661952.0 * sqrt(5.0)) / 5924787.0,
            // the +sqrt(5) sign matters: a prior -sign made sum(b) = -0.25 (not
            // 1) and killed convergence. keep it +.
            1.2055355993965235,
            // (30.0 - 4.0 * sqrt(5.0)) / 123.0,
            0.17118478121951902};
        static const DendroScalar ETS_T[] = {0.0,
                                             // 2.0 / 5.0,
                                             0.4,
                                             // (14.0 - 3.0 * sqrt(5.0)) / 16.0,
                                             0.4557372542187894, 1.0};
        // clang-format off
        static const DendroScalar ETS_U[] = {
            // stage 1
            0.0, 0.0, 0.0, 0.0,
            // stage 2
            0.4, 0.0, 0.0, 0.0,
            // stage 3
            0.2969776092477536, 0.15875964497103584, 0.0, 0.0,
            // stage 4
            0.21810038822592046, -3.050965148692931, 3.8328647604670105, 0.0};
        // clang-format on

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK4_RALSTON");

    } else if (type == ETSType::RK5_NYSTROM) {
        // Nystrom's fifth-order method (6 stages) — an alternative correction
        // to Kutta's RK5. Distinct experiment from the canonical Butcher RK5
        // above; ported from the `derivatives` branch.
        nstages                           = 6;
        static const DendroScalar ETS_C[] = {
            23.0 / 192.0, 0.0, 125.0 / 192.0, 0.0, -27.0 / 64.0, 125.0 / 192.0};
        static const DendroScalar ETS_T[] = {0.0, 1.0 / 3.0, 2.0 / 5.0,
                                             1.0, 2.0 / 3.0, 4.0 / 5.0};
        // clang-format off
        static const DendroScalar ETS_U[] = {
            // stage 1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // stage 2
            1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // stage 3
            4.0 / 25.0, 6.0 / 25.0, 0.0, 0.0, 0.0, 0.0,
            // stage 4
            1.0 / 4.0, -3.0, 15.0 / 4.0, 0.0, 0.0, 0.0,
            // stage 5
            2.0 / 27.0, 10.0 / 9.0, -50.0 / 81.0, 8.0 / 81.0, 0.0, 0.0,
            // stage 6
            2.0 / 25.0, 12.0 / 25.0, 2.0 / 15.0, 8.0 / 75.0, 0.0, 0.0};
        // clang-format on

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK5_NYSTROM");

    } else if (type == ETSType::RK45_CASH_KARP) {
        // Cash-Karp embedded 4(5) pair (6 stages); the 5th-order weights are
        // used here. Enables adaptive step sizing. From the `derivatives`
        // experiment.
        nstages                           = 6;
        static const DendroScalar ETS_C[] = {37.0 / 378.0,  0.0,
                                             250.0 / 621.0, 125.0 / 594.0,
                                             0.0,           512.0 / 1771.0};
        static const DendroScalar ETS_T[] = {0.0,       1.0 / 5.0, 3.0 / 10.0,
                                             3.0 / 5.0, 1.0,       7.0 / 8.0};
        // clang-format off
        static const DendroScalar ETS_U[] = {
            // stage 1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // stage 2
            1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // stage 3
            3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0,
            // stage 4
            3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0, 0.0,
            // stage 5
            -11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0, 0.0,
            // stage 6
            1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0,
            44275.0 / 110592.0, 253.0 / 4096.0, 0.0};
        // clang-format on

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK45_CASH_KARP");

    } else if (type == ETSType::RKF45) {
        // Runge-Kutta-Fehlberg 4(5) (6 stages); 5th-order weights used here.
        // From the `derivatives` experiment.
        nstages                           = 6;
        static const DendroScalar ETS_C[] = {
            16.0 / 135.0,      0.0,         6656.0 / 12825.0,
            28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};
        static const DendroScalar ETS_T[] = {0.0,         1.0 / 4.0, 3.0 / 8.0,
                                             12.0 / 13.0, 1.0,       1.0 / 2.0};
        // clang-format off
        static const DendroScalar ETS_U[] = {
            // stage 1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // stage 2
            1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // stage 3
            3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0,
            // stage 4
            1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0,
            // stage 5
            439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0,
            // stage 6
            -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0,
            0.0};
        // clang-format on

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RKF45");

    } else if (type == ETSType::RK6) {
        // Luther's 1968 6th-order method (7 stages — the minimum for order 6).
        // decimals are the exact q=sqrt(21) coefficients (NRPy+ table), same
        // style as RK4_RALSTON. c = [0,1,1/2,2/3,(7-q)/14,(7+q)/14,1].
        nstages                           = 7;

        static const DendroScalar ETS_C[] = {
            0.050000000000000003, 0, 0.35555555555555557, 0,
            0.2722222222222222,  0.2722222222222222, 0.050000000000000003};
        static const DendroScalar ETS_T[] = {
            0, 1, 0.5, 0.66666666666666663,
            0.17267316464601143, 0.82732683535398854, 1};
        // clang-format off
        static const DendroScalar ETS_U[] = {
            // stage 1
            0, 0, 0, 0, 0, 0, 0,
            // stage 2  a21 = 1
            1, 0, 0, 0, 0, 0, 0,
            // stage 3  a31 = 3/8, a32 = 1/8
            0.375, 0.125, 0, 0, 0, 0, 0,
            // stage 4  a41 = 8/27, a42 = 2/27, a43 = 8/27
            0.29629629629629628, 0.07407407407407407, 0.29629629629629628, 0, 0, 0, 0,
            // stage 5  (-21+9q)/392, (-56+8q)/392, (336-48q)/392, (-63+3q)/392
            0.051640768506639186, -0.049335189898860411, 0.29601113939316243, -0.1256435533549298, 0, 0, 0,
            // stage 6  (-1155-255q)/1960, (-280-40q)/1960, (-320q)/1960, (63+363q)/1960, (2352+392q)/1960
            -1.1854881643947648, -0.23637909581542529, -0.74817562366625956, 0.88085458023927043, 2.1165151389911681, 0, 0,
            // stage 7  (330+105q)/180, 2/3, (-200+280q)/180, (126-189q)/180, (-686-126q)/180, (490-70q)/180
            4.50650248872424, 0.66666666666666663, 6.0173399699313057, -4.111704479703632, -7.0189140975801996, 0.94010945196161799, 0};
        // clang-format on

        nodes   = ETS_T;
        weights = ETS_C;
        matrix  = ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK6 (Luther, 7 stages)");

    } else {
        dendro::logger::error(
            dendro::logger::Scope{"ETS"},
            "UNKNOWN ETS TYPE (supports RK3, RK4, RK5, RK4_RALSTON, "
            "RK5_NYSTROM, RK45_CASH_KARP, RKF45, RK6)");
        return -1;
    }

    return 0;
}

/**
 * @brief Fetch the coefficients for a Multistep Runge-Kutta (MSRK) variant.
 *
 * Single source of truth shared by ETS_MSRK (ets_msrk.h) and the RK
 * convergence/stability test. Coefficients from arXiv:2603.05763, Table 1.
 *
 * Unlike the single-step tableaux above, MSRK methods reuse RHS evaluations
 * from previous time steps. Stages [0, first_fresh_stage) are filled from a
 * history buffer (the aged base-point evaluations f(t_n, y_n)); the remaining
 * stages are fresh. All variants use 4 logical stages.
 *
 * @param[in]  type               RK4_MSRK2_1, RK4_MSRK2_2, or RK4_MSRK3
 * @param[out] aij                4x4 row-major stage matrix (caller array [16])
 * @param[out] b                  final weights b_i (caller array [4])
 * @param[out] c                  stage abscissae c_i (caller array [4])
 * @param[out] first_fresh_stage  index of the first fresh (non-history) stage
 * @param[out] num_history_slots  number of history slots (= first_fresh_stage)
 * @return 0 on success, -1 for a non-MSRK / unknown type.
 */
inline int get_msrk_tableau(ETSType type, DendroScalar* aij, DendroScalar* b,
                            DendroScalar* c, unsigned int& first_fresh_stage,
                            unsigned int& num_history_slots) {
    for (int i = 0; i < 16; i++) aij[i] = 0.0;
    for (int i = 0; i < 4; i++) {
        b[i] = 0.0;
        c[i] = 0.0;
    }

    if (type == ETSType::RK4_MSRK2_1) {
        // RK4-2(1): 2-step, 3 fresh evals/step. arXiv:2603.05763 Eq. 9-13.
        num_history_slots = 1;
        first_fresh_stage = 1;

        b[0]              = -643.0 / 1536.0;
        b[1]              = -4237.0 / 1092.0;
        b[2]              = 38125.0 / 10752.0;
        b[3]              = 4375.0 / 2496.0;

        c[2]              = 7.0 / 25.0;
        c[3]              = -13.0 / 25.0;

        aij[2 * 4 + 0]    = -49.0 / 1250.0;
        aij[2 * 4 + 1]    = 399.0 / 1250.0;
        aij[3 * 4 + 0]    = 7033.0 / 960000.0;
        aij[3 * 4 + 1]    = -217633.0 / 210000.0;
        aij[3 * 4 + 2]    = 5473.0 / 10752.0;

        dendro::logger::debug(dendro::logger::Scope{"ETS_MSRK"},
                              "Coefficients set for RK4-2(1)");

    } else if (type == ETSType::RK4_MSRK2_2) {
        // RK4-2(2): 2-step, 3 fresh evals/step. arXiv:2603.05763 Eq. 9-13.
        num_history_slots = 1;
        first_fresh_stage = 1;

        b[0]              = -191.0 / 882.0;
        b[1]              = 48241.0 / 59994.0;
        b[2]              = 193750.0 / 4351347.0;
        b[3]              = 100000.0 / 271791.0;

        c[2]              = -99.0 / 50.0;
        c[3]              = 101.0 / 100.0;

        aij[2 * 4 + 0]    = 1309.0 / 15500.0;
        aij[2 * 4 + 1]    = -31999.0 / 15500.0;
        aij[3 * 4 + 0]    = -241289.0 / 5880000.0;
        aij[3 * 4 + 1]    = 22846301.0 / 16170000.0;
        aij[3 * 4 + 2]    = -936169.0 / 2587200.0;

        dendro::logger::debug(dendro::logger::Scope{"ETS_MSRK"},
                              "Coefficients set for RK4-2(2)");

    } else if (type == ETSType::RK4_MSRK3) {
        // RK4-3: 3-step, 2 fresh evals/step. arXiv:2603.05763 Eq. 14-18.
        num_history_slots = 2;
        first_fresh_stage = 2;

        b[0]              = -85.0 / 1416.0;
        b[1]              = 131.0 / 408.0;
        b[2]              = -29.0 / 24.0;
        b[3]              = 15625.0 / 8024.0;

        c[3]              = 9.0 / 25.0;

        aij[3 * 4 + 0]    = 2511.0 / 62500.0;
        aij[3 * 4 + 1]    = -2268.0 / 15625.0;
        aij[3 * 4 + 2]    = 29061.0 / 62500.0;

        dendro::logger::debug(dendro::logger::Scope{"ETS_MSRK"},
                              "Coefficients set for RK4-3");

    } else {
        dendro::logger::error(
            dendro::logger::Scope{"ETS_MSRK"},
            "Invalid MSRK type. Use RK4_MSRK2_1, RK4_MSRK2_2, or RK4_MSRK3.");
        return -1;
    }

    return 0;
}

}  // namespace ts
